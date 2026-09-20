"""OpenAI-compatible server for questions about one prepared, sharded text.

Prepare the text first (see mlx_lm/examples/sharded_context.py), then start this
server on every machine, for example:

    mlx.launch --hostfile hosts.json --backend ring -- \
        python -m mlx_lm.sharded_server --model MODEL --cache-dir DIR

Machine 0 answers HTTP requests. The last user message of a request is the
question, and everything else in ``messages`` is ignored. Answers are greedy,
so ``temperature`` and similar options have no effect. Questions that arrive
while others run join the running batch at the next step.
"""

import argparse
import json
import logging
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

import mlx.core as mx

from . import load
from .sharded_batch_engine import BatchEngine
from .sharded_prompt_cache import load_sharded_cache
from .sharded_scheduler import Scheduler

END_TOKENS = ("<end_of_turn>", "<turn|>", "<|eot_id|>", "<|im_end|>")
MAX_QUEUE = 64


def stop_token_ids(tokenizer):
    stop = set(tokenizer.eos_token_ids)
    for name in END_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(name)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop.add(token_id)
    return stop


def last_user_text(messages) -> Optional[str]:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            return "".join(parts)
    return None


class Answer:
    """Turns the token events of one request into text pieces."""

    def __init__(self, tokenizer, stops: List[str]):
        self.detokenizer = tokenizer.detokenizer
        self.stops = stops
        self.hold = max(0, max((len(s) for s in stops), default=0) - 1)
        self.text = ""
        self.sent = 0
        self.finish = None
        self.by_text = False  # ended by a stop string, so the engine must be told
        self.tokens = 0

    def add(self, event):
        """Returns the new text that is safe to send."""
        if event.token is not None:
            self.tokens += 1
            self.detokenizer.add_token(event.token)
            self.text += self.detokenizer.last_segment
        if event.finish:
            self.detokenizer.finalize()
            self.text += self.detokenizer.last_segment
            self.finish = event.finish
        for stop in self.stops:
            at = self.text.find(stop)
            if at >= 0:
                self.text = self.text[:at]
                self.by_text = event.finish is None
                self.finish = "stop"
                break
        safe = len(self.text) if self.finish else max(self.sent, len(self.text) - self.hold)
        piece = self.text[self.sent : safe]
        self.sent = safe
        return piece


def make_handler(scheduler, tokenizer, tail, base, model_name, default_max_tokens):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logging.debug("%s - %s", self.address_string(), fmt % args)

        def send_json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def error(self, status, message):
            self.send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/health"):
                self.send_json(200, {"status": "ok"})
            elif self.path.startswith("/v1/models"):
                self.send_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": model_name, "object": "model", "created": 0, "owned_by": "mlx-lm"}],
                    },
                )
            else:
                self.error(404, "not found")

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self.error(404, "not found")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                question = last_user_text(body["messages"])
                if not question:
                    raise ValueError("there is no user message")
                max_tokens = int(
                    body.get("max_completion_tokens") or body.get("max_tokens") or default_max_tokens
                )
                if max_tokens < 1:
                    raise ValueError("max_tokens must be at least 1")
                stops = body.get("stop") or []
                stops = [stops] if isinstance(stops, str) else list(stops)
                if len(stops) > 4 or not all(isinstance(s, str) and s for s in stops):
                    raise ValueError("stop must be up to 4 non-empty strings")
            except (KeyError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
                return self.error(400, f"bad request: {error}")

            if scheduler.waiting() >= MAX_QUEUE:
                return self.error(503, "too many waiting requests")
            events = queue.Queue()
            ids = tokenizer.encode(" " + question + tail, add_special_tokens=False)
            try:
                uid = scheduler.submit(ids, max_tokens, events.put)
            except ValueError as error:
                return self.error(400, str(error))

            answer = Answer(tokenizer, stops)
            stream = bool(body.get("stream"))
            prompt_tokens = base + len(ids)
            request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())

            def chunk(delta, finish=None):
                return {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }

            def usage():
                return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": answer.tokens,
                    "total_tokens": prompt_tokens + answer.tokens,
                    "prompt_tokens_details": {"cached_tokens": base},
                }

            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.write_event(chunk({"role": "assistant", "content": ""}))
                while answer.finish is None:
                    piece = answer.add(events.get())
                    if stream and piece:
                        self.write_event(chunk({"content": piece}))
                if answer.by_text:
                    scheduler.cancel(uid)
                if stream:
                    self.write_event(chunk({}, answer.finish))
                    if (body.get("stream_options") or {}).get("include_usage"):
                        self.write_event(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [],
                                "usage": usage(),
                            }
                        )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self.send_json(
                        200,
                        {
                            "id": request_id,
                            "object": "chat.completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": answer.text},
                                    "finish_reason": answer.finish,
                                }
                            ],
                            "usage": usage(),
                        },
                    )
            except (BrokenPipeError, ConnectionResetError):
                scheduler.cancel(uid)

        def write_event(self, obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

    return Handler


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for a sharded context")
    parser.add_argument("--model", required=True, help="HF repo or path to local model.")
    parser.add_argument("--cache-dir", required=True, help="Folder with this machine's shard.")
    parser.add_argument("--host", default="127.0.0.1", help="Address of machine 0 to listen on.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-batch", type=int, default=8, help="Most questions answered together.")
    parser.add_argument(
        "--capacity", type=int, default=1024, help="Most tokens of one question and its answer."
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Default answer length.")
    parser.add_argument("--served-model-name", help="Name shown to clients (default: --model).")
    parser.add_argument("--backend", default="ring", help="Distributed backend.")
    args = parser.parse_args()

    group = mx.distributed.init(backend=args.backend)
    model, tokenizer = load(args.model)
    caches, meta = load_sharded_cache(os.path.join(args.cache_dir, "context"), group)
    base, tail = meta["total_tokens"], meta["extra"]["tail"]
    engine = BatchEngine(
        model,
        caches,
        base,
        stop_tokens=stop_token_ids(tokenizer),
        capacity=args.capacity,
        max_batch=args.max_batch,
    )
    scheduler = Scheduler(engine, group)

    server = None
    if group.rank() == 0:
        handler = make_handler(
            scheduler, tokenizer, tail, base, args.served_model_name or args.model, args.max_tokens
        )
        server = ThreadingHTTPServer((args.host, args.port), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Serving {base} cached tokens on http://{args.host}:{args.port}", flush=True)
    try:
        scheduler.run()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.shutdown()


if __name__ == "__main__":
    main()
