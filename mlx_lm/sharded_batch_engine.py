"""Continuous batching over a prepared sharded cache.

Questions can join a running batch at any step and leave when they finish. Every
machine must call ``add`` and ``step`` in the same order with the same arguments.
The choice of the next token is greedy, so all machines get the same tokens.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import mlx.core as mx

from .models.cache import BatchRotatingKVCache, RotatingKVCache
from .models.sharded_cache import ShardedKVCache
from .sharded_prompt_cache import prefill_question


@dataclass
class Event:
    """One step result for one question. ``token`` is None if the question ended
    without a new token. ``finish`` is "stop", "length" or None while it goes on."""

    uid: Any
    token: Optional[int]
    finish: Optional[str]


@dataclass
class _Slot:
    uid: Any
    pending: int
    max_tokens: int
    produced: int = 0


class BatchEngine:
    def __init__(
        self,
        model,
        caches: List[Any],
        base: int,
        stop_tokens: Iterable[int] = (),
        capacity: int = 1024,
        max_batch: int = 16,
    ):
        """``capacity`` is the most tokens (question and answer) one question can have."""
        for c in caches:
            if not isinstance(c, (ShardedKVCache, RotatingKVCache)):
                raise NotImplementedError(f"batch mode does not support {type(c).__name__}")
        self.model = model
        self.caches = caches
        self.base = base
        self.stop_tokens = set(stop_tokens)
        self.capacity = capacity
        self.max_batch = max_batch
        self.sharded = [c for c in caches if isinstance(c, ShardedKVCache)]
        self.slots: List[_Slot] = []
        self.batch_caches: Optional[List[Any]] = None

    def __len__(self):
        return len(self.slots)

    def add(self, uid, question: List[int], max_tokens: int):
        """Prefill a question alone and put it into the batch."""
        if len(question) + max_tokens > self.capacity:
            raise ValueError("the question and its answer do not fit in the capacity")
        if len(self.slots) >= self.max_batch:
            raise ValueError("the batch is full")
        with ExitStack() as stack:
            for c in self.sharded:
                stack.enter_context(c.paused_batch())
            logits, own, windows = prefill_question(
                self.model, self.caches, self.base, question
            )
        first = mx.argmax(logits, axis=-1).astype(mx.int32)
        mx.eval(first)

        if not self.slots:
            for c, pair in zip(self.sharded, own):
                c.start_batch(self.base, [pair], [len(question)], self.capacity)
            self.batch_caches = []
            w = 0
            for c in self.caches:
                if isinstance(c, ShardedKVCache):
                    self.batch_caches.append(c)
                else:
                    single = RotatingKVCache.from_state(windows[w])
                    self.batch_caches.append(BatchRotatingKVCache.merge([single]))
                    w += 1
        else:
            j = w = 0
            for c in self.batch_caches:
                if isinstance(c, ShardedKVCache):
                    c.extend_batch(own[j], len(question))
                    j += 1
                else:
                    single = RotatingKVCache.from_state(windows[w])
                    c.extend(BatchRotatingKVCache.merge([single]))
                    w += 1
        self.slots.append(_Slot(uid, int(first.item()), max_tokens))

    def step(self) -> List[Event]:
        """Give every question its next token and run one decoding pass."""
        events, keep = [], []
        for i, slot in enumerate(self.slots):
            if slot.pending in self.stop_tokens:
                events.append(Event(slot.uid, None, "stop"))
                continue
            slot.produced += 1
            done = slot.produced >= slot.max_tokens
            events.append(Event(slot.uid, slot.pending, "length" if done else None))
            if not done:
                keep.append(i)

        if len(keep) < len(self.slots):
            self._keep(keep)
        if self.slots:
            tokens = mx.array([[s.pending] for s in self.slots], dtype=mx.int32)
            logits = self.model(tokens, cache=self.batch_caches)[:, -1, :]
            nxt = mx.argmax(logits, axis=-1).astype(mx.int32)
            mx.eval(nxt)  # evaluate every pass before the next one
            for slot, token in zip(self.slots, nxt.tolist()):
                slot.pending = token
        return events

    def cancel(self, uid):
        """Drop one running question, if it is still running."""
        keep = [i for i, s in enumerate(self.slots) if s.uid != uid]
        if len(keep) < len(self.slots):
            self._keep(keep)

    def _keep(self, keep: List[int]):
        if not keep:
            self.close()
            return
        index = mx.array(keep)
        for c in self.batch_caches:
            if isinstance(c, ShardedKVCache):
                c.filter_batch(keep)
            else:
                c.filter(index)
        self.slots = [self.slots[i] for i in keep]

    def close(self):
        """Drop all running questions. The prepared cache stays as it was."""
        for c in self.sharded:
            c.end_batch()
        self.batch_caches = None
        self.slots = []
