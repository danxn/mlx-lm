"""Run a BatchEngine on every machine while requests arrive on machine 0.

Machine 0 takes requests from other threads. At every step it tells all machines
which questions to add or drop, so they all do the same work in the same order.
"""

import threading
import time
from collections import deque
from typing import Callable, List

import mlx.core as mx

from .sharded_batch_engine import BatchEngine, Event


class Scheduler:
    def __init__(self, engine: BatchEngine, group, idle_sleep: float = 0.002):
        self.engine = engine
        self.group = group
        self.idle_sleep = idle_sleep
        self._lock = threading.Lock()
        self._pending = deque()
        self._cancels: List[int] = []
        self._callbacks = {}
        self._next_uid = 0
        self._stop = False

    def submit(self, question: List[int], max_tokens: int, on_event: Callable[[Event], None]):
        """Queue a question (machine 0 only). ``on_event`` runs on the scheduler thread."""
        if len(question) + max_tokens > self.engine.capacity:
            raise ValueError("the question and its answer are too long")
        with self._lock:
            uid = self._next_uid
            self._next_uid += 1
            self._pending.append((uid, question, max_tokens))
            self._callbacks[uid] = on_event
        return uid

    def cancel(self, uid: int):
        with self._lock:
            self._callbacks.pop(uid, None)
            waiting = [item for item in self._pending if item[0] == uid]
            if waiting:
                self._pending.remove(waiting[0])
            else:
                self._cancels.append(uid)

    def shutdown(self):
        with self._lock:
            self._stop = True

    def waiting(self):
        with self._lock:
            return len(self._pending)

    def _collect(self):
        """Machine 0: what to tell the others in this round, as a list of ints."""
        with self._lock:
            room = self.engine.max_batch - len(self.engine)
            adds = []
            while self._pending and len(adds) < room:
                adds.append(self._pending.popleft())
            cancels, self._cancels = self._cancels, []
            stop = self._stop
        message = [len(adds)]
        for uid, question, max_tokens in adds:
            message += [uid, max_tokens, len(question), *question]
        message += [len(cancels), *cancels]
        return stop, message

    def _broadcast(self, stop: bool, message: List[int]):
        """Send machine 0's message to all machines; returns (stop, adds, cancels)."""
        header = mx.array([int(stop), len(message)], dtype=mx.int32)
        if self.group.rank() != 0:
            header = mx.zeros((2,), dtype=mx.int32)
        header = mx.distributed.all_sum(header, group=self.group)
        mx.eval(header)
        stop, size = header.tolist()
        if stop:
            return True, [], []
        if size == 2:  # no question to add or drop
            return False, [], []
        payload = mx.array(message, dtype=mx.int32) if self.group.rank() == 0 else mx.zeros((size,), dtype=mx.int32)
        payload = mx.distributed.all_sum(payload, group=self.group)
        mx.eval(payload)
        values = payload.tolist()
        adds, at = [], 1
        for _ in range(values[0]):
            uid, max_tokens, length = values[at : at + 3]
            adds.append((uid, values[at + 3 : at + 3 + length], max_tokens))
            at += 3 + length
        cancels = values[at + 1 : at + 1 + values[at]]
        return False, adds, cancels

    def run(self):
        """Serve until ``shutdown``. Call it on every machine."""
        lead = self.group.rank() == 0
        while True:
            stop, message = self._collect() if lead else (False, [])
            stop, adds, cancels = self._broadcast(stop, message)
            if stop:
                break
            for uid in cancels:
                self.engine.cancel(uid)
            for uid, question, max_tokens in adds:
                self.engine.add(uid, question, max_tokens)
            if len(self.engine):
                for event in self.engine.step():
                    if lead:
                        self._dispatch(event)
            elif lead:
                time.sleep(self.idle_sleep)
        self.engine.close()

    def _dispatch(self, event: Event):
        callback = self._callbacks.get(event.uid)
        if callback is None:
            return
        if event.finish:
            del self._callbacks[event.uid]
        callback(event)
