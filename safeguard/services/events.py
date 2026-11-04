"""Tiny in-process pub/sub used to push alerts and telemetry to browsers over
Server-Sent Events. No extra dependency, works with any WSGI server that streams."""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Iterator


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def publish(self, event: str, data: dict) -> None:
        payload = (event, data)
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:  # slow consumer: drop rather than block the pipeline
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def stream(self, keepalive: float = 15.0) -> Iterator[str]:
        q = self.subscribe()
        try:
            yield "retry: 3000\n\n"
            last = time.monotonic()
            while True:
                try:
                    event, data = q.get(timeout=1.0)
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                    last = time.monotonic()
                except queue.Empty:
                    if time.monotonic() - last > keepalive:
                        yield ": keepalive\n\n"
                        last = time.monotonic()
        finally:
            self.unsubscribe(q)


bus = EventBus()
