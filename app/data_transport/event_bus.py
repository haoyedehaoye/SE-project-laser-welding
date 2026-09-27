"""Thread-safe in-process event transport used by collectors and WebSockets."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from app.core.models import DeviceEvent


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    sequence: int
    event: DeviceEvent
    quality: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "event": self.event.to_dict(), "quality": self.quality}


class EventBus:
    """Small bounded bus; persistence can later subscribe without changing collectors."""

    def __init__(self, history_size: int = 2000) -> None:
        self._items: deque[EventEnvelope] = deque(maxlen=history_size)
        self._sequence = 0
        self._condition = threading.Condition()

    def publish(self, event: DeviceEvent, quality: dict[str, Any]) -> EventEnvelope:
        with self._condition:
            self._sequence += 1
            envelope = EventEnvelope(self._sequence, event, quality)
            self._items.append(envelope)
            self._condition.notify_all()
            return envelope

    def latest(self, source: str | None = None) -> dict[str, Any] | None:
        with self._condition:
            for item in reversed(self._items):
                if source is None or item.event.source.value == source:
                    return item.to_dict()
        return None

    def snapshot(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._condition:
            return [item.to_dict() for item in list(self._items)[-max(1, min(limit, 1000)):]]

    def wait_after(self, sequence: int, timeout: float = 5.0) -> list[dict[str, Any]]:
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > sequence, timeout=timeout)
            return [item.to_dict() for item in self._items if item.sequence > sequence]

