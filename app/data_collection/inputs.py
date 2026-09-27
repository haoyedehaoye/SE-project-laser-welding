"""Push-based collection boundary for devices whose final protocol is not selected yet."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.core.models import DataMode, DeviceEvent, EventType, SourceKind, capture_timestamp


class InputChannels:
    """Normalize thermal-camera and robot-controller payloads into device events."""

    def __init__(self, publish: Callable[[DeviceEvent], object]) -> None:
        self._publish = publish

    def temperature_matrix(
        self, matrix: list[list[float]], captured_at: float | None = None, device_id: str = "thermal-01"
    ) -> object:
        event = DeviceEvent(
            device_id=device_id,
            source=SourceKind.THERMAL,
            event_type=EventType.FRAME,
            captured_at=capture_timestamp(captured_at),
            data={"temperature_matrix": matrix},
            units={"temperature_matrix": "degC"},
            mode=DataMode.REAL,
            metadata={"timestamp_origin": "device_unix_seconds" if captured_at is not None else "host_received"},
        )
        return self._publish(event)

    def temperature_summary(self, summary: dict[str, Any], device_id: str = "thermal-01") -> object:
        event = DeviceEvent(
            device_id=device_id,
            source=SourceKind.THERMAL,
            event_type=EventType.FRAME,
            captured_at=float(summary["received_at"]),
            data=summary,
            units={"minimum_celsius": "degC", "maximum_celsius": "degC", "average_celsius": "degC"},
            mode=DataMode.REAL,
            metadata={
                "timestamp_origin": "host_received",
                "camera_timestamp_raw": summary["camera_timestamp"],
            },
        )
        return self._publish(event)

    def robot_motion(
        self,
        speed: float,
        elapsed_time: float,
        captured_at: float | None = None,
        device_id: str = "robot-01",
        extra: dict[str, Any] | None = None,
    ) -> object:
        event = DeviceEvent(
            device_id=device_id,
            source=SourceKind.ROBOT,
            event_type=EventType.TELEMETRY,
            captured_at=capture_timestamp(captured_at),
            data={"speed": speed, "elapsed_time": elapsed_time, **(extra or {})},
            units={"speed": "m/min", "elapsed_time": "s"},
            mode=DataMode.REAL,
            metadata={"timestamp_origin": "device_unix_seconds" if captured_at is not None else "host_received"},
        )
        return self._publish(event)
