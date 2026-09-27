"""Stable data contracts shared by devices, services, storage and APIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from time import time
from typing import Any


class SourceKind(StrEnum):
    CAMERA = "camera"
    STM32 = "stm32"
    THERMAL = "thermal"
    ROBOT = "robot"
    SYSTEM = "system"


class EventType(StrEnum):
    TELEMETRY = "telemetry"
    STATUS = "status"
    ALARM = "alarm"
    FRAME = "frame"


class DataMode(StrEnum):
    REAL = "real"
    SIMULATED = "simulated"


class Validity(StrEnum):
    VALID = "valid"
    STALE = "stale"
    INVALID = "invalid"


def utc_timestamp(value: float | None) -> str | None:
    """Render Unix seconds as an unambiguous UTC timestamp for API clients."""
    if value is None or not isfinite(value):
        return None
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
    except (OverflowError, OSError, ValueError):
        return None


def capture_timestamp(value: Any | None) -> float:
    """Accept a device Unix timestamp in seconds, or mark host receipt if absent."""
    stamp = time() if value is None else float(value)
    if utc_timestamp(stamp) is None:
        raise ValueError("captured_at 必须是有效的 Unix 秒时间戳")
    return stamp


@dataclass(slots=True)
class DeviceEvent:
    """One normalized event emitted by any connected device."""

    device_id: str
    source: SourceKind
    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    captured_at: float = field(default_factory=time)
    received_at: float = field(default_factory=time)
    mode: DataMode = DataMode.REAL
    validity: Validity = Validity.VALID
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["captured_at_utc"] = utc_timestamp(self.captured_at)
        result["received_at_utc"] = utc_timestamp(self.received_at)
        return result


@dataclass(slots=True)
class DeviceStatus:
    device_id: str
    source: SourceKind
    state: str = "stopped"
    connected: bool = False
    message: str = ""
    last_seen_at: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["last_seen_at_utc"] = utc_timestamp(self.last_seen_at)
        return result
