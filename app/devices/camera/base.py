"""Camera adapter contract. Product code deliberately has no simulated camera."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class CameraDevice:
    index: int
    transport: str
    model: str
    serial_number: str
    ip_address: str = ""
    user_defined_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CameraFrame:
    width: int
    height: int
    channels: int
    data: bytes
    captured_at: float
    frame_number: int = 0


class CameraAdapter(ABC):
    @abstractmethod
    def enumerate_devices(self) -> list[CameraDevice]: ...

    @abstractmethod
    def open(self, serial_number: str = "", ip_address: str = "") -> CameraDevice: ...

    @abstractmethod
    def read(self, timeout_ms: int) -> CameraFrame | None: ...

    @abstractmethod
    def close(self) -> None: ...
