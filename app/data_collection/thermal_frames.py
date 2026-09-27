"""Keep the latest decoded thermal matrix available to in-process analysis."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    sequence: int
    device_index: int
    width: int
    height: int
    slope: int
    offset: int
    camera_timestamp: int
    received_at: float
    raw: np.ndarray

    @property
    def celsius(self) -> np.ndarray:
        return self.raw.astype(np.float32) / self.slope + self.offset


class ThermalFrameStore:
    """Bounded handoff: processing code can read the newest complete frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: ThermalFrame | None = None
        self._received = 0

    def ingest(
        self, payload: bytes, *, sequence: int, device_index: int, width: int,
        height: int, slope: int, offset: int, camera_timestamp: int,
    ) -> dict:
        if width <= 0 or height <= 0 or width * height > 1_000_000:
            raise ValueError("无效的温度矩阵尺寸")
        if slope <= 0 or len(payload) != width * height * 2:
            raise ValueError("温度负载长度或 slope 无效")
        received_at = time.time()
        matrix = np.frombuffer(payload, dtype="<i2").reshape(height, width).copy()
        frame = ThermalFrame(
            sequence, device_index, width, height, slope, offset,
            camera_timestamp, received_at, matrix,
        )
        raw_min = int(matrix.min())
        raw_max = int(matrix.max())
        raw_mean = float(matrix.mean())
        with self._lock:
            self._latest = frame
            self._received += 1
            received = self._received
        return {
            "sequence": sequence,
            "device_index": device_index,
            "width": width,
            "height": height,
            "camera_timestamp": camera_timestamp,
            "received_at": received_at,
            "minimum_celsius": raw_min / slope + offset,
            "maximum_celsius": raw_max / slope + offset,
            "average_celsius": raw_mean / slope + offset,
            "received_frames": received,
        }

    def latest(self) -> ThermalFrame | None:
        with self._lock:
            return self._latest
