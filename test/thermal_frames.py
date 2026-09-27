"""Latest thermal matrix for the TEST dashboard; no recording or STM32 mixing."""

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


class ThermalFrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: ThermalFrame | None = None
        self._received = 0
        self._sequence_gaps = 0
        self._first_at: float | None = None

    def ingest(
        self, payload: bytes, *, sequence: int, device_index: int, width: int,
        height: int, slope: int, offset: int, camera_timestamp: int,
    ) -> dict:
        if not 0 <= sequence <= 2**63 - 1 or not 0 <= device_index <= 65535:
            raise ValueError("无效的帧号或设备序号")
        if width <= 0 or height <= 0 or width * height > 1_000_000:
            raise ValueError("无效的温度矩阵尺寸")
        if slope <= 0 or len(payload) != width * height * 2:
            raise ValueError("温度负载长度或 slope 无效")
        raw = np.frombuffer(payload, dtype="<i2").reshape(height, width).copy()
        now = time.time()
        frame = ThermalFrame(sequence, device_index, width, height, slope, offset,
                             camera_timestamp, now, raw)
        with self._lock:
            previous = self._latest
            if previous and sequence > previous.sequence + 1:
                self._sequence_gaps += sequence - previous.sequence - 1
            self._latest = frame
            self._received += 1
            if self._first_at is None:
                self._first_at = now
            count = self._received
            gaps = self._sequence_gaps
        return {"received_sequence": sequence, "received_frames": count,
                "sequence_gaps": gaps}

    def latest(self) -> ThermalFrame | None:
        with self._lock:
            return self._latest

    def snapshot(self, max_width: int = 80, max_height: int = 64) -> dict:
        with self._lock:
            frame = self._latest
            count = self._received
            gaps = self._sequence_gaps
            first_at = self._first_at
        if frame is None:
            return {"connected": False, "received_frames": count,
                    "sequence_gaps": gaps, "frame": None}
        raw = frame.raw
        grid_width = min(frame.width, max_width)
        grid_height = min(frame.height, max_height)
        x = np.linspace(0, frame.width - 1, grid_width, dtype=np.intp)
        y = np.linspace(0, frame.height - 1, grid_height, dtype=np.intp)
        grid = np.round(raw[np.ix_(y, x)].astype(np.float32) / frame.slope + frame.offset, 2)
        age = max(0.0, time.time() - frame.received_at)
        return {
            "connected": age < 3.0,
            "received_frames": count,
            "sequence_gaps": gaps,
            "average_fps": round((count - 1) / max(frame.received_at - first_at, 0.001), 2)
            if first_at is not None and count > 1 else 0.0,
            "frame": {
                "sequence": frame.sequence,
                "device_index": frame.device_index,
                "width": frame.width,
                "height": frame.height,
                "camera_timestamp": frame.camera_timestamp,
                "received_at": frame.received_at,
                "age_seconds": round(age, 2),
                "minimum_celsius": round(float(raw.min()) / frame.slope + frame.offset, 2),
                "maximum_celsius": round(float(raw.max()) / frame.slope + frame.offset, 2),
                "average_celsius": round(float(raw.mean()) / frame.slope + frame.offset, 2),
                "grid": grid.tolist(),
            },
        }
