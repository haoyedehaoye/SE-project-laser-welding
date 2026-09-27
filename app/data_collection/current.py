"""STM32 voltage/current acquisition with a protocol parser independent of transport."""

from __future__ import annotations

import logging
import math
import random
import struct
import threading
import time
from collections.abc import Callable

from app.config.settings import CurrentSettings
from app.core.models import DataMode, DeviceEvent, DeviceStatus, EventType, SourceKind
from app.core.service import ManagedService

logger = logging.getLogger("collection.current")


class CurrentFrameParser:
    """Parse AA + voltage(float32) + current(float32) + XOR + 55 frames."""

    HEADER, FOOTER, FRAME_LENGTH = 0xAA, 0x55, 11

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[tuple[float, float]]:
        self._buffer.extend(chunk)
        output: list[tuple[float, float]] = []
        while len(self._buffer) >= self.FRAME_LENGTH:
            try:
                header = self._buffer.index(self.HEADER)
            except ValueError:
                self._buffer.clear()
                break
            if header:
                del self._buffer[:header]
            if len(self._buffer) < self.FRAME_LENGTH:
                break
            raw = self._buffer[: self.FRAME_LENGTH]
            checksum = 0
            for value in raw[:9]:
                checksum ^= value
            if raw[10] != self.FOOTER or raw[9] != checksum:
                del self._buffer[0]
                continue
            voltage, current = struct.unpack("<ff", bytes(raw[1:9]))
            del self._buffer[: self.FRAME_LENGTH]
            if math.isfinite(voltage) and math.isfinite(current):
                output.append((voltage, current))
        return output


class CurrentTelemetryCollector(ManagedService):
    def __init__(self, settings: CurrentSettings, publish: Callable[[DeviceEvent], object]) -> None:
        self.settings = settings
        self._publish = publish
        self._parser = CurrentFrameParser()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._serial = None
        self._count = 0
        self._lock = threading.Lock()
        self._status = DeviceStatus(settings.device_id, SourceKind.STM32, message="电流采集未启动")

    @property
    def name(self) -> str:
        return "current_collection"

    def start(self) -> None:
        if not self.settings.enabled:
            self._set_status("disabled", False, "电流采集已在配置中禁用")
            return
        if self._running.is_set():
            return
        self._running.set()
        self._set_status("starting", False, "正在启动电流采集")
        self._thread = threading.Thread(target=self._run, name="current-collection", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=max(3.0, self.settings.timeout_seconds + 1.0))
        self._thread = None
        serial_port, self._serial = self._serial, None
        if serial_port:
            serial_port.close()
        self._set_status("stopped", False, "电流采集已停止")

    def status(self) -> dict:
        with self._lock:
            result = self._status.to_dict()
            result["details"] = {**result["details"], "samples_received": self._count}
            return result

    def publish(self, event: DeviceEvent) -> object:
        """Feed externally transported STM32 data through the same pipeline."""
        return self._publish(event)

    def _run(self) -> None:
        if self.settings.simulated:
            self._set_status("streaming", True, "电流模拟采集运行中")
            while self._running.is_set():
                now = time.time()
                self._emit(25 + random.gauss(0, .3), 200 + 10 * math.sin(now) + random.gauss(0, 2))
                time.sleep(1 / max(self.settings.sample_rate_hz, 1))
            return
        while self._running.is_set():
            try:
                import serial

                self._set_status("connecting", False, f"正在连接 {self.settings.port}")
                self._serial = serial.Serial(
                    self.settings.port,
                    self.settings.baudrate,
                    timeout=self.settings.timeout_seconds,
                )
                self._set_status("streaming", True, f"已连接 {self.settings.port}")
                while self._running.is_set():
                    chunk = self._serial.read(max(self._serial.in_waiting, 1))
                    for voltage, current in self._parser.feed(chunk):
                        self._emit(voltage, current)
            except Exception as exc:
                logger.error("电流采集不可用: %s", exc)
                self._set_status("error", False, str(exc))
            finally:
                serial_port, self._serial = self._serial, None
                if serial_port:
                    serial_port.close()
            deadline = time.monotonic() + self.settings.reconnect_interval_seconds
            while self._running.is_set() and time.monotonic() < deadline:
                time.sleep(min(0.1, deadline - time.monotonic()))

    def _emit(self, voltage: float, current: float) -> None:
        event = DeviceEvent(
            device_id=self.settings.device_id,
            source=SourceKind.STM32,
            event_type=EventType.TELEMETRY,
            data={"voltage": round(voltage, 4), "current": round(current, 4)},
            units={"voltage": "V", "current": "A"},
            mode=DataMode.SIMULATED if self.settings.simulated else DataMode.REAL,
            metadata={
                "timestamp_origin": "host_simulated" if self.settings.simulated else "host_frame_parsed"
            },
        )
        self._publish(event)
        with self._lock:
            self._count += 1
            self._status.last_seen_at = event.captured_at

    def _set_status(self, state: str, connected: bool, message: str) -> None:
        with self._lock:
            self._status.state = state
            self._status.connected = connected
            self._status.message = message
