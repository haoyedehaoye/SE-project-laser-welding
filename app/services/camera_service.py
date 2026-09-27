"""Resilient background camera acquisition and browser-preview service."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from app.config.settings import CameraSettings
from app.core.models import DataMode, DeviceEvent, DeviceStatus, EventType, SourceKind, utc_timestamp
from app.core.service import ManagedService
from app.devices.camera.base import CameraAdapter, CameraDevice
from app.devices.camera.hikrobot import HikrobotMvsCamera
from app.services.png import encode_png

logger = logging.getLogger("camera.service")


class CameraService(ManagedService):
    def __init__(
        self,
        settings: CameraSettings,
        adapter_factory: Callable[[], CameraAdapter] | None = None,
        publish: Callable[[DeviceEvent], object] | None = None,
    ) -> None:
        self.settings = settings
        self._publish = publish
        self._factory = adapter_factory or (
            lambda: HikrobotMvsCamera(settings.mvs_python_path, settings.packet_size_auto)
        )
        self._adapter: CameraAdapter | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._latest_png: bytes | None = None
        self._sequence = 0
        self._frame_count = 0
        self._started_at = 0.0
        self._last_error = ""
        self._status = DeviceStatus(
            device_id=settings.device_id,
            source=SourceKind.CAMERA,
            state="stopped",
            message="相机服务未启动",
        )

    @property
    def name(self) -> str:
        return "camera"

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running.is_set():
                return
            if not self.settings.enabled:
                self._set_status("disabled", False, "相机已在配置中禁用")
                return
            with self._frame_ready:
                self._latest_png = None
                self._frame_count = 0
                self._last_error = ""
            self._running.set()
            self._started_at = time.monotonic()
            self._set_status("starting", False, "正在启动相机服务")
            self._thread = threading.Thread(
                target=self._run, name="camera-acquisition", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._running.clear()
            with self._frame_ready:
                self._latest_png = None
                self._sequence += 1
                self._frame_ready.notify_all()
            if self._thread:
                self._thread.join(timeout=max(3.0, self.settings.frame_timeout_ms / 1000 + 2.0))
            self._thread = None
            self._close_adapter()
            self._set_status("stopped", False, "相机服务已停止")

    def status(self) -> dict:
        with self._lock:
            result = self._status.to_dict()
            elapsed = max(time.monotonic() - self._started_at, 0.001) if self._started_at else 0
            result["details"] = {
                **result["details"],
                "requested_on": self._running.is_set(),
                "frames_received": self._frame_count,
                "average_fps": round(self._frame_count / elapsed, 2) if elapsed else 0,
                "preview_available": self._latest_png is not None,
                "last_error": self._last_error,
            }
            return result

    def discover(self) -> list[dict]:
        adapter = self._factory()
        try:
            return [device.to_dict() for device in adapter.enumerate_devices()]
        finally:
            adapter.close()

    def wait_for_frame(self, after_sequence: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._sequence > after_sequence or not self._running.is_set(), timeout=timeout
            )
            return self._sequence, self._latest_png

    def _run(self) -> None:
        while self._running.is_set():
            try:
                self._set_status("connecting", False, "正在查找并连接工业相机")
                adapter = self._factory()
                self._adapter = adapter
                device = adapter.open(self.settings.serial_number, self.settings.ip_address)
                self._last_error = ""
                self._set_status("streaming", True, "工业相机已连接", device)
                logger.info(
                    "相机已连接 model=%s serial=%s ip=%s",
                    device.model, device.serial_number, device.ip_address,
                )
                self._capture_loop(adapter)
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                if message != self._last_error:
                    logger.error("相机不可用: %s", message)
                self._last_error = message
                self._set_status("error", False, message)
            finally:
                self._close_adapter()
            deadline = time.monotonic() + self.settings.reconnect_interval_seconds
            while self._running.is_set() and time.monotonic() < deadline:
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _capture_loop(self, adapter: CameraAdapter) -> None:
        publish_interval = 1.0 / max(self.settings.preview_max_fps, 0.1)
        last_published = 0.0
        while self._running.is_set():
            frame = adapter.read(self.settings.frame_timeout_ms)
            if frame is None:
                continue
            now = time.monotonic()
            if now - last_published < publish_interval:
                continue
            png = encode_png(
                frame.width,
                frame.height,
                frame.channels,
                frame.data,
                self.settings.png_compression,
            )
            received_at = time.time()
            with self._frame_ready:
                self._latest_png = png
                self._sequence += 1
                self._frame_count += 1
                self._status.last_seen_at = frame.captured_at
                self._status.details["width"] = frame.width
                self._status.details["height"] = frame.height
                self._status.details["frame_number"] = frame.frame_number
                self._status.details["captured_at_utc"] = utc_timestamp(frame.captured_at)
                self._status.details["received_at_utc"] = utc_timestamp(received_at)
                self._frame_ready.notify_all()
            if self._publish is not None:
                event = DeviceEvent(
                    device_id=self.settings.device_id,
                    source=SourceKind.CAMERA,
                    event_type=EventType.FRAME,
                    captured_at=frame.captured_at,
                    received_at=received_at,
                    data={
                        "frame_number": frame.frame_number,
                        "width": frame.width,
                        "height": frame.height,
                    },
                    mode=DataMode.REAL,
                    metadata={"timestamp_origin": "host_frame_read"},
                )
                try:
                    self._publish(event)
                except Exception:
                    logger.exception("发布相机帧时间戳失败")
            last_published = now

    def _set_status(
        self,
        state: str,
        connected: bool,
        message: str,
        device: CameraDevice | None = None,
    ) -> None:
        with self._lock:
            details = dict(self._status.details)
            if device:
                details["device"] = device.to_dict()
            self._status.state = state
            self._status.connected = connected
            self._status.message = message
            self._status.details = details

    def _close_adapter(self) -> None:
        adapter, self._adapter = self._adapter, None
        if adapter:
            try:
                adapter.close()
            except Exception:
                logger.exception("关闭相机时发生异常")
