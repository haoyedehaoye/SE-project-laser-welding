"""Test-stage Hikrobot camera acquisition and browser preview service."""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import socket
import struct
import sys
import threading
import time
import zlib
from pathlib import Path

logger = logging.getLogger("camera")
_DLL_HANDLES: list[object] = []
_PRELOADED_DLLS: list[object] = []


def _decode(value) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="ignore").strip()


def _ip(value: int) -> str:
    try:
        return socket.inet_ntoa(struct.pack(">I", int(value)))
    except (OSError, struct.error, ValueError):
        return ""


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload)) + kind + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_png(width: int, height: int, channels: int, pixels: bytes) -> bytes:
    stride = width * channels
    expected = stride * height
    if width <= 0 or height <= 0 or channels not in (1, 3) or len(pixels) < expected:
        raise ValueError("相机图像尺寸或缓冲区无效")
    rows = b"".join(
        b"\x00" + pixels[offset:offset + stride]
        for offset in range(0, expected, stride)
    )
    color_type = 0 if channels == 1 else 2
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, 3))
        + _png_chunk(b"IEND", b"")
    )


class HikCameraService:
    def __init__(
        self,
        mvs_python_path: str,
        serial_number: str = "",
        ip_address: str = "",
        fps: float = 15.0,
        frame_timeout_ms: int = 1000,
        reconnect_seconds: float = 3.0,
    ) -> None:
        self.mvs_python_path = mvs_python_path
        self.serial_number = serial_number
        self.ip_address = ip_address
        self.fps = fps
        self.frame_timeout_ms = frame_timeout_ms
        self.reconnect_seconds = reconnect_seconds
        self._running = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._camera = None
        self._latest_png: bytes | None = None
        self._sequence = 0
        self._frame_count = 0
        self._started_at = 0.0
        self._state = "stopped"
        self._message = "相机采集已停止"
        self._connected = False
        self._last_error = ""
        self._device: dict = {}
        self._width = 0
        self._height = 0

    def _load_sdk(self):
        candidates = [
            self.mvs_python_path,
            os.getenv("MVS_PYTHON_PATH", ""),
            r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
            r"C:\Program Files\MVS\Development\Samples\Python\MvImport",
            r"D:\海康机器人摄像头\MVS\Development\Samples\Python\MvImport",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_dir() and candidate not in sys.path:
                sys.path.insert(0, candidate)
        for dll_dir in (
            r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
            r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
        ):
            dll_path = Path(dll_dir)
            if hasattr(os, "add_dll_directory") and dll_path.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(dll_dir))
                dll = dll_path / "MvCameraControl.dll"
                if dll.exists():
                    _PRELOADED_DLLS.append(ctypes.WinDLL(str(dll), winmode=0))
        try:
            return importlib.import_module("MvCameraControl_class")
        except (ImportError, OSError) as exc:
            raise RuntimeError("未找到海康 MVS Python SDK，请检查 test/config.py 中的路径") from exc

    def _native_devices(self, mvs):
        devices = mvs.MV_CC_DEVICE_INFO_LIST()
        mask = mvs.MV_GIGE_DEVICE | getattr(mvs, "MV_USB_DEVICE", 0)
        result = mvs.MvCamera.MV_CC_EnumDevices(mask, devices)
        if result != 0:
            raise RuntimeError(f"MVS 枚举相机失败: 0x{result:08x}")
        return devices

    def _describe(self, mvs, native, index: int) -> dict:
        info = native.contents
        if info.nTLayerType & mvs.MV_GIGE_DEVICE:
            detail = info.SpecialInfo.stGigEInfo
            return {
                "index": index,
                "transport": "gige",
                "model": _decode(detail.chModelName),
                "serial_number": _decode(detail.chSerialNumber),
                "ip_address": _ip(detail.nCurrentIp),
            }
        detail = info.SpecialInfo.stUsb3VInfo
        return {
            "index": index,
            "transport": "usb3",
            "model": _decode(detail.chModelName),
            "serial_number": _decode(detail.chSerialNumber),
            "ip_address": "",
        }

    def discover(self) -> list[dict]:
        mvs = self._load_sdk()
        native = self._native_devices(mvs)
        return [self._describe(mvs, native.pDeviceInfo[i], i) for i in range(native.nDeviceNum)]

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running.is_set():
                return
            with self._frame_ready:
                self._latest_png = None
                self._frame_count = 0
                self._last_error = ""
            self._running.set()
            self._started_at = time.monotonic()
            self._set_status("starting", False, "正在启动工业相机")
            self._thread = threading.Thread(target=self._run, name="test-camera", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._running.clear()
            with self._frame_ready:
                self._latest_png = None
                self._sequence += 1
                self._frame_ready.notify_all()
            if self._thread:
                self._thread.join(timeout=max(3.0, self.frame_timeout_ms / 1000 + 2.0))
            self._thread = None
            self._close_camera()
            self._set_status("stopped", False, "相机采集已停止")

    def status(self) -> dict:
        with self._lock:
            elapsed = max(time.monotonic() - self._started_at, 0.001) if self._started_at else 0
            return {
                "enabled": True,
                "source_type": "hikrobot_mvs",
                "requested_on": self._running.is_set(),
                "connected": self._connected,
                "state": self._state,
                "message": self._message,
                "device": dict(self._device),
                "width": self._width,
                "height": self._height,
                "frames_received": self._frame_count,
                "average_fps": round(self._frame_count / elapsed, 2) if elapsed else 0,
                "preview_available": self._latest_png is not None,
                "last_error": self._last_error,
            }

    def wait_for_frame(self, sequence: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._sequence > sequence or not self._running.is_set(), timeout=timeout
            )
            return self._sequence, self._latest_png

    def _set_status(self, state: str, connected: bool, message: str) -> None:
        with self._lock:
            self._state = state
            self._connected = connected
            self._message = message

    def _run(self) -> None:
        while self._running.is_set():
            try:
                self._set_status("connecting", False, "正在查找并连接工业相机")
                self._capture()
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                if message != self._last_error:
                    logger.error("工业相机不可用: %s", message)
                self._last_error = message
                self._set_status("error", False, message)
            finally:
                self._close_camera()
            deadline = time.monotonic() + self.reconnect_seconds
            while self._running.is_set() and time.monotonic() < deadline:
                time.sleep(0.1)

    def _capture(self) -> None:
        mvs = self._load_sdk()
        native = self._native_devices(mvs)
        devices = [self._describe(mvs, native.pDeviceInfo[i], i) for i in range(native.nDeviceNum)]
        selected = next(
            (d for d in devices if (not self.serial_number or d["serial_number"] == self.serial_number)
             and (not self.ip_address or d["ip_address"] == self.ip_address)),
            None,
        )
        if selected is None:
            raise RuntimeError("没有发现符合配置的工业相机")

        camera = mvs.MvCamera()
        result = camera.MV_CC_CreateHandle(native.pDeviceInfo[selected["index"]].contents)
        if result != 0:
            raise RuntimeError(f"MVS 创建相机句柄失败: 0x{result:08x}")
        self._camera = camera
        result = camera.MV_CC_OpenDevice(getattr(mvs, "MV_ACCESS_Exclusive", 1), 0)
        if result != 0:
            if result == getattr(mvs, "MV_E_ACCESS_DENIED", 0x80000203):
                raise RuntimeError("相机无访问权限，请先停止 MVS 客户端取流并关闭相机连接")
            raise RuntimeError(f"MVS 打开相机失败: 0x{result:08x}")
        if selected["transport"] == "gige":
            packet_size = camera.MV_CC_GetOptimalPacketSize()
            if packet_size > 0:
                camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
        camera.MV_CC_SetEnumValue("TriggerMode", getattr(mvs, "MV_TRIGGER_MODE_OFF", 0))

        value_type = getattr(mvs, "MVCC_INTVALUE_EX", None) or mvs.MVCC_INTVALUE
        payload_value = value_type()
        getter = getattr(camera, "MV_CC_GetIntValueEx", None) or camera.MV_CC_GetIntValue
        result = getter("PayloadSize", payload_value)
        if result != 0 or payload_value.nCurValue <= 0:
            raise RuntimeError(f"MVS 读取图像大小失败: 0x{result:08x}")
        payload_size = int(payload_value.nCurValue)
        raw = (ctypes.c_ubyte * payload_size)()
        result = camera.MV_CC_StartGrabbing()
        if result != 0:
            raise RuntimeError(f"MVS 开始取流失败: 0x{result:08x}")

        self._device = selected
        self._last_error = ""
        self._set_status("streaming", True, "工业相机正在采集")
        logger.info("工业相机已连接: %s / %s / %s", selected["model"], selected["serial_number"], selected["ip_address"])
        interval = 1.0 / max(self.fps, 0.1)
        last_frame = 0.0
        while self._running.is_set():
            info = mvs.MV_FRAME_OUT_INFO_EX()
            ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(info))
            result = camera.MV_CC_GetOneFrameTimeout(raw, payload_size, info, self.frame_timeout_ms)
            if result == getattr(mvs, "MV_E_NODATA", 0x80000007):
                continue
            if result != 0:
                raise RuntimeError(f"MVS 取帧失败: 0x{result:08x}")
            now = time.monotonic()
            if now - last_frame < interval:
                continue
            png = self._convert_frame(mvs, camera, raw, info)
            with self._frame_ready:
                self._latest_png = png
                self._sequence += 1
                self._frame_count += 1
                self._width = int(info.nWidth)
                self._height = int(info.nHeight)
                self._frame_ready.notify_all()
            last_frame = now

    def _convert_frame(self, mvs, camera, raw, info) -> bytes:
        mono = str(self._device.get("model", "")).upper().endswith("GM")
        channels = 1 if mono else 3
        target = mvs.PixelType_Gvsp_Mono8 if mono else mvs.PixelType_Gvsp_RGB8_Packed
        size = int(info.nWidth) * int(info.nHeight) * channels
        output = (ctypes.c_ubyte * size)()
        params = mvs.MV_CC_PIXEL_CONVERT_PARAM()
        ctypes.memset(ctypes.byref(params), 0, ctypes.sizeof(params))
        params.nWidth, params.nHeight = info.nWidth, info.nHeight
        params.pSrcData = ctypes.cast(raw, ctypes.POINTER(ctypes.c_ubyte))
        params.nSrcDataLen, params.enSrcPixelType = info.nFrameLen, info.enPixelType
        params.enDstPixelType = target
        params.pDstBuffer = ctypes.cast(output, ctypes.POINTER(ctypes.c_ubyte))
        params.nDstBufferSize = size
        result = camera.MV_CC_ConvertPixelType(params)
        if result != 0:
            raise RuntimeError(f"MVS 图像转换失败: 0x{result:08x}")
        return encode_png(int(info.nWidth), int(info.nHeight), channels, bytes(output))

    def _close_camera(self) -> None:
        camera, self._camera = self._camera, None
        if camera is None:
            return
        for action in (camera.MV_CC_StopGrabbing, camera.MV_CC_CloseDevice, camera.MV_CC_DestroyHandle):
            try:
                action()
            except Exception:
                pass
