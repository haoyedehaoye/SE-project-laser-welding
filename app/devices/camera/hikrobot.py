"""Hikrobot MVS SDK adapter for GigE Vision cameras."""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import socket
import struct
import sys
import time
from pathlib import Path

from .base import CameraAdapter, CameraDevice, CameraFrame

logger = logging.getLogger("camera.hikrobot")
_DLL_DIRECTORY_HANDLES: list[object] = []
_PRELOADED_DLLS: list[object] = []


class MvsUnavailableError(RuntimeError):
    pass


class CameraNotFoundError(RuntimeError):
    pass


def _decode_ctypes_string(value) -> str:
    try:
        raw = bytes(value)
    except TypeError:
        return ""
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="ignore").strip()


def _ipv4(value: int) -> str:
    try:
        return socket.inet_ntoa(struct.pack(">I", int(value)))
    except (OSError, struct.error, ValueError):
        return ""


class HikrobotMvsCamera(CameraAdapter):
    def __init__(self, mvs_python_path: str = "", packet_size_auto: bool = True) -> None:
        self._mvs = self._load_mvs(mvs_python_path)
        self._camera = None
        self._device: CameraDevice | None = None
        self._payload_size = 0
        self._raw_buffer = None
        self._packet_size_auto = packet_size_auto

    @staticmethod
    def _load_mvs(configured_path: str):
        candidates = [
            configured_path,
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
                # The returned handle must stay alive; otherwise Windows removes
                # the DLL search directory before the MVS module is imported.
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_dir))
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
                control_dll = dll_path / "MvCameraControl.dll"
                if control_dll.exists():
                    _PRELOADED_DLLS.append(ctypes.WinDLL(str(control_dll), winmode=0))
        try:
            return importlib.import_module("MvCameraControl_class")
        except (ImportError, OSError) as exc:
            raise MvsUnavailableError(
                "未找到海康 MVS Python SDK。请安装 MVS，并把 MvImport 目录填入 "
                "config/app.toml 的 mvs_python_path。"
            ) from exc

    def _enum_native(self):
        device_list = self._mvs.MV_CC_DEVICE_INFO_LIST()
        transport = self._mvs.MV_GIGE_DEVICE | getattr(self._mvs, "MV_USB_DEVICE", 0)
        result = self._mvs.MvCamera.MV_CC_EnumDevices(transport, device_list)
        if result != 0:
            raise RuntimeError(f"MVS 枚举相机失败: 0x{result:08x}")
        return device_list

    def _describe(self, native, index: int) -> CameraDevice:
        info = native.contents
        is_gige = bool(info.nTLayerType & self._mvs.MV_GIGE_DEVICE)
        if is_gige:
            detail = info.SpecialInfo.stGigEInfo
            return CameraDevice(
                index=index,
                transport="gige",
                model=_decode_ctypes_string(detail.chModelName),
                serial_number=_decode_ctypes_string(detail.chSerialNumber),
                ip_address=_ipv4(detail.nCurrentIp),
                user_defined_name=_decode_ctypes_string(detail.chUserDefinedName),
            )
        detail = info.SpecialInfo.stUsb3VInfo
        return CameraDevice(
            index=index,
            transport="usb3",
            model=_decode_ctypes_string(detail.chModelName),
            serial_number=_decode_ctypes_string(detail.chSerialNumber),
            user_defined_name=_decode_ctypes_string(detail.chUserDefinedName),
        )

    def enumerate_devices(self) -> list[CameraDevice]:
        device_list = self._enum_native()
        return [self._describe(device_list.pDeviceInfo[i], i) for i in range(device_list.nDeviceNum)]

    def open(self, serial_number: str = "", ip_address: str = "") -> CameraDevice:
        self.close()
        device_list = self._enum_native()
        devices = [self._describe(device_list.pDeviceInfo[i], i) for i in range(device_list.nDeviceNum)]
        selected = next(
            (d for d in devices if (not serial_number or d.serial_number == serial_number)
             and (not ip_address or d.ip_address == ip_address)),
            None,
        )
        if selected is None:
            selector = serial_number or ip_address or "第一台可用相机"
            raise CameraNotFoundError(f"没有找到目标相机: {selector}")

        camera = self._mvs.MvCamera()
        # The Python wrapper calls byref(stDevInfo) internally, so it expects
        # MV_CC_DEVICE_INFO itself rather than the pointer stored in pDeviceInfo.
        result = camera.MV_CC_CreateHandle(device_list.pDeviceInfo[selected.index].contents)
        if result != 0:
            raise RuntimeError(f"MVS 创建句柄失败: 0x{result:08x}")
        self._camera = camera
        try:
            access = getattr(self._mvs, "MV_ACCESS_Exclusive", 1)
            result = camera.MV_CC_OpenDevice(access, 0)
            if result != 0:
                access_denied = getattr(self._mvs, "MV_E_ACCESS_DENIED", 0x80000203)
                if result == access_denied:
                    raise RuntimeError(
                        "MVS 打开相机失败: 0x80000203（无访问权限；请先停止 MVS 客户端取流并关闭其相机连接）"
                    )
                raise RuntimeError(f"MVS 打开相机失败: 0x{result:08x}")
            if selected.transport == "gige" and self._packet_size_auto:
                packet_size = camera.MV_CC_GetOptimalPacketSize()
                if packet_size > 0:
                    camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
            camera.MV_CC_SetEnumValue("TriggerMode", getattr(self._mvs, "MV_TRIGGER_MODE_OFF", 0))
            self._payload_size = self._get_payload_size(camera)
            self._raw_buffer = (ctypes.c_ubyte * self._payload_size)()
            result = camera.MV_CC_StartGrabbing()
            if result != 0:
                raise RuntimeError(f"MVS 开始取流失败: 0x{result:08x}")
            self._device = selected
            return selected
        except Exception:
            self.close()
            raise

    def _get_payload_size(self, camera) -> int:
        value_type = getattr(self._mvs, "MVCC_INTVALUE_EX", None) or self._mvs.MVCC_INTVALUE
        value = value_type()
        getter = getattr(camera, "MV_CC_GetIntValueEx", None) or camera.MV_CC_GetIntValue
        result = getter("PayloadSize", value)
        if result != 0 or value.nCurValue <= 0:
            raise RuntimeError(f"MVS 读取 PayloadSize 失败: 0x{result:08x}")
        return int(value.nCurValue)

    def read(self, timeout_ms: int) -> CameraFrame | None:
        if self._camera is None or self._raw_buffer is None:
            raise RuntimeError("camera is not open")
        info = self._mvs.MV_FRAME_OUT_INFO_EX()
        ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(info))
        result = self._camera.MV_CC_GetOneFrameTimeout(
            self._raw_buffer, self._payload_size, info, timeout_ms
        )
        if result != 0:
            no_data = getattr(self._mvs, "MV_E_NODATA", 0x80000007)
            if result == no_data:
                return None
            raise RuntimeError(f"MVS 取帧失败: 0x{result:08x}")
        return self._convert_frame(info)

    def _convert_frame(self, info) -> CameraFrame:
        is_mono = bool(self._device and self._device.model.upper().endswith("GM"))
        channels = 1 if is_mono else 3
        target_type = (
            self._mvs.PixelType_Gvsp_Mono8 if is_mono else self._mvs.PixelType_Gvsp_RGB8_Packed
        )
        output_size = int(info.nWidth) * int(info.nHeight) * channels
        output = (ctypes.c_ubyte * output_size)()
        params = self._mvs.MV_CC_PIXEL_CONVERT_PARAM()
        ctypes.memset(ctypes.byref(params), 0, ctypes.sizeof(params))
        params.nWidth = info.nWidth
        params.nHeight = info.nHeight
        params.pSrcData = ctypes.cast(self._raw_buffer, ctypes.POINTER(ctypes.c_ubyte))
        params.nSrcDataLen = info.nFrameLen
        params.enSrcPixelType = info.enPixelType
        params.enDstPixelType = target_type
        params.pDstBuffer = ctypes.cast(output, ctypes.POINTER(ctypes.c_ubyte))
        params.nDstBufferSize = output_size
        result = self._camera.MV_CC_ConvertPixelType(params)
        if result != 0:
            raise RuntimeError(f"MVS 像素格式转换失败: 0x{result:08x}")
        return CameraFrame(
            width=int(info.nWidth),
            height=int(info.nHeight),
            channels=channels,
            data=bytes(output),
            captured_at=time.time(),
            frame_number=int(info.nFrameNum),
        )

    def close(self) -> None:
        camera, self._camera = self._camera, None
        self._raw_buffer = None
        self._payload_size = 0
        self._device = None
        if camera is None:
            return
        try:
            camera.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            camera.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            camera.MV_CC_DestroyHandle()
        except Exception:
            pass
