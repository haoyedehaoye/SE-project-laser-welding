"""
robot_monitor v0.2 — 采集器核心

v0.2 变更：
  - DataPoint 移至 types.py，新增 quality 字段
  - BaseCollector 新增 set_quality_checker()，在 _store 前自动质检
  - 重连时自动重置质检器状态
"""

from __future__ import annotations

import socket
import struct
import time
import threading
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Callable

from config import (
    SWITCH_IP, SWITCH_PORT, SOCKET_TIMEOUT,
    FRAME_HEAD, FRAME_TAIL, FRAME_MIN_LEN,
    BUFFER_SIZE, RECONNECT_INTERVAL, STATS_INTERVAL,
)
from data_types import DataPoint                                          # ← 改为从 types 导入
from quality_checker import BaseQualityChecker, NoOpQualityChecker     # ← NEW

logger = logging.getLogger("collector")


# ============================================================
# BaseCollector
# ============================================================

class BaseCollector(ABC):
    """
    所有采集器的基类。

    子类只需实现 4 个方法：
        _connect()    → 建立连接
        _disconnect() → 断开连接
        _read_raw()   → 返回 bytes | b''(超时) | None(断开)
        _extract(data: bytes) → list[DataPoint]

    质量检测：
        通过 set_quality_checker() 注入质检器。
        每个 DataPoint 在 _store 之前自动经过质检。
    """

    def __init__(self) -> None:
        self._buffer: deque[DataPoint] = deque(maxlen=BUFFER_SIZE)
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[DataPoint], None]] = []
        self._running = False
        self._quality_checker: BaseQualityChecker = NoOpQualityChecker()  # ← NEW

        # stats
        self._frame_count = 0
        self._last_stats = time.monotonic()

    # ---- 公开 API ----

    def set_quality_checker(self, checker: BaseQualityChecker) -> None:   # ← NEW
        """注入质量检测器。在 run() 之前调用。"""
        self._quality_checker = checker
        logger.info(f"质检器已设置: {type(checker).__name__}")

    def add_callback(self, cb: Callable[[DataPoint], None]) -> None:
        self._callbacks.append(cb)

    def snapshot(self) -> list[DataPoint]:
        with self._lock:
            return list(self._buffer)

    def stop(self) -> None:
        self._running = False

    # ---- 主循环 ----

    def run(self) -> None:
        self._running = True

        while self._running:
            # 连接
            try:
                self._connect()
                self._quality_checker.reset()                            # ← NEW 重连重置质检器
                logger.info("已连接")
            except Exception as e:
                logger.error(f"连接失败: {e}")
                if self._running:
                    time.sleep(RECONNECT_INTERVAL)
                continue

            # 读-解析-质检 循环
            try:
                while self._running:
                    raw = self._read_raw()

                    if raw is None:
                        logger.warning("连接断开")
                        break
                    elif raw:
                        dps = self._extract(raw)
                        for dp in dps:
                            dp.quality = self._quality_checker.check(dp)  # ← NEW 质检
                            self._store(dp)
                            self._emit(dp)
                        self._tick_stats(len(dps))
                    # else: b'' → 超时，无事发生

            except Exception as e:
                logger.error(f"采集异常: {e}", exc_info=True)

            self._disconnect()
            if self._running:
                logger.info(f"将在 {RECONNECT_INTERVAL}s 后重连…")
                time.sleep(RECONNECT_INTERVAL)

        logger.info("采集器已停止")

    # ---- 内部 ----

    def _store(self, dp: DataPoint) -> None:
        with self._lock:
            self._buffer.append(dp)

    def _emit(self, dp: DataPoint) -> None:
        for cb in self._callbacks:
            try:
                cb(dp)
            except Exception:
                logger.exception("回调异常")

    def _tick_stats(self, n_frames: int) -> None:
        self._frame_count += n_frames
        now = time.monotonic()
        elapsed = now - self._last_stats
        if elapsed >= STATS_INTERVAL and self._frame_count > 0:
            fps = self._frame_count / elapsed
            qs = self._quality_checker
            qstats = getattr(qs, 'stats', {})
            logger.info(
                f"FPS={fps:.1f}  缓冲={len(self._buffer)}/{BUFFER_SIZE}"
                + (f"  质量={qstats}" if qstats else "")
            )
            self._frame_count = 0
            self._last_stats = now

    # ---- 子类必须实现 ----

    @abstractmethod
    def _connect(self) -> None: ...
    @abstractmethod
    def _disconnect(self) -> None: ...
    @abstractmethod
    def _read_raw(self) -> Optional[bytes]: ...
    @abstractmethod
    def _extract(self, raw: bytes) -> list[DataPoint]: ...


# ============================================================
# Stm32Collector（帧协议部分不变）
# ============================================================

class Stm32Collector(BaseCollector):
    """
    连接交换机透传的 STM32 串口数据（TCP）。

    帧协议（8 字节）：
        帧头  1B  0xAA
        温度  2B  int16 大端 单位 0.1°C
        湿度  2B  uint16 大端 单位 0.1%
        状态  1B  uint8
        校验  1B  XOR
        帧尾  1B  0x55
    """

    def __init__(self) -> None:
        super().__init__()
        self._sock: Optional[socket.socket] = None
        self._frame_buf = bytearray()

    def _connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(SOCKET_TIMEOUT)
        self._sock.connect((SWITCH_IP, SWITCH_PORT))
        self._frame_buf.clear()
        logger.info(f"连接 {SWITCH_IP}:{SWITCH_PORT} 成功")

    def _disconnect(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _read_raw(self) -> Optional[bytes]:
        assert self._sock, "socket 未初始化"
        try:
            data = self._sock.recv(4096)
            return data if data else None
        except socket.timeout:
            return b''
        except OSError:
            return None

    def _extract(self, raw: bytes) -> list[DataPoint]:
        self._frame_buf.extend(raw)
        results: list[DataPoint] = []

        while len(self._frame_buf) >= FRAME_MIN_LEN:
            if self._frame_buf[0] != FRAME_HEAD:
                self._frame_buf.pop(0)
                continue
            if self._frame_buf[FRAME_MIN_LEN - 1] != FRAME_TAIL:
                self._frame_buf.pop(0)
                continue

            segment = self._frame_buf[1:6]
            expected = self._frame_buf[6]
            actual = 0
            for b in segment:
                actual ^= b
            actual &= 0xFF

            if actual != expected:
                self._frame_buf.pop(0)
                continue

            dp = self._parse_segment(bytes(segment))
            results.append(dp)
            del self._frame_buf[:FRAME_MIN_LEN]

        return results

    @staticmethod
    def _parse_segment(seg: bytes) -> DataPoint:
        temp_raw, humi_raw, status = struct.unpack(">hHB", seg)
        return DataPoint(
            timestamp=time.time(),
            source="stm32",
            data={
                "temperature": round(temp_raw / 10.0, 1),
                "humidity":    round(humi_raw / 10.0, 1),
                "status":      status,
                "is_running":  bool(status & 0b010),
                "is_emergency": bool(status & 0b001),
                "is_fault":    bool(status & 0b100),
            },
        )