"""
ST M32 模拟器 — 在本地 127.0.0.1:8234 起 TCP 服务端，
按协议格式周期发送假数据，供采集器联调。

用法： python simulator.py
关闭： Ctrl+C

协议假设（见 config.py 注释），硬件到位后改 _build_frame() 即可。
"""

import struct
import socket
import time
import random
import signal
import sys

# ---- 直接从 config 拿参数，避免硬编码两处不一致 ----
try:
    from config import SWITCH_IP, SWITCH_PORT, FRAME_HEAD, FRAME_TAIL
except ImportError:
    SWITCH_IP = "127.0.0.1"
    SWITCH_PORT = 8234
    FRAME_HEAD = 0xAA
    FRAME_TAIL = 0x55

FRAME_INTERVAL = 0.1          # 发送间隔（秒），10Hz
TEMP_BASE = 42.0              # 模拟温度基线（°C）
VOLTAGE_BASE = 25.0              # 模拟电压基线（%）


def _build_frame(temp: float, voltage: float, status: int) -> bytes:
    """
    按协议拼一帧。这是你拿到真协议后唯一要改的函数。
    协议格式：
      帧头(1B) + 温度(int16, 0.1°C) + 电压(uint16, 0.1V) + 状态(uint8) + 校验(XOR,1B) + 帧尾(1B)
    """
    temp_scaled   = int(temp * 10)          # 42.5 → 425
    voltage_scaled   = int(voltage * 10)      # 58.3 → 583

    # 5 字节数据段（不含头尾）
    data = struct.pack(">hHB", temp_scaled, voltage_scaled, status)
    #       大端: 温度 int16, 电压 uint16, 状态 uint8

    checksum = 0
    for b in data:
        checksum ^= b
    checksum &= 0xFF

    return bytes([FRAME_HEAD]) + data + bytes([checksum, FRAME_TAIL])


def _random_walk(base: float, spread: float, prev: float) -> float:
    """在 base 附近做 ±spread 的随机游走"""
    delta = random.uniform(-0.3, 0.3)
    new = (prev if prev else base) + delta
    # 钳在 base ± spread 内
    return max(base - spread, min(base + spread, new))


def _format_frame(raw: bytes) -> str:
    """把二进制帧格式化成可读的 hex 串，方便终端查看"""
    return " ".join(f"{b:02X}" for b in raw)


# ============================================================
# 主逻辑
# ============================================================

def main():
    print(f"STM32 Simulator — 监听 {SWITCH_IP}:{SWITCH_PORT}")
    print(f"帧间隔={FRAME_INTERVAL}s, 帧头={FRAME_HEAD:#04X}, 帧尾={FRAME_TAIL:#04X}")
    print("Ctrl+C 退出\n")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SWITCH_IP, SWITCH_PORT))
    server.listen(1)
    print("等待采集器连接...")

    conn, addr = server.accept()
    print(f"采集器已连接: {addr}\n")

    temp_prev = TEMP_BASE
    voltage_prev = VOLTAGE_BASE
    seq = 0

    def shutdown(sig, frame):
        print("\n模拟器关闭")
        conn.close()
        server.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            seq += 1
            temp = _random_walk(TEMP_BASE, 2.0, temp_prev)
            voltage = _random_walk(VOLTAGE_BASE, 3.0, voltage_prev)
            status = 0b010        # bit1=1 → 运行中
            temp_prev, voltage_prev = temp, voltage

            frame = _build_frame(temp, voltage, status)
            conn.sendall(frame)

            print(f"[{seq:04d}] → 温度={temp:.1f}°C  电压={voltage:.1f}V  "
                  f"状态={status:#05b}  HEX: {_format_frame(frame)}")

            time.sleep(FRAME_INTERVAL)

    except (BrokenPipeError, ConnectionResetError):
        print("采集器断开连接")
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()