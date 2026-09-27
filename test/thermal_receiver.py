"""Receive TMP1 frames from the work PC and forward them to the TEST backend."""

from __future__ import annotations

import argparse
import http.client
import logging
import socket
import struct
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

HEADER = struct.Struct("<IHHqHHHHhHqi")
MAGIC = 0x31504D54
CONFIG_PATH = Path(__file__).with_name("thermal_config.toml")
LOG = logging.getLogger("thermal_receiver")


def read_exact(sock: socket.socket, size: int) -> bytes:
    parts = bytearray(size)
    view = memoryview(parts)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise EOFError("发送端已断开")
        offset += received
    return parts


def parse_header(data: bytes) -> tuple[dict, int]:
    magic, version, header_size, sequence, device, width, height, slope, offset, _, stamp, length = HEADER.unpack(data)
    if magic != MAGIC or version != 1 or header_size != HEADER.size:
        raise ValueError("无效的 TMP1 温度帧头")
    if width <= 0 or height <= 0 or width * height > 1_000_000 or slope <= 0:
        raise ValueError("无效的矩阵尺寸或温度斜率")
    if length != width * height * 2:
        raise ValueError("温度帧负载长度不匹配")
    headers = {
        "X-Thermal-Sequence": str(sequence),
        "X-Thermal-Device-Index": str(device),
        "X-Thermal-Width": str(width),
        "X-Thermal-Height": str(height),
        "X-Thermal-Slope": str(slope),
        "X-Thermal-Offset": str(offset),
        "X-Thermal-Timestamp": str(stamp),
        "Content-Type": "application/octet-stream",
    }
    return headers, length


def settings(path: Path) -> tuple[str, int, str]:
    with path.open("rb") as stream:
        data = tomllib.load(stream)["receiver"]
    host = str(data.get("bind_host", "0.0.0.0"))
    port = int(data.get("port", 5000))
    url = str(data.get("backend_url", "http://127.0.0.1:18080/api/data/temperature/raw"))
    if not 1 <= port <= 65535:
        raise ValueError("监听端口必须在 1~65535 之间")
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError("backend_url 必须是包含端口的 http 地址")
    return host, port, url


def serve(host: str, port: int, backend_url: str) -> None:
    parsed = urlsplit(backend_url)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    connection: http.client.HTTPConnection | None = None
    received_frames = forwarded = failed = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        LOG.info("监听 %s:%s；TEST 后端 %s", host, port, backend_url)
        LOG.info("THERMAL_RECEIVER_READY")
        while True:
            client, address = listener.accept()
            LOG.info("工作机已连接：%s:%s", *address)
            with client:
                client.settimeout(10)
                while True:
                    try:
                        headers, payload_size = parse_header(read_exact(client, HEADER.size))
                        payload = read_exact(client, payload_size)
                    except (EOFError, OSError) as exc:
                        LOG.info("工作机连接结束：%s", exc)
                        break
                    except ValueError as exc:
                        LOG.error("温度帧协议错误，等待重新连接：%s", exc)
                        break
                    received_frames += 1
                    try:
                        if connection is None:
                            connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
                        connection.request("POST", target, payload, headers)
                        response = connection.getresponse()
                        response.read()
                        if response.status != 200:
                            raise RuntimeError(f"后端 HTTP {response.status}")
                        forwarded += 1
                    except (OSError, http.client.HTTPException, RuntimeError) as exc:
                        failed += 1
                        if connection:
                            connection.close()
                        connection = None
                        if failed == 1 or failed % 25 == 0:
                            LOG.error("后端交帧失败 %s 次：%s", failed, exc)
                    if received_frames == 1 or received_frames % 25 == 0:
                        LOG.info("收到=%s 成功=%s 失败=%s 帧号=%s", received_frames,
                                 forwarded, failed, headers["X-Thermal-Sequence"])


def main() -> None:
    parser = argparse.ArgumentParser(description="TEST 温度矩阵接收端")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host, port, backend_url = settings(args.config)
    serve(host, port, backend_url)


if __name__ == "__main__":
    main()
