"""Minimal PNG encoder used so camera preview has no OpenCV/Pillow dependency."""

from __future__ import annotations

import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_png(width: int, height: int, channels: int, pixels: bytes, compression: int = 3) -> bytes:
    if width <= 0 or height <= 0 or channels not in (1, 3):
        raise ValueError("invalid image dimensions or channel count")
    stride = width * channels
    expected = stride * height
    if len(pixels) < expected:
        raise ValueError(f"short pixel buffer: expected {expected}, got {len(pixels)}")
    rows = b"".join(b"\x00" + pixels[offset:offset + stride] for offset in range(0, expected, stride))
    color_type = 0 if channels == 1 else 2
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    level = max(0, min(9, compression))
    return PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows, level)) + _chunk(b"IEND", b"")
