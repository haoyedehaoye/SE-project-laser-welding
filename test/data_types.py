"""
robot_monitor v0.2 — 共享数据类型

DataPoint     : 统一数据容器，新增 quality 字段
QualityResult : 质量检测结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualityResult:
    """质量检测结果。可 JSON 序列化。"""
    score: float = 1.0                     # 0.0 ~ 1.0，1.0 最佳
    label: str = "normal"                  # normal / suspicious / anomaly
    flags: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "flags": self.flags,
            "details": self.details,
        }

    def is_valid(self, min_score: float = 0.6) -> bool:
        """快速判断数据是否可用"""
        return self.score >= min_score


@dataclass
class DataPoint:
    """一条传感器数据。可哈希、可比较。"""
    timestamp: float
    source: str
    data: dict
    quality: Optional[QualityResult] = None  # ← 新增字段

    def __repr__(self) -> str:
        ts = f"{self.timestamp:.3f}"
        q = f" q={self.quality.label}" if self.quality else ""
        return f"DataPoint(ts={ts}, src={self.source}{q}, data={self.data})"