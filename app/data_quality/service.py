"""Source-aware validation for the four predictive-maintenance modalities."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.models import DeviceEvent, SourceKind, Validity


@dataclass(slots=True)
class QualityReport:
    validity: Validity = Validity.VALID
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataQualityService:
    """Checks transport/data integrity. This does not judge weld quality."""

    def __init__(self, stale_after_seconds: float = 5.0) -> None:
        self.stale_after_seconds = stale_after_seconds

    def check(self, event: DeviceEvent) -> QualityReport:
        issues: list[str] = []
        age = max(0.0, time.time() - event.captured_at)
        if age > self.stale_after_seconds:
            issues.append(f"数据已过期 {age:.2f}s")

        if event.source is SourceKind.STM32:
            self._finite(event.data, ("current", "voltage"), issues)
            current = event.data.get("current")
            if _number(current) and not 0 <= float(current) <= 2000:
                issues.append("current 超出 0~2000A 安全校验范围")
        elif event.source is SourceKind.THERMAL:
            matrix = event.data.get("temperature_matrix")
            if matrix is not None:
                self._matrix(matrix, issues)
            else:
                if not all(_number(event.data.get(key)) for key in (
                    "sequence", "width", "height", "minimum_celsius",
                    "maximum_celsius", "average_celsius",
                )):
                    issues.append("温度帧统计无效")
                elif event.data["width"] <= 0 or event.data["height"] <= 0:
                    issues.append("温度矩阵尺寸无效")
        elif event.source is SourceKind.ROBOT:
            self._finite(event.data, ("speed", "elapsed_time"), issues)
            if _number(event.data.get("speed")) and float(event.data["speed"]) < 0:
                issues.append("speed 不能为负数")
            if _number(event.data.get("elapsed_time")) and float(event.data["elapsed_time"]) < 0:
                issues.append("elapsed_time 不能为负数")

        validity = Validity.INVALID if issues else Validity.VALID
        if issues and all(issue.startswith("数据已过期") for issue in issues):
            validity = Validity.STALE
        event.validity = validity
        return QualityReport(validity, issues, {"age_seconds": round(age, 3)})

    @staticmethod
    def _finite(data: dict[str, Any], fields: tuple[str, ...], issues: list[str]) -> None:
        for name in fields:
            if name in data and not _number(data[name]):
                issues.append(f"{name} 必须是有限数值")

    @staticmethod
    def _matrix(matrix: Any, issues: list[str]) -> None:
        if not isinstance(matrix, list) or not matrix:
            issues.append("temperature_matrix 必须是非空二维数组")
            return
        if not all(isinstance(row, list) and row for row in matrix):
            issues.append("temperature_matrix 行格式无效")
            return
        widths = {len(row) for row in matrix}
        if len(widths) != 1:
            issues.append("temperature_matrix 每行长度必须一致")
        if any(not _number(value) for row in matrix for value in row):
            issues.append("temperature_matrix 包含非有限数值")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
