"""
robot_monitor v0.2 — 数据质量检测

设计思路：
  - 基类 BaseQualityChecker 定义接口，方便替换不同模型
  - RuleBasedQualityChecker  基于规则的实时质检（范围/变化率/冻结）
  - MLQualityChecker         预留 ML 模型接口（ONNX/PyTorch）
  - 质检器内部维护滑动窗口历史，不依赖 Collector 状态
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from data_types import DataPoint, QualityResult

logger = logging.getLogger("quality")


# ============================================================
# 基类
# ============================================================

class BaseQualityChecker(ABC):
    """
    质量检测器抽象基类。

    子类只需实现 check()。
    内部可维护状态（滑动窗口、计数器等）。
    """

    @abstractmethod
    def check(self, dp: DataPoint) -> QualityResult:
        """
        对单个 DataPoint 做质量检测。
        返回 QualityResult，同时应更新内部状态。
        """
        ...

    def reset(self) -> None:
        """重置内部状态（连接断开重连时调用）。"""
        pass


# ============================================================
# 空检测器（默认，不做任何过滤）
# ============================================================

class NoOpQualityChecker(BaseQualityChecker):
    """不做检测，所有数据满分通过。"""

    def check(self, dp: DataPoint) -> QualityResult:
        return QualityResult(score=1.0, label="normal", flags=[], details={})


# ============================================================
# 基于规则的检测器
# ============================================================

class RuleBasedQualityChecker(BaseQualityChecker):
    """
    多维度规则检测。

    检测维度：
      1. 范围检测  — 温度/湿度是否在物理合理范围内
      2. 变化率    — 相邻帧跳变是否超过阈值（传感器故障/接触不良）
      3. 冻结检测  — 连续 N 帧数值不变（传感器卡死）

    评分规则：
      - 满分 1.0，每个 flag 扣 0.2，最低 0.0
      - score >= 0.8 → normal
      - score >= 0.4 → suspicious
      - score <  0.4 → anomaly
    """

    def __init__(
        self,
        # 合理范围
        temp_range: tuple[float, float] = (-20.0, 85.0),
        humi_range: tuple[float, float] = (0.0, 100.0),
        # 变化率阈值（相邻帧最大允许变化）
        max_temp_delta: float = 5.0,         # °C
        max_humi_delta: float = 10.0,        # %
        # 冻结检测
        freeze_window: int = 8,              # 连续多少帧不变视为冻结
        freeze_temp_tol: float = 0.2,        # 温度视为"不变"的容差
        freeze_humi_tol: float = 0.5,        # 湿度视为"不变"的容差
        # 历史窗口大小（用于统计，0=不限制）
        history_size: int = 64,
    ) -> None:
        self.temp_range = temp_range
        self.humi_range = humi_range
        self.max_temp_delta = max_temp_delta
        self.max_humi_delta = max_humi_delta
        self.freeze_window = freeze_window
        self.freeze_temp_tol = freeze_temp_tol
        self.freeze_humi_tol = freeze_humi_tol

        # 内部状态
        self._history: deque[DataPoint] = deque(maxlen=history_size or 64)
        self._freeze_counter: int = 0
        self._last_temp: Optional[float] = None
        self._last_humi: Optional[float] = None

        # 统计
        self.total_checked: int = 0
        self.normal_count: int = 0
        self.suspicious_count: int = 0
        self.anomaly_count: int = 0

    # ---- 主检测逻辑 ----

    def check(self, dp: DataPoint) -> QualityResult:
        temp = dp.data.get("temperature")
        humi = dp.data.get("humidity")
        flags: list[str] = []
        details: dict = {}

        # 1) 范围检测
        if temp is not None:
            lo, hi = self.temp_range
            if temp < lo or temp > hi:
                flags.append("temp_out_of_range")
                details["temp"] = temp
                details["temp_range"] = [lo, hi]

        if humi is not None:
            lo, hi = self.humi_range
            if humi < lo or humi > hi:
                flags.append("humidity_out_of_range")
                details["humidity"] = humi
                details["humi_range"] = [lo, hi]

        # 2) 变化率检测（需要前一帧）
        if self._last_temp is not None and temp is not None:
            delta = abs(temp - self._last_temp)
            if delta > self.max_temp_delta:
                flags.append("temp_rapid_change")
                details["temp_delta"] = round(delta, 2)
                details["temp_prev"] = self._last_temp
                details["temp_curr"] = temp

        if self._last_humi is not None and humi is not None:
            delta = abs(humi - self._last_humi)
            if delta > self.max_humi_delta:
                flags.append("humi_rapid_change")
                details["humi_delta"] = round(delta, 2)
                details["humi_prev"] = self._last_humi
                details["humi_curr"] = humi

        # 3) 冻结检测
        if temp is not None and humi is not None:
            if (
                self._last_temp is not None
                and abs(temp - self._last_temp) < self.freeze_temp_tol
                and abs(humi - self._last_humi) < self.freeze_humi_tol
            ):
                self._freeze_counter += 1
            else:
                self._freeze_counter = 0

            if self._freeze_counter >= self.freeze_window:
                flags.append("sensor_frozen")
                details["frozen_frames"] = self._freeze_counter

        # 4) 综合评分
        score = max(0.0, 1.0 - len(flags) * 0.2)

        if score >= 0.8:
            label = "normal"
            self.normal_count += 1
        elif score >= 0.4:
            label = "suspicious"
            self.suspicious_count += 1
        else:
            label = "anomaly"
            self.anomaly_count += 1

        # 5) 更新内部状态
        self._last_temp = temp
        self._last_humi = humi
        self._history.append(dp)
        self.total_checked += 1

        return QualityResult(score=score, label=label, flags=flags, details=details)

    # ---- 重置（重连时调用） ----

    def reset(self) -> None:
        self._history.clear()
        self._freeze_counter = 0
        self._last_temp = None
        self._last_humi = None

    # ---- 统计快照 ----

    @property
    def stats(self) -> dict:
        total = max(self.total_checked, 1)
        return {
            "total": self.total_checked,
            "normal": self.normal_count,
            "suspicious": self.suspicious_count,
            "anomaly": self.anomaly_count,
            "normal_rate": round(self.normal_count / total, 3),
            "anomaly_rate": round(self.anomaly_count / total, 3),
        }


# ============================================================
# ML 模型检测器（预留骨架）
# ============================================================

class MLQualityChecker(BaseQualityChecker):
    """
    基于机器学习模型的质量检测器。

    使用方式：
        checker = MLQualityChecker(model_path="models/quality.onnx")
        collector.set_quality_checker(checker)

    模型输入：最近 N 帧的特征向量（温度、湿度、状态位、时间间隔等）
    模型输出：质量分数 0~1

    TODO: 接入实际模型
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        logger.warning(f"MLQualityChecker 尚未加载模型: {model_path}")

    def check(self, dp: DataPoint) -> QualityResult:
        # TODO: 特征提取 → 模型推理 → QualityResult
        # 目前返回默认值
        return QualityResult(
            score=1.0,
            label="normal",
            flags=[],
            details={"model": "not_loaded", "path": self.model_path},
        )