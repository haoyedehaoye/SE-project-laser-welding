"""
robot_monitor v0.4 - XGBoost quality checker

Integrates an XGBoost anomaly-detection model into the existing quality
pipeline:

    Collector -> XGBoostQualityChecker.check(dp) -> QualityResult

How it works:
  1. Keeps a sliding window of the most recent N frames (window_size).
  2. Extracts a fixed-order feature vector per window (mean/std/min/max/
     last/delta plus status-bit statistics), defined by feature_spec.
  3. Predicts "anomaly probability" p with the trained XGBoost model.
  4. Maps p to QualityResult: score = 1 - p, label = normal/suspicious/
     anomaly (same semantics as RuleBasedQualityChecker).

Relationship with RuleBasedQualityChecker:
  - During warm-up (window not full) the checker delegates to the rule
    checker so startup does not produce misleading predictions.
  - If inference fails at runtime it also falls back to the rule checker.

Feature consistency:
  - train_xgboost.py saves the training feature names to features.json.
  - At inference time the checker aligns inputs by name, so a future
    change of feature order cannot silently corrupt predictions.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

from data_types import DataPoint, QualityResult
from quality_checker import BaseQualityChecker, RuleBasedQualityChecker

logger = logging.getLogger("quality.xgb")


# ============================================================
# Feature definition
# ============================================================

DEFAULT_FEATURE_SPEC = {
    "temperature":  ["mean", "std", "min", "max", "last", "delta"],
    "voltage":     ["mean", "std", "min", "max", "last", "delta"],
    "status":       ["last"],
    "is_running":   ["last", "rate"],
    "is_fault":     ["last", "count_true"],
    "is_emergency": ["last", "count_true"],
}


def _compute_stat(nums: list[float], stat: str) -> float:
    if not nums:
        return float("nan")
    if stat == "mean":
        return sum(nums) / len(nums)
    if stat == "std":
        if len(nums) < 2:
            return 0.0
        m = sum(nums) / len(nums)
        return math.sqrt(sum((x - m) ** 2 for x in nums) / len(nums))
    if stat == "min":
        return min(nums)
    if stat == "max":
        return max(nums)
    if stat == "last":
        return nums[-1]
    if stat == "delta":
        return max(nums) - min(nums)
    if stat == "rate":
        return sum(1.0 for v in nums if v) / len(nums)
    if stat == "count_true":
        return float(sum(1 for v in nums if v))
    raise ValueError(f"unknown feature statistic: {stat}")


def build_feature_vector(
    history: Sequence[DataPoint],
    feature_spec: Optional[dict] = None,
) -> tuple[list[str], list[float]]:
    """
    Convert a window of DataPoints into a fixed-order feature vector.

    Returns (feature_names, values). Missing fields become NaN, which
    XGBoost natively handles.
    """
    spec = feature_spec or DEFAULT_FEATURE_SPEC
    names: list[str] = []
    values: list[float] = []

    for field, stats in spec.items():
        series = [
            dp.data.get(field) if isinstance(dp.data.get(field), (int, float)) else None
            for dp in history
        ]
        nums = [v for v in series if v is not None]
        for stat in stats:
            names.append(f"{field}_{stat}")
            values.append(_compute_stat(nums, stat))

    return names, values


# ============================================================
# XGBoost quality checker
# ============================================================

class XGBoostQualityChecker(BaseQualityChecker):
    """
    Real-time quality checker backed by an XGBoost model.

    Usage:
        checker = XGBoostQualityChecker(model_path=...)
        checker.load()
        collector.set_quality_checker(checker)

    Model files (produced by train_xgboost.py):
        models/xgb_anomaly/model.json      native XGBoost model (JSON)
        models/xgb_anomaly/features.json   training feature names
    """

    def __init__(
        self,
        model_path: str | Path,
        feature_spec: Optional[dict] = None,
        window_size: int = 32,
        normal_threshold: float = 0.8,
        suspicious_threshold: float = 0.4,
        anomaly_prob_threshold: float = 0.6,
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
        self.window_size = max(1, int(window_size))
        self.normal_threshold = normal_threshold
        self.suspicious_threshold = suspicious_threshold
        self.anomaly_prob_threshold = anomaly_prob_threshold

        self._model = None
        self._feature_names: Optional[list[str]] = None
        # 特征窗口必须与训练一致（train_xgboost.py 用固定 `window_size` 帧建窗）：
        # 只保留最近 `window_size` 帧，避免统计量在 128 帧的可变长度上计算，
        # 造成推理特征分布与训练分布不一致（OOD）。
        self._history: deque[DataPoint] = deque(maxlen=self.window_size)

        # Rule-based fallback used while the window is warming up.
        self._fallback = RuleBasedQualityChecker()

        # Statistics (same shape as RuleBasedQualityChecker.stats).
        self.total_checked = 0
        self.normal_count = 0
        self.suspicious_count = 0
        self.anomaly_count = 0

    # ---- model loading ----

    def load(self) -> "XGBoostQualityChecker":
        """Load the XGBoost model plus feature names. Raises on failure."""
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError(
                "xgboost is not installed; run: pip install -r requirements-ml.txt"
            ) from e

        if not self.model_path.exists():
            raise FileNotFoundError(f"model file not found: {self.model_path}")

        features_path = self.model_path.with_name("features.json")
        if features_path.exists():
            self._feature_names = json.loads(
                features_path.read_text(encoding="utf-8")
            )

        model = XGBClassifier()
        model.load_model(str(self.model_path))
        self._model = model
        logger.info(
            f"XGBoost model loaded: {self.model_path} "
            f"(features={len(self._feature_names or [])}, window={self.window_size})"
        )
        return self

    # ---- checking ----

    def check(self, dp: DataPoint) -> QualityResult:
        self._history.append(dp)
        self.total_checked += 1

        # Cold start: delegate to rule checker until the window is full.
        if self._model is None or len(self._history) < self.window_size:
            result = self._fallback.check(dp)
            result.details = {**result.details, "model": "rule_fallback(warmup)"}
            self._count(result.label)
            return result

        names, values = build_feature_vector(self._history, self.feature_spec)
        if self._feature_names:
            # Align inputs to the training feature order; fill missing with NaN.
            row = dict(zip(names, values))
            values = [row.get(n, float("nan")) for n in self._feature_names]

        try:
            import numpy as np

            proba = self._model.predict_proba(
                np.asarray([values], dtype=np.float32)
            )[0]
        except Exception:
            logger.exception("XGBoost inference failed; falling back to rules")
            result = self._fallback.check(dp)
            result.details = {**result.details, "model": "rule_fallback(error)"}
            self._count(result.label)
            return result

        p_anomaly = float(proba[1]) if len(proba) > 1 else float(proba[0])
        score = round(max(0.0, min(1.0, 1.0 - p_anomaly)), 4)

        if score >= self.normal_threshold:
            label = "normal"
        elif score >= self.suspicious_threshold:
            label = "suspicious"
        else:
            label = "anomaly"
        self._count(label)

        flags = ["xgb_high_risk"] if p_anomaly >= self.anomaly_prob_threshold else []
        return QualityResult(
            score=score,
            label=label,
            flags=flags,
            details={
                "model": "xgboost",
                "anomaly_probability": round(p_anomaly, 4),
                "samples": len(self._history),
            },
        )

    def _count(self, label: str) -> None:
        if label == "normal":
            self.normal_count += 1
        elif label == "suspicious":
            self.suspicious_count += 1
        else:
            self.anomaly_count += 1

    def reset(self) -> None:
        self._history.clear()
        self._fallback.reset()

    # ---- statistics ----

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
