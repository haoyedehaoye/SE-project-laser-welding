# -*- coding: utf-8 -*-
"""
robot_monitor v0.9 — 多输出质量模型在线预测（服务/演示共用）

加载 models/xgb_multiout：
  - crack/model.json                  裂纹概率（XGBClassifier）
  - geo/<steel_w|copper_w|copper_d|gap>/model.json  几何量（µm）

用法：
    python predict_quality.py --power 1050 --speed 1 --gas 15 --focal 0 --angular 0 --thickness 0.6
    # 若训练时含横截面位置特征：
    python predict_quality.py ... --pos 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "xgb_multiout"

CANONICAL = ["power", "speed", "gas_flow", "focal", "angular", "thickness"]
PARAM_LABELS = {
    "power": "功率(W)", "speed": "焊接速度(m/min)", "gas_flow": "气流量(l/min)",
    "focal": "焦点位置(mm)", "angular": "角度(°)", "thickness": "板厚(mm)",
    "cross_section_mm": "焊缝横截面位置(mm)",
}
ALIASES = {"gas": "gas_flow", "cross_section": "cross_section_mm", "pos": "cross_section_mm"}
GEO_LABELS = {
    "steel_w": "钢侧熔宽(µm)", "copper_w": "铜侧熔宽(µm)",
    "copper_d": "铜侧熔深(µm)", "gap": "间隙(µm)",
}


class MultiOutputPredictor:
    def __init__(self, model_dir: str | Path = MODEL_DIR) -> None:
        self.dir = Path(model_dir)
        self.input_features = json.loads(
            (self.dir / "input_features.json").read_text(encoding="utf-8"))
        ranges_path = self.dir / "input_ranges.json"
        self.ranges = json.loads(ranges_path.read_text(encoding="utf-8")) \
            if ranges_path.exists() else {}
        self._crack = None
        self._geo = {}

    def load(self) -> "MultiOutputPredictor":
        self._crack = xgb.XGBClassifier()
        self._crack.load_model(str(self.dir / "crack" / "model.json"))
        for key in GEO_LABELS:
            m = xgb.XGBRegressor()
            m.load_model(str(self.dir / "geo" / key / "model.json"))
            self._geo[key] = m
        return self

    def _canon(self, params: dict) -> dict:
        out = {}
        for k, v in params.items():
            key = ALIASES.get(k, k)
            out[key] = float(v)
        return out

    def predict(self, params: dict) -> dict:
        """params: {canonical/别名: float}（缺省取训练范围中点）→ 预测结果 dict。"""
        p = self._canon(params)
        values = []
        warnings = []
        for key in self.input_features:
            if key not in p:
                lo, hi = self.ranges.get(key, (0.0, 1.0))
                p[key] = (lo + hi) / 2.0
                warnings.append(f"{key} 未提供，取训练范围中点 {p[key]:g}")
            rng = self.ranges.get(key)
            if rng and not (rng[0] - 1e-6 <= p[key] <= rng[1] + 1e-6):
                warnings.append(f"{key}={p[key]:g} 超出训练范围 {rng}，外推结果仅供参考")
            values.append(p[key])

        X = np.asarray([values], dtype=np.float32)
        proba = float(self._crack.predict_proba(X)[0][1])
        geometry = {key: float(self._geo[key].predict(X)[0]) for key in GEO_LABELS}
        return {
            "input": {k: round(v, 4) for k, v in p.items()},
            "crack_probability": round(proba, 4),
            "crack_label": "有裂纹风险" if proba >= 0.5 else "无裂纹",
            "risk_level": _risk_level(proba),
            "geometry": {k: round(v, 2) for k, v in geometry.items()},
            "warnings": warnings,
        }

    def summary_text(self, params: dict) -> str:
        r = self.predict(params)
        lines = [
            "输入工艺参数：" + "  ".join(
                f"{PARAM_LABELS.get(k, k)}={v:g}" for k, v in r["input"].items()),
            f"裂纹概率 {r['crack_probability']:.1%}（{r['crack_label']}）",
            "几何预测：" + "  ".join(f"{GEO_LABELS[k]}={v:g}" for k, v in r["geometry"].items()),
        ]
        if r["warnings"]:
            lines.append("注意：" + "；".join(r["warnings"]))
        return "\n".join(lines)


def _risk_level(proba: float) -> str:
    if proba >= 0.5:
        return "高"
    if proba >= 0.3:
        return "中（接近阈值，建议复核）"
    return "低"


def main() -> None:
    ap = argparse.ArgumentParser(description="多输出质量模型预测")
    for key in CANONICAL:
        ap.add_argument(f"--{key}", type=float, default=None,
                        help=PARAM_LABELS[key])
    ap.add_argument("--gas", dest="gas_flow", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pos", dest="cross_section_mm", type=float, default=None,
                    help="焊缝横截面位置(mm)，仅当训练含该特征时有意义")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    pred = MultiOutputPredictor().load()
    params = {k: v for k, v in vars(args).items() if k != "out" and v is not None}
    print(pred.summary_text(params))


if __name__ == "__main__":
    main()
