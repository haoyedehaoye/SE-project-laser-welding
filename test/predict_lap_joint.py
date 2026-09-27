"""
robot_monitor v0.6 - Steel-copper lap joint crack predictor

Loads models/xgb_lap_joint and predicts crack probability from the six
laser-welding process parameters.

Usage:
    python predict_lap_joint.py --power 1050 --speed 1 --gas 15 --focal 0 --angular 0 --thickness 0.6
    python predict_lap_joint.py --importance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "xgb_lap_joint"

FEATURE_KEYS = ["power", "speed", "gas_flow", "focal", "angular", "thickness"]
FEATURE_LABELS = [
    "功率 (W)",
    "焊接速度 (m/min)",
    "气流量 (l/min)",
    "焦点位置 (mm)",
    "角度 (°)",
    "板厚 (mm)",
]
PARAM_RANGES = {
    "power": (900, 1200),
    "speed": (0.8, 1.2),
    "gas_flow": (10, 20),
    "focal": (-2, 2),
    "angular": (-15, 15),
    "thickness": (0.5, 0.7),
}
DEFAULTS = {"power": 1050.0, "speed": 1.0, "gas_flow": 15.0, "focal": 0.0, "angular": 0.0, "thickness": 0.6}


def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "model.json"))
    feature_names = json.loads((MODEL_DIR / "features.json").read_text(encoding="utf-8"))
    return model, feature_names


def predict_from_keys(model, feature_names, params: dict) -> float:
    """params: {key: float} for the six FEATURE_KEYS -> crack probability."""
    # Build the vector in features.json order (same as training).
    names = feature_names
    values = []
    for name in names:
        key = _key_for_feature_name(name)
        values.append(float(params.get(key, DEFAULTS[key])))
    proba = model.predict_proba(np.asarray([values], dtype=np.float32))[0]
    return float(proba[1])


def _key_for_feature_name(name: str) -> str:
    low = name.lower()
    for alias, key in [("power", "power"), ("speed", "speed"),
                        ("gas flow", "gas_flow"), ("gas_flow", "gas_flow"),
                        ("focal", "focal"), ("angular", "angular"),
                        ("thickness", "thickness")]:
        if alias in low:
            return key
    raise KeyError(name)


def feature_importance(model, feature_names) -> list[tuple[str, float]]:
    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    out = []
    for key, g in gain.items():
        idx = int(key[1:]) if key.startswith("f") else -1
        name = feature_names[idx] if 0 <= idx < len(feature_names) else key
        out.append((name, g / total))
    return sorted(out, key=lambda t: -t[1])


def _risk_level(proba: float) -> str:
    if proba >= 0.5:
        return "高（判为有裂纹风险）"
    if proba >= 0.3:
        return "中（接近阈值，建议复核）"
    return "低（判为无裂纹）"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict crack risk from process parameters")
    for key in FEATURE_KEYS:
        lo, hi = PARAM_RANGES[key]
        arg_name = "gas" if key == "gas_flow" else key
        parser.add_argument(f"--{arg_name}", type=float, default=DEFAULTS[key],
                            help=f"{FEATURE_LABELS[FEATURE_KEYS.index(key)]}（范围 {lo}~{hi}）")
    parser.add_argument("--importance", action="store_true", help="show feature importance and exit")
    args = parser.parse_args()

    model, feature_names = load_model()
    if args.importance:
        print("特征重要性（按 gain，越大越影响裂纹判定）：")
        for name, rel in feature_importance(model, feature_names):
            print(f"  {name:<28} {rel:.4f}")
        return

    params = {
        "power": args.power,
        "speed": args.speed,
        "gas_flow": args.gas,
        "focal": args.focal,
        "angular": args.angular,
        "thickness": args.thickness,
    }
    proba = predict_from_keys(model, feature_names, params)
    print("输入工艺参数：")
    for k in FEATURE_KEYS:
        print(f"  {FEATURE_LABELS[FEATURE_KEYS.index(k)]:<16} {params[k]}")
    print("-" * 46)
    print(f"裂纹概率: {proba:.1%}")
    print(f"判定    : {_risk_level(proba)}")


if __name__ == "__main__":
    main()
