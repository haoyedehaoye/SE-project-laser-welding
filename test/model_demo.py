"""
robot_monitor v0.5 - Model detection demo

One-click run of the detection model: loads xgb_anomaly and demonstrates
anomaly detection on a synthetic frame stream.

Usage:
    python model_demo.py
or run the VS Code task "焊接监控: 一键运行检测模型".
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from data_types import DataPoint
from xgb_checker import XGBoostQualityChecker

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "xgb_anomaly"
    / "model.json"
)
WINDOW_SIZE = 32


def make_frame(temp: float, voltage: float, fault: bool = False, i: int = 0) -> DataPoint:
    status = 0b010 | (0b100 if fault else 0)
    return DataPoint(
        timestamp=time.time() + i * 0.1,
        source="stm32",
        data={
            "temperature": temp,
            "voltage": voltage,
            "status": status,
            "is_running": True,
            "is_emergency": False,
            "is_fault": fault,
        },
    )


def main() -> None:
    print("[demo] 正在启动检测模型（首次加载可能需要十几秒，请稍候）…", flush=True)
    print(f"[demo] 加载模型: {MODEL_PATH}", flush=True)
    checker = XGBoostQualityChecker(model_path=MODEL_PATH, window_size=WINDOW_SIZE)
    checker.load()
    print("[demo] 开始演示（前 32 帧为规则冷启动，之后由 XGBoost 判定）\n")

    # Phase 1: normal welding (random walk around 42C / 58%).
    random.seed(1)
    temp, voltage = 42.0, 25.0
    last_normal = None
    for i in range(40):
        temp = max(40.0, min(44.0, temp + random.uniform(-0.3, 0.3)))
        voltage = max(22.0, min(28.0, voltage + random.uniform(-0.5, 0.5)))
        last_normal = checker.check(make_frame(temp, voltage, i=i))
    print(
        f"[demo] 正常段结束: label={last_normal.label} "
        f"score={last_normal.score} {last_normal.details}"
    )

    # Phase 2: temperature ramp + fault bit set.
    last_anomaly = None
    for i in range(40, 80):
        t = 42.0 + (i - 40) * 2.0 if i < 60 else 42.0
        last_anomaly = checker.check(
            make_frame(t, 25.0, fault=55 <= i < 65, i=i)
        )
    print(
        f"[demo] 异常段结束: label={last_anomaly.label} "
        f"score={last_anomaly.score} {last_anomaly.details}"
    )
    print(f"[demo] 质检统计: {checker.stats}")
    print("\n[demo] 完成：正常段应判 normal/suspicious，温度爬升+故障段应判 anomaly")


if __name__ == "__main__":
    main()
