"""
robot_monitor v0.7 - Multi-method detection demo (terminal friendly)

Runs all detection methods on the same synthetic stream (normal phase +
anomaly phase) and prints a readable comparison: per-method accuracy,
per-window verdicts, and a fused vote.

Methods compared:
    - XGBoost window model (the production checker, loaded from models/xgb_anomaly)
    - Z-Score statistics
    - Isolation Forest
    - Local Outlier Factor
    - One-Class SVM

Usage:
    python detect_demo.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_types import DataPoint
from detectors import DETECTOR_CLASSES, THRESHOLD
from xgb_checker import DEFAULT_FEATURE_SPEC, XGBoostQualityChecker, build_feature_vector

WINDOW = 32
N_FIT_SEQS = 30
FRAMES_PER_SEQ = 200
TEST_NORMAL = 80
TEST_ANOMALY = 80
STRIDE = 4

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgb_anomaly" / "model.json"


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


def fmt_table(headers: list[str], rows: list[list]) -> str:
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    sep = "-+-".join("-" * w for w in widths)
    line = lambda cells: " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    out = [line(headers), sep]
    out += [line(r) for r in rows]
    return "\n".join(out)


def build_fit_windows() -> list[list[DataPoint]]:
    random.seed(42)
    frames: list[DataPoint] = []
    for _ in range(N_FIT_SEQS):
        temp, voltage = 42.0, 25.0
        for i in range(FRAMES_PER_SEQ):
            temp = max(40.0, min(44.0, temp + random.uniform(-0.3, 0.3)))
            voltage = max(22.0, min(28.0, voltage + random.uniform(-0.5, 0.5)))
            frames.append(make_frame(temp, voltage, i=i))
    return [frames[k : k + WINDOW] for k in range(0, len(frames) - WINDOW + 1, 8)]


def build_test_stream() -> list[DataPoint]:
    random.seed(1)
    frames: list[DataPoint] = []
    temp, voltage = 42.0, 25.0
    for i in range(TEST_NORMAL):
        temp = max(40.0, min(44.0, temp + random.uniform(-0.3, 0.3)))
        voltage = max(22.0, min(28.0, voltage + random.uniform(-0.5, 0.5)))
        frames.append(make_frame(temp, voltage, i=i))
    for i in range(TEST_ANOMALY):
        j = i + TEST_NORMAL
        temp = min(60.0, 42.0 + i * 1.2)
        voltage = max(15.0, 25.0 - i * 0.3)
        fault = 12 <= i < 20
        frames.append(make_frame(temp, voltage, fault=fault, i=j))
    return frames


def main() -> None:
    print("[demo] 拟合无监督检测器（正常窗口基线）...", flush=True)
    fit_windows = build_fit_windows()
    X_fit = np.asarray(
        [build_feature_vector(w, DEFAULT_FEATURE_SPEC)[1] for w in fit_windows],
        dtype=np.float32,
    )
    detectors = [cls() for cls in DETECTOR_CLASSES]
    for d in detectors:
        d.fit(X_fit)

    print(f"[demo] 加载 XGBoost 窗口模型: {MODEL_PATH}", flush=True)
    checker = XGBoostQualityChecker(model_path=MODEL_PATH, window_size=WINDOW)
    checker.load()

    print(f"[demo] 生成测试流：正常段 {TEST_NORMAL} 帧 + 异常段 {TEST_ANOMALY} 帧（温度爬升/电压跌落/故障位）\n", flush=True)
    test_frames = build_test_stream()
    frame_labels: list[int] = []
    for f in test_frames:
        r = checker.check(f)
        frame_labels.append(1 if r.label == "anomaly" else 0)

    # Sliding windows with known phase labels.
    windows: list[tuple[int, int, list[DataPoint]]] = []  # (start, true_label, frames)
    for k in range(0, len(test_frames) - WINDOW + 1, STRIDE):
        if k + WINDOW <= TEST_NORMAL:
            true = 0
        elif k >= TEST_NORMAL:
            true = 1
        else:
            continue  # boundary window
        windows.append((k, true, test_frames[k : k + WINDOW]))

    # Score every window with every method.
    method_scores = {d.name: [] for d in detectors}
    method_scores["XGBoost 窗口模型"] = []
    fused_scores = []
    detail_rows: list[list] = []
    for k, true, w in windows:
        x = np.asarray(build_feature_vector(w, DEFAULT_FEATURE_SPEC)[1], dtype=np.float32)
        scores = {}
        for d in detectors:
            scores[d.name] = d.score(x)
        scores["XGBoost 窗口模型"] = float(frame_labels[k + WINDOW - 1])
        vote = 1 if sum(v for v in scores.values()) / len(scores) >= THRESHOLD else 0
        for name, s in scores.items():
            method_scores[name].append(1 if s >= THRESHOLD else 0)
        fused_scores.append(vote)

        if (true == 0 and k < 3 * STRIDE) or (true == 1 and k < TEST_NORMAL + 3 * STRIDE):
            detail_rows.append(
                [k, "正常" if true == 0 else "异常"]
                + ["A" if scores[d.name] >= THRESHOLD else "N" for d in detectors]
                + ["A" if scores["XGBoost 窗口模型"] >= THRESHOLD else "N"]
                + ["A" if vote else "N"]
            )

    # Summary table.
    def phase_acc(preds: list[int], true: int) -> float:
        idx = [i for i, t in enumerate([w[1] for w in windows]) if t == true]
        if not idx:
            return float("nan")
        return sum(preds[i] == true for i in idx) / len(idx)

    headers = ["方法", "正常段准确率", "异常段准确率", "总体准确率"]
    rows = []
    for name, preds in method_scores.items():
        overall = sum(p == t for p, t in zip(preds, [w[1] for w in windows])) / len(windows)
        rows.append([name, f"{phase_acc(preds, 0):.0%}", f"{phase_acc(preds, 1):.0%}", f"{overall:.0%}"])
    overall_fused = sum(p == t for p, t in zip(fused_scores, [w[1] for w in windows])) / len(windows)
    rows.append(["综合投票（平均分）", f"{phase_acc(fused_scores, 0):.0%}", f"{phase_acc(fused_scores, 1):.0%}", f"{overall_fused:.0%}"])

    print("========== 各方法检测准确率（窗口级） ==========")
    print(fmt_table(headers, rows))
    print(f"\n窗口总数: {len(windows)}  正常段窗口: {sum(1 for w in windows if w[1] == 0)}  异常段窗口: {sum(1 for w in windows if w[1] == 1)}")

    print("\n========== 逐窗口判定样例（A=异常 N=正常） ==========")
    detail_headers = ["起始帧", "真实"] + [d.name for d in detectors] + ["XGBoost 窗口模型", "投票"]
    print(fmt_table(detail_headers, detail_rows))

    print("\n[demo] 完成：异常段包含温度爬升、电压跌落与故障位，各方法对未见过异常的检出能力如上。")


if __name__ == "__main__":
    main()
