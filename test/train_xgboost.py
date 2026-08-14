"""
robot_monitor v0.4 - XGBoost anomaly detection model training script

Usage:
    1) Train on synthetic data (mirrors test/simulator.py scenarios):
       python train_xgboost.py

    2) Train on your own labeled data:
       python train_xgboost.py --csv data/welding_labeled.csv

       CSV columns: temperature, humidity, status, label
       (label: 0 = normal, 1 = anomaly; optional run_id groups sequences,
        meaning one welding run is kept inside one train/val split)

Output:
    models/xgb_anomaly/model.json      native XGBoost model (JSON)
    models/xgb_anomaly/features.json   feature names (inference aligns by name)
    models/xgb_anomaly/config.json     training config + thresholds
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def split_sequences(
    n: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sequence (welding run) indices into train/val/test
    without cross-run leakage: a run goes entirely into one split."""
    idx = np.arange(n)
    rest_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=stratify
    )
    stratify_rest = stratify[rest_idx] if stratify is not None else None
    train_idx, val_idx = train_test_split(
        rest_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_rest,
    )
    return train_idx, val_idx, test_idx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_types import DataPoint
from xgb_checker import DEFAULT_FEATURE_SPEC, build_feature_vector

random.seed(42)
np.random.seed(42)

WINDOW_SIZE = 32
FRAMES_PER_SEQ = 200
N_SEQUENCES = 80

ANOMALY_TYPES = [
    "temp_spike",
    "humidity_drop",
    "frozen",
    "status_fault",
    "rapid_change",
]


# ============================================================
# Synthetic data (mirrors the scenarios the rule checker detects)
# ============================================================

def _walk(base: float, spread: float, prev: Optional[float]) -> float:
    prev = base if prev is None else prev
    delta = random.uniform(-0.3, 0.3)
    return max(base - spread, min(base + spread, prev + delta))


def simulate_sequence(seq_id: int) -> list[tuple[dict, int]]:
    temp = 42.0
    humi = 58.0
    status = 0b010
    frozen_temp, frozen_humi = temp, humi
    episode_left = 0
    episode_type: Optional[str] = None
    frames: list[tuple[dict, int]] = []

    for _ in range(FRAMES_PER_SEQ):
        label = 0

        # Start a new anomaly episode occasionally.
        if episode_left <= 0 and random.random() < 0.03:
            episode_type = random.choice(ANOMALY_TYPES)
            episode_left = random.randint(6, 14)

        if episode_left > 0:
            episode_left -= 1
            label = 1
            if episode_type == "temp_spike":
                temp += random.uniform(1.5, 3.0)
            elif episode_type == "humidity_drop":
                humi -= random.uniform(2.0, 4.0)
            elif episode_type == "frozen":
                temp, humi = frozen_temp, frozen_humi
            elif episode_type == "status_fault":
                status |= 0b100
            elif episode_type == "rapid_change":
                temp += random.choice([-7.0, 7.0])
        else:
            temp = _walk(42.0, 2.0, temp)
            humi = _walk(58.0, 5.0, humi)
            status = 0b010
            frozen_temp, frozen_humi = temp, humi

        temp = max(-20.0, min(85.0, temp))
        humi = max(0.0, min(100.0, humi))

        frames.append(
            (
                {
                    "temperature": round(temp, 2),
                    "humidity": round(humi, 2),
                    "status": status,
                    "is_running": bool(status & 0b010),
                    "is_emergency": bool(status & 0b001),
                    "is_fault": bool(status & 0b100),
                },
                label,
            )
        )

    return frames


# ============================================================
# Window dataset
# ============================================================

def build_dataset(
    sequences: list[list[tuple[dict, int]]],
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Slice frame sequences into window features. A window is anomalous
    if ANY frame inside it is labeled anomalous."""
    Xs: list[list[float]] = []
    ys: list[int] = []
    feature_names: list[str] = []

    for frames in sequences:
        data_hist: deque = deque(maxlen=window_size)
        label_hist: deque = deque(maxlen=window_size)
        for data, label in frames:
            data_hist.append(
                DataPoint(timestamp=0.0, source="synthetic", data=data)
            )
            label_hist.append(label)
            if len(data_hist) < window_size:
                continue
            names, values = build_feature_vector(data_hist, DEFAULT_FEATURE_SPEC)
            if not feature_names:
                feature_names = names
            Xs.append(values)
            ys.append(1 if any(label_hist) else 0)

    if not Xs:
        raise SystemExit("no window samples produced; check the input data")
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int32),
        feature_names,
    )


# ============================================================
# Labeled CSV loader
# ============================================================

def load_csv(path: str) -> list[list[tuple[dict, int]]]:
    """Read a labeled CSV. Consecutive rows with the same run_id form one
    sequence; without run_id everything is one sequence."""
    sequences: list[list[tuple[dict, int]]] = []
    last_run = None

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            run_id = row.get("run_id")
            if run_id is not None and run_id != last_run:
                sequences.append([])
                last_run = run_id
            if not sequences:
                sequences.append([])

            status = int(row.get("status", 0))
            data = {
                "temperature": float(row["temperature"]),
                "humidity": float(row["humidity"]),
                "status": status,
                "is_running": bool(status & 0b010),
                "is_emergency": bool(status & 0b001),
                "is_fault": bool(status & 0b100),
            }
            sequences[-1].append((data, int(row["label"])))

    if not sequences or all(not s for s in sequences):
        raise SystemExit(f"no valid rows in CSV: {path}")
    return sequences


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost anomaly detector")
    parser.add_argument("--csv", help="optional labeled CSV instead of synthetic data")
    parser.add_argument("--out", help="output dir (default: test/models/xgb_anomaly)")
    args = parser.parse_args()

    if args.csv:
        print(f"[data] loading labeled CSV: {args.csv}")
        sequences = load_csv(args.csv)
    else:
        print(f"[data] generating synthetic data: {N_SEQUENCES} seqs x {FRAMES_PER_SEQ} frames")
        sequences = [simulate_sequence(i) for i in range(N_SEQUENCES)]

    # Split by sequence (welding run) so windows from the same run never
    # leak across splits. Three-way: train / val (early stopping) / test.
    seq_sets = [build_dataset([seq]) for seq in sequences]
    n = len(seq_sets)
    seq_labels = np.asarray(
        [1 if any(lbl for _, lbl in seq) else 0 for seq in sequences]
    )
    stratify = (
        seq_labels
        if np.unique(seq_labels).size == 2
        and min(np.bincount(seq_labels)) >= 2
        else None
    )
    train_idx, val_idx, test_idx = split_sequences(
        n, random_state=42, stratify=stratify
    )

    X_tr = np.vstack([seq_sets[i][0] for i in train_idx])
    y_tr = np.concatenate([seq_sets[i][1] for i in train_idx])
    X_va = np.vstack([seq_sets[i][0] for i in val_idx])
    y_va = np.concatenate([seq_sets[i][1] for i in val_idx])
    X_te = np.vstack([seq_sets[i][0] for i in test_idx])
    y_te = np.concatenate([seq_sets[i][1] for i in test_idx])
    feature_names = seq_sets[0][2]

    print(f"[split] runs: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"[data] windows: train={len(X_tr)} val={len(X_va)} test={len(X_te)} features={len(feature_names)}")
    print(f"[data] anomaly ratio train={y_tr.mean():.1%}  val={y_va.mean():.1%}  test={y_te.mean():.1%}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    y_pred = model.predict(X_va)
    print("\n========== validation evaluation ==========")
    print(f"accuracy = {accuracy_score(y_va, y_pred):.4f}")
    print(f"roc_auc  = {roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]):.4f}")
    print("confusion matrix (TN FP / FN TP):")
    print(confusion_matrix(y_va, y_pred))

    # Test set: runs the model never saw during training.
    y_pred_te = model.predict(X_te)
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_te, y_pred_te)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred_te).tolist(),
    }
    print("\n========== test evaluation (unseen runs) ==========")
    print(f"accuracy = {test_metrics['accuracy']:.4f}")
    print(f"roc_auc  = {test_metrics['roc_auc']:.4f}")
    print("confusion matrix (TN FP / FN TP):")
    print(confusion_matrix(y_te, y_pred_te))

    print("\n========== feature importance top 15 (gain) ==========")
    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    scored: list[tuple[str, float]] = []
    for key, g in gain.items():
        idx = int(key[1:]) if key.startswith("f") else -1
        label = feature_names[idx] if 0 <= idx < len(feature_names) else key
        scored.append((label, g / total))
    for label, rel in sorted(scored, key=lambda t: -t[1])[:15]:
        print(f"  {label:<32} {rel:.4f}")

    out_dir = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent.parent / "models" / "xgb_anomaly"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "model.json"))
    (out_dir / "features.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "window_size": WINDOW_SIZE,
                "feature_spec": DEFAULT_FEATURE_SPEC,
                "thresholds": {
                    "normal": 0.8,
                    "suspicious": 0.4,
                    "anomaly_prob": 0.6,
                },
                "test_metrics": test_metrics,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[ok] model saved to: {out_dir}")


if __name__ == "__main__":
    main()
