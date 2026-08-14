"""
robot_monitor v0.4 - Train XGBoost on the public industrial robot dataset

Dataset: "Industrial Robot Sensor and Vision Fusion Dataset" (10,000 rows).
Each row is one observation with force/proximity/temperature sensors,
task_type / object_class categories, 128 image features, and a binary label.

This is a ROW-LEVEL classifier (one observation -> label), so it does NOT
use the sliding-window features of xgb_checker.py. It is meant to prove
the XGBoost pipeline on real-world data and can later be wired into the
production multi-sensor (sensor + vision) pipeline.

Usage:
    python train_public_dataset.py
    python train_public_dataset.py --sensor-only
    python train_public_dataset.py --csv <path> --out <dir>

Outputs:
    models/xgb_public/model.json        XGBoost model (JSON)
    models/xgb_public/features.json     feature names (inference order)
    models/xgb_public/encoders.json     category -> id mappings
    models/xgb_public/config.json       metrics + training info
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DEFAULT_CSV = (
    Path(r"D:\SEproject\dataset\online\工业机器人传感器与视觉融合数据集\archive (2)")
    / "Industrial Robot Sensor and Vision Fusion Dataset.csv"
)

IMG_FEATURES = [f"img_feat_{i}" for i in range(1, 129)]
SENSOR_FEATURES = ["force_sensor", "proximity_sensor", "temperature_sensor"]
CAT_FEATURES = ["task_type_enc", "object_class_enc"]


def load_rows(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise SystemExit(f"no rows in CSV: {csv_path}")
    return rows


def build_X(
    rows: list[dict],
    task_map: dict,
    obj_map: dict,
    include_img: bool,
) -> tuple[np.ndarray, list[str]]:
    names = list(SENSOR_FEATURES + CAT_FEATURES)
    if include_img:
        names += IMG_FEATURES

    X: list[list[float]] = []
    for row in rows:
        vals: list[float] = [
            float(row["force_sensor"]),
            float(row["proximity_sensor"]),
            float(row["temperature_sensor"]),
            float(task_map.get(row["task_type"], -1)),
            float(obj_map.get(row["object_class"], -1)),
        ]
        if include_img:
            vals += [float(row[f]) for f in IMG_FEATURES]
        X.append(vals)
    return np.asarray(X, dtype=np.float32), names


def build_maps(rows: list[dict]) -> tuple[dict, dict]:
    task_map = {v: i for i, v in enumerate(sorted({r["task_type"] for r in rows}))}
    obj_map = {v: i for i, v in enumerate(sorted({r["object_class"] for r in rows}))}
    return task_map, obj_map


def train_variant(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    names: list[str],
    tag: str,
    out_dir: Path,
) -> dict:
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    print(f"\n========== variant: {tag} ==========")
    print(f"[data] train={len(X_tr)}  val={len(X_va)}  features={len(names)}")
    print(f"[data] class balance train: 0={neg}  1={pos}  scale_pos_weight={scale_pos_weight:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=30,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    y_pred = model.predict(X_va)
    proba = model.predict_proba(X_va)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_va, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_va, proba), 4),
        "confusion_matrix": confusion_matrix(y_va, y_pred).tolist(),
    }
    print(f"accuracy = {metrics['accuracy']:.4f}")
    print(f"roc_auc  = {metrics['roc_auc']:.4f}")
    print("confusion matrix (TN FP / FN TP):")
    print(confusion_matrix(y_va, y_pred))
    print(classification_report(y_va, y_pred, digits=4))

    print(f"\nfeature importance top 20 ({tag})")
    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    scored: list[tuple[str, float]] = []
    for key, g in gain.items():
        idx = int(key[1:]) if key.startswith("f") else -1
        label = names[idx] if 0 <= idx < len(names) else key
        scored.append((label, g / total))
    for label, rel in sorted(scored, key=lambda t: -t[1])[:20]:
        print(f"  {label:<24} {rel:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "model.json"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on the public robot dataset")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="path to dataset CSV")
    parser.add_argument("--sensor-only", action="store_true", help="exclude image features")
    parser.add_argument("--out", default=None, help="output dir (default: test/models/xgb_public)")
    parser.add_argument("--group-col", default=None,
                        help="column that identifies one welding run (e.g. run_id)")
    args = parser.parse_args()

    include_img = not args.sensor_only
    rows = load_rows(args.csv)
    print(f"[data] loaded {len(rows)} rows from {args.csv}")

    task_map, obj_map = build_maps(rows)
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)

    group_col = args.group_col or ("run_id" if "run_id" in rows[0] else None)
    if group_col:
        groups = np.asarray([str(r.get(group_col, "")) for r in rows])
        n_groups = len(set(groups))
        if n_groups < 3:
            raise SystemExit(
                f"group column {group_col!r} has only {n_groups} groups; "
                "need >= 3 welding runs for a run-level split"
            )
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_tr, idx_va = next(gss.split(np.arange(len(rows)), y, groups=groups))
        overlap = set(groups[idx_tr]) & set(groups[idx_va])
        assert not overlap, f"leakage: runs in both splits: {overlap}"
        print(
            f"[split] run-level split by {group_col}: "
            f"train runs={len(set(groups[idx_tr]))} val runs={len(set(groups[idx_va]))}"
        )
    else:
        idx_tr, idx_va = train_test_split(
            np.arange(len(rows)), test_size=0.2, stratify=y, random_state=42
        )
        print("[split] no run_id column: random stratified split (NOT leakage-safe)")
    rows_tr = [rows[i] for i in idx_tr]
    rows_va = [rows[i] for i in idx_va]
    y_tr, y_va = y[idx_tr], y[idx_va]

    X_tr, names = build_X(rows_tr, task_map, obj_map, include_img)
    X_va, _ = build_X(rows_va, task_map, obj_map, include_img)

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "models" / "xgb_public"

    metrics = train_variant(X_tr, y_tr, X_va, y_va, names, "sensor+vision", out_dir)

    (out_dir / "features.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "encoders.json").write_text(
        json.dumps({"task_type": task_map, "object_class": obj_map}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": args.csv,
                "include_image_features": include_img,
                "split_mode": ("by_run:" + group_col) if group_col else "random_stratified",
                "metrics": metrics,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Self-check: predict a few validation rows.
    model = xgb.XGBClassifier()
    model.load_model(str(out_dir / "model.json"))
    sample = X_va[:5]
    prob = model.predict_proba(sample)[:, 1]
    print("\n[self-check] first 5 validation rows, anomaly/success probability:")
    for i, p in enumerate(prob):
        print(f"  row idx_va[{i}] -> p(class=1) = {p:.4f}  true={int(y_va[i])}")
    print(f"\n[ok] model saved to: {out_dir}")


if __name__ == "__main__":
    main()
