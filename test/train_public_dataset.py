"""
robot_monitor v0.7 - Train the steel-copper lap joint crack model variants

说明：本脚本原名「公开数据集训练脚本」，原用于公开工业机器人传感器与
视觉融合数据集（133 维）。2026-09 起改为基于钢-铜搭接接头 V1 数据集，
训练 xgb_public 系列三个变体（均为逐行分类：工艺参数 → 裂纹是/否）。

Variants:
    xgb_public                6 工艺参数，随机分层切分（不防泄漏，对比用）
    xgb_public_group          6 工艺参数，按 weld number 分组切分（防泄漏）
    xgb_public_sensor_only    4 核心工艺参数（功率/速度/气流量/板厚），随机分层切分

Usage:
    python train_public_dataset.py                         # 训练全部 3 个变体
    python train_public_dataset.py --variant group         # 只训练其中一个
    python train_public_dataset.py --xlsx <path> --seed 42

Outputs per variant directory:
    model.json        XGBoost 模型（JSON）
    features.json     特征名（推理顺序）
    encoders.json     标签映射（no -> 0, yes -> 1）
    config.json       训练信息 + 指标
    eval_report.json  留出集评估报告
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_lap_joint import DEFAULT_XLSX, FEATURE_ROLES, load_dataset


VARIANTS = {
    "public": {
        "out_dir": "xgb_public",
        "desc": "V1 全特征版（6 工艺参数，随机分层切分）",
        "features": list(FEATURE_ROLES),
        "split": "random",
    },
    "group": {
        "out_dir": "xgb_public_group",
        "desc": "V1 全特征版（6 工艺参数，按焊缝分组切分，防泄漏）",
        "features": list(FEATURE_ROLES),
        "split": "by_weld",
    },
    "sensor_only": {
        "out_dir": "xgb_public_sensor_only",
        "desc": "V1 核心工艺参数子集版（功率/速度/气流量/板厚，消融）",
        "features": ["power", "speed", "gas_flow", "thickness"],
        "split": "random",
    },
}


def build_X(records_X: np.ndarray, roles: list[str]) -> np.ndarray:
    """Select the variant's feature columns from the full 6-feature matrix.

    records_X is built by train_lap_joint.load_dataset with columns in
    FEATURE_ROLES order.
    """
    idx = [FEATURE_ROLES.index(r) for r in roles]
    return records_X[:, idx].astype(np.float32, copy=False)


def train_model(X_tr, y_tr, X_va, y_va, seed: int = 42):
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        early_stopping_rounds=30,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model, scale_pos_weight


def three_class_labels(proba: np.ndarray) -> np.ndarray:
    # Same mapping as the project's xgb_checker: score = 1 - p.
    labels = []
    for p in proba:
        if p <= 0.2:
            labels.append("normal")
        elif p <= 0.6:
            labels.append("suspicious")
        else:
            labels.append("anomaly")
    return np.asarray(labels)


def best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> dict:
    best = {"threshold": 0.5, "f1": 0.0}
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": round(float(t), 2), "f1": round(float(f1), 4)}
    return best


def run_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    report = {
        "n": int(len(y_true)),
        "pos_rate": round(float(y_true.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "best_threshold_by_f1": best_threshold_by_f1(y_true, proba),
    }
    labels = three_class_labels(proba)
    report["three_class"] = {
        "true=0": {
            "normal": int(((y_true == 0) & (labels == "normal")).sum()),
            "suspicious": int(((y_true == 0) & (labels == "suspicious")).sum()),
            "anomaly": int(((y_true == 0) & (labels == "anomaly")).sum()),
        },
        "true=1": {
            "normal": int(((y_true == 1) & (labels == "normal")).sum()),
            "suspicious": int(((y_true == 1) & (labels == "suspicious")).sum()),
            "anomaly": int(((y_true == 1) & (labels == "anomaly")).sum()),
        },
    }
    return report


def train_variant(
    variant: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    xlsx_path: str,
    seed: int,
) -> dict:
    vdef = VARIANTS[variant]
    roles = vdef["features"]
    Xv = build_X(X, roles)
    names = [feature_names[FEATURE_ROLES.index(r)] for r in roles]
    print(f"\n========== variant: {variant} ==========")
    print(f"[data] rows={len(y)}  features={len(names)}  split={vdef['split']}")

    if vdef["split"] == "by_weld":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        rest_idx, test_idx = next(gss.split(np.arange(len(Xv)), y, groups=groups))
        rest_groups = groups[rest_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr_idx, va_idx = next(gss2.split(rest_idx, y[rest_idx], groups=rest_groups))
        tr_idx, va_idx = rest_idx[tr_idx], rest_idx[va_idx]
        overlap = set(groups[tr_idx]) & set(groups[test_idx])
        assert not overlap, f"leakage: welds in both train and test: {overlap}"
        split_mode = "by_weld"
        print(
            f"[split] by weld number: train={sorted(set(groups[tr_idx]))} "
            f"val={sorted(set(groups[va_idx]))} test={sorted(set(groups[test_idx]))}"
        )
    else:
        tr_idx, rest_idx = train_test_split(
            np.arange(len(Xv)), test_size=0.3, stratify=y, random_state=seed
        )
        va_idx, test_idx = train_test_split(
            rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=seed
        )
        split_mode = "random_stratified"
        print("[split] random stratified split (NOT leakage-safe)")

    model, scale_pos_weight = train_model(
        Xv[tr_idx], y[tr_idx], Xv[va_idx], y[va_idx], seed
    )
    proba = model.predict_proba(Xv[test_idx])[:, 1]
    metrics = run_metrics(y[test_idx], proba)
    print(
        f"accuracy={metrics['accuracy']:.4f}  bal_acc={metrics['balanced_accuracy']:.4f}  "
        f"roc_auc={metrics['roc_auc']:.4f}  f1={metrics['f1']:.4f}"
    )
    print("confusion matrix (TN FP / FN TP):")
    print(np.asarray(metrics["confusion_matrix"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "model.json"))
    (out_dir / "features.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "encoders.json").write_text(
        json.dumps({"label": {"no": 0, "yes": 1}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "dataset": xlsx_path,
        "variant": variant,
        "description": vdef["desc"],
        "feature_names": names,
        "split_mode": split_mode,
        "test_metrics": metrics,
        "scale_pos_weight": round(float(scale_pos_weight), 4),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "config.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    eval_report = {
        "dataset": xlsx_path,
        "variant": variant,
        "description": vdef["desc"],
        "split_mode": split_mode,
        "holdout": metrics,
        "model_dir": str(out_dir),
    }
    (out_dir / "eval_report.json").write_text(
        json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] saved to: {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train xgb_public variants on the steel-copper lap joint V1 dataset"
    )
    parser.add_argument(
        "--variant",
        choices=["all", *VARIANTS.keys(), "sensor-only"],
        default="all",
        help="which variant(s) to train (default: all)",
    )
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="V1 xlsx path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_dataset(args.xlsx)
    X, y, groups = data["X"], data["y"], data["groups"]
    feature_names = data["feature_names"]
    print(f"[data] {data['n']} rows from {args.xlsx}")
    print(f"[data] full features={len(feature_names)}  label 0/1 = {(y == 0).sum()}/{(y == 1).sum()}")

    models_root = Path(__file__).resolve().parent.parent / "models"
    variant_name = "sensor_only" if args.variant == "sensor-only" else args.variant
    variants = [variant_name] if variant_name != "all" else list(VARIANTS)
    for v in variants:
        out_dir = models_root / VARIANTS[v]["out_dir"]
        train_variant(v, X, y, groups, feature_names, out_dir, args.xlsx, args.seed)


if __name__ == "__main__":
    main()
