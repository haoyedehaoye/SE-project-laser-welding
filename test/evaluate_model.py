"""
robot_monitor v0.7 - Evaluate a trained model on the lap-joint V1 dataset

说明：本脚本原用于评估公开机器人数据集模型（按 task_type / object_class
分组）。2026-09 起适配基于钢-铜搭接接头 V1 数据训练的模型：按 features.json
自动取列，按 weld number（默认分组列）输出逐焊缝结果。

Usage:
    python evaluate_model.py --model-dir models/xgb_public
    python evaluate_model.py --model-dir models/xgb_public_group --cv 5
    python evaluate_model.py --model-dir models/xgb_public_sensor_only --test-size 0
    python evaluate_model.py --model-dir models/xgb_public --xlsx <path> --test-size 0

What it reports:
    - standard metrics: accuracy, balanced accuracy, ROC-AUC, PR-AUC,
      Brier score, confusion matrix
    - three-class mapping used by the project (normal/suspicious/anomaly)
    - per-weld accuracy
    - best positive-class threshold by F1
    - optional N-fold cross-validation (by weld when grouped)

Output:
    <model_dir>/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    train_test_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_lap_joint import DEFAULT_XLSX, FEATURE_ROLES, load_dataset

# Project three-class mapping (see xgb_checker.py):
#   score = 1 - p_anomaly ; normal: score >= 0.8  (p <= 0.2)
#   suspicious: 0.4 <= score < 0.8  (0.2 < p <= 0.6)
#   anomaly: score < 0.4            (p > 0.6)
NORMAL_MAX_P = 0.2
ANOMALY_MIN_P = 0.6


def load_model_artifacts(model_dir: Path) -> tuple:
    model = xgb.XGBClassifier()
    model.load_model(str(model_dir / "model.json"))
    feature_names = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    return model, feature_names


def subset_X(X: np.ndarray, full_names: list[str], want_names: list[str]) -> np.ndarray:
    """Select model features from the full 6-feature matrix (by name)."""
    idx = []
    for want in want_names:
        low = want.lower()
        if want in full_names:
            idx.append(full_names.index(want))
            continue
        # Keyword fallback: map "gas flow rate (l/min)" -> "gas_flow", etc.
        matched = None
        for role in FEATURE_ROLES:
            if role.replace("_", " ") in low or role in low:
                matched = full_names[FEATURE_ROLES.index(role)]
                break
        if matched is None:
            raise SystemExit(f"cannot map model feature {want!r} to V1 columns")
        idx.append(full_names.index(matched))
    return X[:, idx].astype(np.float32, copy=False)


def three_class_labels(proba: np.ndarray) -> np.ndarray:
    labels = []
    for p in proba:
        if p <= NORMAL_MAX_P:
            labels.append("normal")
        elif p <= ANOMALY_MIN_P:
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


def per_weld_accuracy(
    groups: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    result = {}
    for w in sorted(set(groups)):
        idx = np.where(groups == w)[0]
        result[str(int(w))] = {
            "n": int(len(idx)),
            "accuracy": round(float(accuracy_score(y_true[idx], y_pred[idx])), 4),
        }
    return result


def run_metrics(y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> dict:
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

    # Project three-class mapping.
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
    report["per_weld"] = per_weld_accuracy(groups, y_true, y_pred)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the lap-joint V1 dataset"
    )
    parser.add_argument("--model-dir", default="models/xgb_public")
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="V1 xlsx path")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="holdout size (0 = evaluate all rows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=0,
                        help="run N-fold cross-validation instead of holdout")
    parser.add_argument("--group-col", default="weld number",
                        help="grouping column (default: weld number)")
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parent.parent / args.model_dir
    model, feature_names = load_model_artifacts(model_dir)
    print(f"[model] {model_dir}")
    print(f"[model] features={len(feature_names)}")

    data = load_dataset(args.xlsx)
    X = subset_X(data["X"], data["feature_names"], feature_names)
    y, groups = data["y"], data["groups"]
    print(f"[data] {data['n']} rows, {X.shape[1]} features, pos rate {y.mean():.1%}")
    print(f"[split] grouped evaluation by {args.group_col}: welds={sorted(set(groups))}")

    report: dict = {}
    if args.cv > 1:
        n_folds = min(args.cv, len(set(groups)))
        if n_folds < args.cv:
            print(f"[split] folds > welds; using {n_folds} folds")
        kfold = GroupKFold(n_splits=n_folds)
        accs, aucs = [], []
        for tr_idx, va_idx in kfold.split(X, y, groups=groups):
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=float((y[tr_idx] == 0).sum() / max((y[tr_idx] == 1).sum(), 1)),
                random_state=args.seed, n_jobs=-1,
            )
            m.fit(X[tr_idx], y[tr_idx])
            proba = m.predict_proba(X[va_idx])[:, 1]
            accs.append(accuracy_score(y[va_idx], (proba >= 0.5).astype(int)))
            aucs.append(roc_auc_score(y[va_idx], proba))
        report["cv"] = {
            "folds": n_folds,
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
        }
        print(f"\n========== {n_folds}-fold CV (by weld) ==========")
        print(f"accuracy = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"roc_auc  = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    if args.test_size >= 0 and (args.cv <= 1 or args.test_size > 0):
        if args.test_size > 0:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=args.test_size, random_state=args.seed
            )
            tr_idx, va_idx = next(gss.split(np.arange(len(X)), y, groups=groups))
            overlap = set(groups[tr_idx]) & set(groups[va_idx])
            assert not overlap, f"leakage: welds in both splits: {overlap}"
        else:
            va_idx = np.arange(len(X))
        proba = model.predict_proba(X[va_idx])[:, 1]
        y_va, groups_va = y[va_idx], groups[va_idx]

        print(f"\n========== holdout evaluation (n={len(va_idx)}) ==========")
        m = run_metrics(y_va, proba, groups_va)
        report["holdout"] = m
        print(f"accuracy      = {m['accuracy']:.4f}")
        print(f"bal. accuracy = {m['balanced_accuracy']:.4f}")
        print(f"roc_auc       = {m['roc_auc']:.4f}")
        print(f"pr_auc        = {m['pr_auc']:.4f}")
        print(f"brier         = {m['brier']:.4f}")
        print(f"precision/recall/f1 = {m['precision']} / {m['recall']} / {m['f1']}")
        print("confusion matrix (TN FP / FN TP):")
        print(np.asarray(m["confusion_matrix"]))
        print(f"best threshold by F1 = {m['best_threshold_by_f1']}")
        print("\nthree-class mapping:")
        for k, v in m["three_class"].items():
            print(f"  {k}: {v}")
        print("\nper-weld accuracy:")
        for k, v in m["per_weld"].items():
            print(f"  weld {k:<4} n={v['n']:<5} acc={v['accuracy']}")

    report["split_mode"] = "by_weld:" + args.group_col
    report["model_dir"] = str(model_dir)
    (model_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[ok] report saved to: {model_dir / 'eval_report.json'}")


if __name__ == "__main__":
    main()
