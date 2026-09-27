"""
robot_monitor v0.6 - Evaluate the steel-copper lap joint crack model

Loads models/xgb_lap_joint and reports the current model's metrics on
unseen welds (same by-weld split as training, seed 42): F1, precision,
recall, accuracy, balanced accuracy, ROC-AUC, PR-AUC, confusion matrix,
best-F1 threshold, and per-weld breakdown.

Usage:
    python evaluate_lap_joint.py
    python evaluate_lap_joint.py --cv 5
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_lap_joint import DEFAULT_XLSX, load_dataset, train_model

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "xgb_lap_joint"


def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "model.json"))
    return model


def metrics_dict(y_true, proba) -> dict:
    y_pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "pos_rate": round(float(y_true.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def best_f1_threshold(y_true, proba) -> dict:
    best = {"threshold": 0.5, "f1": 0.0}
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": round(float(t), 2), "f1": round(float(f1), 4)}
    return best


def print_metrics(m: dict, label: str) -> None:
    print(f"\n========== {label} ==========")
    print(f"n={m['n']}  pos_rate={m['pos_rate']:.1%}")
    print(f"accuracy={m['accuracy']:.4f}  balanced_acc={m['balanced_accuracy']:.4f}")
    print(f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"roc_auc={m['roc_auc']:.4f}  pr_auc={m['pr_auc']:.4f}")
    print("confusion matrix (TN FP / FN TP):")
    print(np.asarray(m["confusion_matrix"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the crack model (F1 etc.)")
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="V1 xlsx path")
    parser.add_argument("--cv", type=int, default=0, help="also run N-fold CV by weld")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_dataset(args.xlsx)
    X, y, groups = data["X"], data["y"], data["groups"]
    model = load_model()
    report = {}

    # Same by-weld holdout split as training (seed 42) -> comparable metrics.
    has_groups = len(set(groups)) >= 3
    if has_groups:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
        rest_idx, test_idx = next(gss.split(np.arange(len(X)), y, groups=groups))
        rest_groups = groups[rest_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
        tr_idx, va_idx = next(gss2.split(rest_idx, y[rest_idx], groups=rest_groups))
        print(
            f"[split] test welds = {sorted(set(groups[test_idx]))} "
            f"(train welds={sorted(set(groups[rest_idx[tr_idx]]))}, val={sorted(set(groups[rest_idx[va_idx]]))})"
        )
    else:
        _, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.25, stratify=y, random_state=args.seed
        )
        print("[warn] no weld grouping: random stratified test set")

    proba = model.predict_proba(X[test_idx])[:, 1]
    m = metrics_dict(y[test_idx], proba)
    print_metrics(m, "current model metrics (unseen welds)")
    report["test_metrics"] = m
    report["best_f1_threshold"] = best_f1_threshold(y[test_idx], proba)
    print(f"best-F1 threshold = {report['best_f1_threshold']}")

    # Per-weld breakdown.
    print("\nper-weld results (test welds):")
    weld_rows = {}
    for w in sorted(set(groups[test_idx])):
        idx = np.where(groups[test_idx] == w)[0]
        sub = metrics_dict(y[test_idx][idx], proba[idx])
        weld_rows[str(int(w))] = {"n": sub["n"], "accuracy": sub["accuracy"],
                                  "f1": sub["f1"], "confusion_matrix": sub["confusion_matrix"]}
        print(f"  weld {int(w)}: n={sub['n']} acc={sub['accuracy']:.4f} f1={sub['f1']:.4f} cm={sub['confusion_matrix']}")
    report["per_weld"] = weld_rows

    # Compare with metrics recorded at training time.
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    trained = cfg.get("test_metrics")
    if trained:
        print("\n对比训练时记录（config.json test_metrics）:")
        for k in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "f1"):
            print(f"  {k:<20} trained={trained.get(k)}  now={m.get(k)}")

    if args.cv > 1:
        accs, aucs = [], []
        kfold = GroupKFold(n_splits=args.cv)
        for tr_idx, va_idx in kfold.split(X, y, groups=groups):
            mm, _ = train_model(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], args.seed)
            p = mm.predict_proba(X[va_idx])[:, 1]
            accs.append(accuracy_score(y[va_idx], (p >= 0.5).astype(int)))
            aucs.append(roc_auc_score(y[va_idx], p))
        report["cv"] = {
            "folds": args.cv,
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
        }
        print(f"\n{args.cv}-fold CV (by weld): acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}  auc={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")

    (MODEL_DIR / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[ok] report saved to: {MODEL_DIR / 'eval_report.json'}")


if __name__ == "__main__":
    main()
