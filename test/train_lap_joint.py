"""
robot_monitor v0.6 - Steel-copper lap joint laser welding crack classifier

Trains an XGBoost model to predict weld cracking from process parameters,
using the public dataset:
    Definitive screening steel-copper lap joints V1.xlsx

Dataset summary (V1):
    - 360 rows, no missing values
    - 6 process-parameter features
    - label: "cracking in the weld metal" (yes / no), 50 yes / 310 no
    - "weld number" (1..5) groups rows from the same weld: used for
      group-aware splitting to prevent cross-weld leakage

Usage:
    python train_lap_joint.py
    python train_lap_joint.py --xlsx <path> --out models/xgb_lap_joint
    python train_lap_joint.py --cv 5

Outputs:
    models/xgb_lap_joint/model.json       XGBoost model (JSON)
    models/xgb_lap_joint/features.json    feature names (inference order)
    models/xgb_lap_joint/config.json      training info + metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from openpyxl import load_workbook
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split

DEFAULT_XLSX = (
    Path(r"D:\SEproject\dataset\online\对激光焊接的钢-铜搭接接头的数据集")
    / "Definitive screening steel-copper lap joints V1.xlsx"
)

# Keyword -> role. Robust to unit symbols (degree, micrometer) and naming variants.
ROLE_KEYWORDS = {
    "power": "power",
    "speed": "speed",
    "gas flow": "gas_flow",
    "focal": "focal",
    "angular": "angular",
    "thickness": "thickness",
    "cracking": "label",
    "weld number": "group",
}

FEATURE_ROLES = ["power", "speed", "gas_flow", "focal", "angular", "thickness"]


def _to_float(value):
    """Parse a cell value, tolerating comma decimals like '1,2'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def load_dataset(xlsx_path: str) -> dict:
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        raise SystemExit(f"empty workbook: {xlsx_path}")

    header = rows[0]
    col_role = {}
    feature_names = []
    for j, cell in enumerate(header):
        if cell is None:
            continue
        key = str(cell).strip().lower()
        for keyword, role in ROLE_KEYWORDS.items():
            if keyword in key:
                col_role[j] = role
                if role in FEATURE_ROLES:
                    feature_names.append(str(cell).strip())
                break

    if "label" not in col_role.values():
        raise SystemExit(f"label column ('cracking...') not found in {xlsx_path}")
    if "group" not in col_role.values():
        print("[warn] 'weld number' column not found; falling back to random split")

    records = []
    for row in rows[1:]:
        rec = {"features": {}, "label": None, "group": None}
        ok = True
        for j, role in col_role.items():
            if role in FEATURE_ROLES:
                v = _to_float(row[j])
                if v is None:
                    ok = False
                    break
                rec["features"][role] = v
            elif role == "label":
                text = str(row[j]).strip().lower() if row[j] is not None else ""
                if text in ("yes", "y", "1"):
                    rec["label"] = 1
                elif text in ("no", "n", "0"):
                    rec["label"] = 0
                else:
                    ok = False
            elif role == "group":
                rec["group"] = _to_float(row[j])
        if ok and rec["label"] is not None and len(rec["features"]) == len(FEATURE_ROLES):
            records.append(rec)

    if not records:
        raise SystemExit(f"no valid rows parsed from {xlsx_path}")

    X = np.asarray(
        [[rec["features"][r] for r in FEATURE_ROLES] for rec in records],
        dtype=np.float32,
    )
    y = np.asarray([rec["label"] for rec in records], dtype=np.int32)
    groups = np.asarray(
        [int(rec["group"]) if rec["group"] is not None else -1 for rec in records],
        dtype=np.int32,
    )
    return {"X": X, "y": y, "groups": groups, "feature_names": feature_names, "n": len(records)}


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


def report_metrics(y_true, proba, label: str) -> dict:
    y_pred = (proba >= 0.5).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "pos_rate": round(float(y_true.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    print(f"\n========== {label} ==========")
    print(f"n={metrics['n']}  pos_rate={metrics['pos_rate']:.1%}")
    print(f"accuracy={metrics['accuracy']:.4f}  bal_acc={metrics['balanced_accuracy']:.4f}")
    print(f"roc_auc={metrics['roc_auc']:.4f}  pr_auc={metrics['pr_auc']:.4f}  f1={metrics['f1']:.4f}")
    print("confusion matrix (TN FP / FN TP):")
    print(np.asarray(metrics["confusion_matrix"]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train steel-copper lap joint crack classifier")
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="path to V1 xlsx")
    parser.add_argument("--out", default=None, help="output dir (default: models/xgb_lap_joint)")
    parser.add_argument("--cv", type=int, default=0, help="run N-fold CV (by weld) instead of holdout")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_dataset(args.xlsx)
    X, y, groups = data["X"], data["y"], data["groups"]
    feature_names = data["feature_names"]
    print(f"[data] {data['n']} rows from {args.xlsx}")
    print(f"[data] features={len(feature_names)}  label 0/1 = {(y == 0).sum()}/{(y == 1).sum()}")
    print(f"[data] weld groups={sorted(set(groups))}")

    out_dir = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent.parent / "models" / "xgb_lap_joint"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    has_groups = len(set(groups)) >= 3
    report = {"dataset": args.xlsx, "feature_names": feature_names, "split": None}

    if args.cv > 1:
        accs, aucs = [], []
        if has_groups:
            kfold = GroupKFold(n_splits=args.cv)
            split_kwargs = {"groups": groups}
            print(f"\n========== {args.cv}-fold CV (by weld) ==========")
        else:
            kfold = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
            split_kwargs = {}
            print(f"\n========== {args.cv}-fold CV (stratified) ==========")
        for tr_idx, va_idx in kfold.split(X, y, **split_kwargs):
            m, _ = train_model(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], args.seed)
            proba = m.predict_proba(X[va_idx])[:, 1]
            accs.append(accuracy_score(y[va_idx], (proba >= 0.5).astype(int)))
            aucs.append(roc_auc_score(y[va_idx], proba))
        report["cv"] = {
            "folds": args.cv,
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
        }
        print(f"accuracy={np.mean(accs):.4f} +/- {np.std(accs):.4f}  auc={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Holdout: group-aware (by weld) when possible.
    if has_groups:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
        rest_idx, test_idx = next(gss.split(np.arange(len(X)), y, groups=groups))
        rest_groups = groups[rest_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
        tr_idx, va_idx = next(gss2.split(rest_idx, y[rest_idx], groups=rest_groups))
        tr_idx, va_idx = rest_idx[tr_idx], rest_idx[va_idx]
        report["split"] = "by_weld"
        print(
            f"[split] by weld number: train welds={sorted(set(groups[tr_idx]))} "
            f"val welds={sorted(set(groups[va_idx]))} test welds={sorted(set(groups[test_idx]))}"
        )
        overlap = set(groups[tr_idx]) & set(groups[test_idx])
        assert not overlap, f"leakage: welds in both train and test: {overlap}"
    else:
        tr_idx, rest_idx = train_test_split(
            np.arange(len(X)), test_size=0.25, stratify=y, random_state=args.seed
        )
        va_idx, test_idx = train_test_split(
            rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=args.seed
        )
        report["split"] = "random_stratified"
        print("[warn] no weld grouping: random stratified split (NOT leakage-safe)")

    model, scale_pos_weight = train_model(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], args.seed)
    test_metrics = report_metrics(
        y[test_idx], model.predict_proba(X[test_idx])[:, 1], "test evaluation (unseen welds)"
    )
    report["test_metrics"] = test_metrics
    report["scale_pos_weight"] = round(float(scale_pos_weight), 4)
    report["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    model.save_model(str(out_dir / "model.json"))
    (out_dir / "features.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "config.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[ok] model saved to: {out_dir}")


if __name__ == "__main__":
    main()
