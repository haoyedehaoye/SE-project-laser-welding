"""
robot_monitor v0.4 - Evaluate a trained model on held-out data

Usage:
    python evaluate_model.py --model-dir models/xgb_public
    python evaluate_model.py --model-dir models/xgb_public_sensor_only --test-size 0.3
    python evaluate_model.py --model-dir models/xgb_public --csv new_data.csv --test-size 0
    python evaluate_model.py --model-dir models/xgb_public --cv 5

What it reports:
    - standard metrics: accuracy, balanced accuracy, ROC-AUC, PR-AUC,
      Brier score, classification report, confusion matrix
    - three-class mapping used by the project (normal/suspicious/anomaly)
    - per task_type / object_class accuracy (weld matters most for us)
    - best positive-class threshold by F1
    - optional 5-fold cross-validation

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
    classification_report,
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

from train_public_dataset import DEFAULT_CSV, build_X, load_rows

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
    enc = json.loads((model_dir / "encoders.json").read_text(encoding="utf-8"))
    task_map = {str(k): int(v) for k, v in enc["task_type"].items()}
    obj_map = {str(k): int(v) for k, v in enc["object_class"].items()}
    include_img = any(n.startswith("img_feat_") for n in feature_names)
    return model, feature_names, task_map, obj_map, include_img


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


def per_group_accuracy(
    rows: list[dict], y_true: np.ndarray, y_pred: np.ndarray, col: str
) -> dict:
    result = {}
    for group in sorted({r[col] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r[col] == group]
        if not idx:
            continue
        result[group] = {
            "n": len(idx),
            "accuracy": round(float(accuracy_score(y_true[idx], y_pred[idx])), 4),
        }
    return result


def run_metrics(y_true: np.ndarray, proba: np.ndarray, rows: list[dict]) -> dict:
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
    table: dict = {}
    for true_val, label in [(0, "true=0"), (1, "true=1")]:
        idx = np.where(y_true == true_val)[0]
        sub = labels[idx]
        table[f"true={true_val}"] = {
            "normal": int((sub == "normal").sum()),
            "suspicious": int((sub == "suspicious").sum()),
            "anomaly": int((sub == "anomaly").sum()),
        }
    report["three_class"] = table

    report["per_task"] = per_group_accuracy(rows, y_true, y_pred, "task_type")
    report["per_object"] = per_group_accuracy(rows, y_true, y_pred, "object_class")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost model")
    parser.add_argument("--model-dir", default="models/xgb_public")
    parser.add_argument("--csv", default=None, help="dataset CSV (default: public dataset)")
    parser.add_argument("--test-size", type=float, default=0.2, help="holdout size (0 = evaluate all rows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=0, help="run N-fold cross-validation instead of holdout")
    parser.add_argument("--group-col", default=None,
                        help="column that identifies one welding run (e.g. run_id)")
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parent.parent / args.model_dir
    model, feature_names, task_map, obj_map, include_img = load_model_artifacts(model_dir)
    print(f"[model] {model_dir}")
    print(f"[model] features={len(feature_names)}  include_img={include_img}")

    csv_path = args.csv or str(DEFAULT_CSV)
    rows = load_rows(csv_path)
    X, names = build_X(rows, task_map, obj_map, include_img)
    if X.shape[1] != len(feature_names):
        raise SystemExit(
            f"feature mismatch: built {X.shape[1]} cols, model expects {len(feature_names)}"
        )
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    print(f"[data] {len(rows)} rows, {X.shape[1]} features, pos rate {y.mean():.1%}")

    group_col = args.group_col or ("run_id" if "run_id" in rows[0] else None)
    groups = (
        np.asarray([str(r.get(group_col, "")) for r in rows])
        if group_col else None
    )
    if group_col:
        n_groups = len(set(groups))
        if n_groups < 3:
            raise SystemExit(f"group column {group_col!r} has only {n_groups} groups")
        print(f"[split] run-level evaluation by {group_col}: {n_groups} runs")
    else:
        print("[split] no run_id column: random/stratified evaluation (NOT leakage-safe)")

    report: dict = {}
    if args.cv > 1:
        if groups is not None:
            if args.cv > len(set(groups)):
                args.cv = len(set(groups))
                print(f"[split] folds > runs; using {args.cv} folds")
            kfold = GroupKFold(n_splits=args.cv)
            print("[split] CV uses GroupKFold (each fold = whole runs)")
        else:
            kfold = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        accs, aucs = [], []
        split_kwargs = {"groups": groups} if groups is not None else {}
        for tr_idx, va_idx in kfold.split(X, y, **split_kwargs):
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8,
                scale_pos_weight=float((y[tr_idx] == 0).sum() / max((y[tr_idx] == 1).sum(), 1)),
                random_state=args.seed, n_jobs=-1,
            )
            m.fit(X[tr_idx], y[tr_idx])
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
        print(f"\n========== {args.cv}-fold CV ==========")
        print(f"accuracy = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"roc_auc  = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    if args.test_size >= 0 and (args.cv <= 1 or args.test_size > 0):
        if args.test_size > 0:
            if groups is not None:
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=args.test_size, random_state=args.seed
                )
                tr_idx, va_idx = next(gss.split(np.arange(len(rows)), y, groups=groups))
                overlap = set(groups[tr_idx]) & set(groups[va_idx])
                assert not overlap, f"leakage: runs in both splits: {overlap}"
            else:
                tr_idx, va_idx = train_test_split(
                    np.arange(len(rows)), test_size=args.test_size,
                    stratify=y, random_state=args.seed,
                )
        else:
            va_idx = np.arange(len(rows))
        proba = model.predict_proba(X[va_idx])[:, 1]
        y_va = y[va_idx]
        rows_va = [rows[i] for i in va_idx]

        print(f"\n========== holdout evaluation (n={len(va_idx)}) ==========")
        m = run_metrics(y_va, proba, rows_va)
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
        print("\nper task_type accuracy:")
        for k, v in m["per_task"].items():
            print(f"  {k:<10} n={v['n']:<5} acc={v['accuracy']}")
        print("\nper object_class accuracy:")
        for k, v in m["per_object"].items():
            print(f"  {k:<10} n={v['n']:<5} acc={v['accuracy']}")

    report["split_mode"] = ("by_run:" + group_col) if group_col else "random"
    report["model_dir"] = str(model_dir)
    (model_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[ok] report saved to: {model_dir / 'eval_report.json'}")


if __name__ == "__main__":
    main()
