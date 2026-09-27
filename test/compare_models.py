# -*- coding: utf-8 -*-
"""
robot_monitor v0.9 — 分类模型选型与对比（要求 2：Select ≥3 classification models）

在项目【两个】现有分类任务上，用同一组模型做同协议对比：
  - task=crack   V1 真实数据：6 工艺参数(+横截面位置) → 裂纹二分类
                （按 weld number 分组，GroupKFold）
  - task=window  合成 32 帧窗口特征 → 正常/异常二分类（xgb_anomaly 任务，
                按焊接序列分组防泄漏，数据为合成，结论仅作流程参考）
模型（≥3 个现有分类模型）：
  XGBoost / RandomForest / ExtraTrees / HistGradientBoosting / LogisticRegression

用法：
  python compare_models.py --task crack            # V1 裂纹
  python compare_models.py --task window           # 32帧窗口质检（合成数据）
产物（JSON，含 selected_model 建议）：
  crack  → models/xgb_multiout/model_selection.json
  window → models/xgb_anomaly/model_selection.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
MODEL_MULTI = ROOT / "models" / "xgb_multiout"
MODEL_ANOMALY = ROOT / "models" / "xgb_anomaly"


def _factory(seed: int) -> dict:
    def _xgb(tr_x, tr_y):
        neg, pos = int((tr_y == 0).sum()), int((tr_y == 1).sum())
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss",
            scale_pos_weight=(neg / pos if pos > 0 else 1.0),
            random_state=seed, n_jobs=-1).fit(tr_x, tr_y)

    def _rf(tr_x, tr_y):
        return RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1).fit(tr_x, tr_y)

    def _et(tr_x, tr_y):
        return ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1).fit(tr_x, tr_y)

    def _hist(tr_x, tr_y):
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
            random_state=seed).fit(tr_x, tr_y)

    def _lr(tr_x, tr_y):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=seed)).fit(tr_x, tr_y)

    return {"XGBoost": _xgb, "RandomForest": _rf, "ExtraTrees": _et,
            "HistGB": _hist, "LogisticReg": _lr}


def _pooled(y_all, p_all, g_all) -> dict:
    y_all, p_all, g_all = map(np.asarray, (y_all, p_all, g_all))
    pred = (p_all >= 0.5).astype(int)
    m = {
        "n": int(len(y_all)),
        "pos_rate": round(float(y_all.mean()), 4),
        "accuracy": round(float(accuracy_score(y_all, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_all, pred)), 4),
        "f1": round(float(f1_score(y_all, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_all, p_all)), 4)
        if len(np.unique(y_all)) == 2 else None,
        "pr_auc": round(float(average_precision_score(y_all, p_all)), 4),
    }
    weld = []
    for w in sorted(set(g_all.tolist())):
        mask = g_all == w
        if mask.sum() == 0:
            continue
        weld.append(int((p_all[mask].mean() >= 0.5) == (y_all[mask].mean() >= 0.5)))
    m["group_accuracy"] = round(float(np.mean(weld)), 4) if weld else None
    m["n_groups"] = len(weld)
    return m


# ---------------------------------------------------------------------------
# task=crack（V1 真实数据）
# ---------------------------------------------------------------------------

def load_crack():
    from train_multiout import DEFAULT_XLSX, load_records, to_matrix
    data = load_records(str(DEFAULT_XLSX))
    X, y, g, input_keys = to_matrix(data["records"], with_pos=True)
    return {"X": X, "y": y, "g": g, "input_features": input_keys,
            "n_rows": data["n"], "n_crack": data["n_crack"],
            "note": "V1 真实数据；按 weld number 分组 GroupKFold(5)"}


# ---------------------------------------------------------------------------
# task=window（合成 32 帧窗口质检，xgb_anomaly 任务）
# ---------------------------------------------------------------------------

def load_window(seqs: int, seed: int):
    import train_xgboost as tx
    tx.random.seed(seed)
    Xs, ys = [], []
    groups = []
    names = None
    for s in range(seqs):
        seq = tx.simulate_sequence(s)
        Xi, yi, names_i = tx.build_dataset([seq])
        Xs.append(Xi)
        ys.append(yi)
        groups.extend([s] * len(yi))
        names = names_i
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int32)
    g = np.asarray(groups, dtype=np.int32)
    return {"X": X, "y": y, "g": g, "input_features": names,
            "n_rows": int(len(y)), "n_anomaly": int(y.sum()),
            "note": f"合成 32 帧窗口({seqs} 序列 × 200 帧)；按焊接序列分组 GroupKFold(5)——数据为合成，结论仅作流程参考"}


# ---------------------------------------------------------------------------
# 通用 runner
# ---------------------------------------------------------------------------

def run_task(task: str, cv: int, seed: int, seqs: int) -> dict:
    ds = load_crack() if task == "crack" else load_window(seqs, seed)
    X, y, g = ds["X"], ds["y"], ds["g"]
    if len(set(g.tolist())) < cv:
        raise SystemExit(f"分组数 {len(set(g.tolist()))} < CV 折数 {cv}，请调小 --cv")
    folds = list(GroupKFold(n_splits=cv).split(X, y, groups=g))
    results = {}
    for name, fit_fn in _factory(seed).items():
        t0 = time.time()
        y_all, p_all, g_all = [], [], []
        for tr, te in folds:
            model = fit_fn(X[tr], y[tr])
            p_all.extend(model.predict_proba(X[te])[:, 1])
            y_all.extend(y[te].tolist())
            g_all.extend(g[te].tolist())
        m = _pooled(y_all, p_all, g_all)
        m["elapsed_s"] = round(time.time() - t0, 1)
        results[name] = m
        print(f"[{name:<12}] acc={m['accuracy']} bal={m['balanced_accuracy']} "
              f"F1={m['f1']} AUC={m['roc_auc']} PR-AUC={m['pr_auc']} "
              f"group_acc={m['group_accuracy']} ({m['elapsed_s']}s)")
    best = max(results, key=lambda k: (results[k]["roc_auc"] or 0, results[k]["pr_auc"] or 0))
    meta = {k: v for k, v in ds.items() if k not in ("X", "y", "g")}  # 排除大数组
    return {"task": task, **meta, "cv_folds": cv, "seed": seed,
            "models": results, "selected_model": best,
            "selected_note": ("ROC-AUC 最高（平手取 PR-AUC）；差异 < 数据不确定度时建议沿用现方案"),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def main() -> None:
    ap = argparse.ArgumentParser(description="分类模型选型对比（crack=真实V1 / window=合成窗口质检）")
    ap.add_argument("--task", choices=["crack", "window"], default="crack")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seqs", type=int, default=60, help="window 任务用多少个合成序列")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    report = run_task(args.task, args.cv, args.seed, args.seqs)
    out_dir = MODEL_MULTI if args.task == "crack" else MODEL_ANOMALY
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model_selection.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] task={args.task} 建议模型: {report['selected_model']}  报告: {out}")


if __name__ == "__main__":
    main()
