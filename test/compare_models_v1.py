# -*- coding: utf-8 -*-
"""
robot_monitor v0.9 — V1 数据集上的多分类模型初步对比（要求 2/3）

在【同一】数据（V1，6 工艺参数 + 横截面位置）与【同一】评估协议
（GroupKFold 按 weld number 分组，5 折，与 train_multiout 一致）下，
对比多种现有分类模型预测裂纹的性能，回答：“换模型/加模型是否比 XGB 更好”。

对比模型（均来自已装依赖 scikit-learn / xgboost）：
    - XGBClassifier（现生产方案）
    - RandomForestClassifier（bagging 基线，class_weight=balanced）
    - ExtraTreesClassifier（更强随机化，class_weight=balanced）
    - HistGradientBoostingClassifier（直方图 GBDT）
    - LogisticRegression（可解释线性基线，折内 StandardScaler，class_weight=balanced）

评估指标（pooled，跨折汇总；同一折切分保证可比）：
    accuracy / balanced_accuracy / F1(0.5) / ROC-AUC / PR-AUC / 按焊缝聚合准确率

用法：
    python compare_models_v1.py
产物：
    models/xgb_multiout/comparison_report.json   （机器可读 + 摘要 markdown 表）
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
                             balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_multiout import DEFAULT_XLSX, load_records, to_matrix

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "xgb_multiout"


def _pooled_metrics(y_all, p_all, g_all) -> dict:
    y_all = np.asarray(y_all)
    p_all = np.asarray(p_all)
    g_all = np.asarray(g_all)
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
    weld_acc, weld_n = [], 0
    for w in sorted(set(g_all.tolist())):
        mask = g_all == w
        if mask.sum() == 0:
            continue
        weld_n += 1
        weld_acc.append(int((p_all[mask].mean() >= 0.5) == (y_all[mask].mean() >= 0.5)))
    m["weld_accuracy"] = round(float(np.mean(weld_acc)), 4) if weld_acc else None
    m["n_welds"] = weld_n
    return m


def make_models(seed: int) -> dict:
    return {
        "XGBoost": lambda tr_x, tr_y: xgb.XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss",
            scale_pos_weight=((tr_y == 0).sum() / max(1, (tr_y == 1).sum())),
            random_state=seed, n_jobs=-1).fit(tr_x, tr_y),
        "RandomForest": lambda tr_x, tr_y: RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1).fit(tr_x, tr_y),
        "ExtraTrees": lambda tr_x, tr_y: ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1).fit(tr_x, tr_y),
        "HistGB": lambda tr_x, tr_y: HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
            random_state=seed).fit(tr_x, tr_y),
        "LogisticReg": lambda tr_x, tr_y: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=seed)).fit(tr_x, tr_y),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V1 裂纹分类多模型初步对比")
    ap.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    data = load_records(args.xlsx)
    records = data["records"]
    X, y, g, input_keys = to_matrix(records, with_pos=True)
    if len(data["welds"]) < args.cv:
        raise SystemExit(f"weld 组数 {len(data['welds'])} < {args.cv}")

    kf = GroupKFold(n_splits=args.cv)
    folds = list(kf.split(X, y, groups=g))
    models = make_models(args.seed)

    results = {}
    for name, fit_fn in models.items():
        t0 = time.time()
        y_all, p_all, g_all = [], [], []
        for tr, te in folds:
            model = fit_fn(X[tr], y[tr])
            p = model.predict_proba(X[te])[:, 1]
            y_all.extend(y[te].tolist())
            p_all.extend(p.tolist())
            g_all.extend(g[te].tolist())
        m = _pooled_metrics(y_all, p_all, g_all)
        m["elapsed_s"] = round(time.time() - t0, 1)
        results[name] = m
        print(f"[{name:<12}] acc={m['accuracy']} bal={m['balanced_accuracy']} "
              f"F1={m['f1']} AUC={m['roc_auc']} PR-AUC={m['pr_auc']} "
              f"weld_acc={m['weld_accuracy']} ({m['elapsed_s']}s)")

    # 输出：JSON + 摘要表
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.xlsx,
        "task": "crack_classification",
        "protocol": f"GroupKFold({args.cv}) by weld number, seed={args.seed}",
        "input_features": input_keys,
        "n_rows": data["n"],
        "n_crack": data["n_crack"],
        "n_welds": len(data["welds"]),
        "models": results,
        "best_by_auc": max(results, key=lambda k: (results[k]["roc_auc"] or 0)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = MODEL_DIR / "comparison_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] 对比报告已保存: {out}")


if __name__ == "__main__":
    main()
