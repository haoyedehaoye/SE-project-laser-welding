# -*- coding: utf-8 -*-
"""
robot_monitor v0.9 — 多输出质量模型评估（读取/复核 models/xgb_multiout 报告）

train_multiout.py 的评估发生在训练期（GroupKFold 按 weld 分组，诚实口径），
本脚本负责：
  1. 打印 eval_report.json 的裂纹/几何 CV 汇总；
  2. --recheck 时重跑一次同口径 CV（与训练交叉验证一致性）；
  3. --weld-detail 输出按焊缝聚合的裂纹明细（正确/错误）。

用法：
    python evaluate_multiout.py
    python evaluate_multiout.py --recheck --weld-detail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_multiout import (DEFAULT_XLSX, GEO_TARGETS, crack_group_cv,
                            geo_group_cv, load_records, to_matrix)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "xgb_multiout"


def _print_report(report: dict) -> None:
    print(f"数据集: {report['dataset']}")
    print(f"输入特征: {report['input_features']}  (横截面位置: {'含' if report['with_position'] else '不含'})")
    print(f"CV: {report['cv_folds']} 折按 weld 分组")
    c = report["crack_cv"]
    print("\n========== 裂纹分类（行级 pooled） ==========")
    print(f"n={c['n']} pos_rate={c['pos_rate']:.1%} acc={c['accuracy']} "
          f"bal_acc={c['balanced_accuracy']} F1={c['f1']} AUC={c['roc_auc']} "
          f"PR-AUC={c['pr_auc']} 按焊缝准确率={c['weld_accuracy']} "
          f"(焊缝 {c['n_welds']})")
    print("混淆矩阵 (TN FP / FN TP):")
    print(np.asarray(c["confusion_matrix"]))
    print("\n========== 几何回归 ==========")
    for key, (_, label) in GEO_TARGETS.items():
        g = report["geo_cv"][key]
        print(f"{label:<12} n={g['n']:>3} MAE={g['mae']} RMSE={g['rmse']} R²={g['r2']}")


def recheck(cv: int, seed: int, with_pos: bool, weld_detail: bool) -> None:
    data = load_records(str(DEFAULT_XLSX))
    records = data["records"]
    X, y, g, _ = to_matrix(records, with_pos)
    c, (y_all, p_all, g_all) = crack_group_cv(X, y, g, cv, seed)
    print("\n--- recheck 裂纹 CV ---")
    print(f"acc={c['accuracy']} bal_acc={c['balanced_accuracy']} F1={c['f1']} "
          f"AUC={c['roc_auc']} PR-AUC={c['pr_auc']} 按焊缝={c['weld_accuracy']}")
    if weld_detail:
        print("\n按焊缝明细（weld | 行数 | 平均标签 | 平均预测 | 判定）:")
        for w in sorted(set(g_all.tolist())):
            mask = g_all == w
            if mask.sum() == 0:
                continue
            mean_y = y_all[mask].mean()
            mean_p = p_all[mask].mean()
            verdict = "OK" if (mean_p >= 0.5) == (mean_y >= 0.5) else "X"
            print(f"  weld {w:>3} | {int(mask.sum()):>2} | {mean_y:.2f} | "
                  f"{mean_p:.3f} | {verdict}")
    for key, (_, label) in GEO_TARGETS.items():
        from train_multiout import geo_matrix
        Xg, yg, gg = geo_matrix(records, key, with_pos)
        mm = geo_group_cv(Xg, yg, gg, cv, seed)
        print(f"recheck 几何 {label:<10} MAE={mm['mae']} RMSE={mm['rmse']} R²={mm['r2']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--recheck", action="store_true")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-position", action="store_true")
    ap.add_argument("--weld-detail", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    report_path = Path(args.model_dir) / "eval_report.json"
    if not report_path.exists():
        raise SystemExit(f"缺少 {report_path}，先运行 train_multiout.py")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _print_report(report)
    if args.recheck:
        recheck(args.cv, args.seed, not args.no_position, args.weld_detail)


if __name__ == "__main__":
    main()
