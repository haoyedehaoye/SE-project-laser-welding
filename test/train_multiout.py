# -*- coding: utf-8 -*-
"""
robot_monitor v0.9 — 钢-铜搭接接头“工艺 → 裂纹 + 几何”多输出质量模型

数据：Definitive screening steel-copper lap joints V1.xlsx（360 行真实数据）
输入（工艺参数，可选第 7 个 = 焊缝横截面位置 mm）：
    power, speed, gas_flow, focal, angular, thickness[, cross_section_mm]
输出：
    1) 裂纹二分类   crack（yes/no）            → XGBClassifier
    2) 几何量回归   钢侧熔宽/铜侧熔宽/铜侧熔深/gap → XGBRegressor × 4
评估口径（真实、防泄漏）：
    - 一律 GroupKFold 按 weld number 分组（同一条焊缝的样本永不跨折）
    - 裂纹：accuracy/balanced/F1/ROC-AUC/PR-AUC + 按焊缝聚合准确率
    - 几何：每目标 pooled MAE/RMSE/R²

用法：
    python train_multiout.py                        # 输出 models/xgb_multiout
    python train_multiout.py --no-position          # 只用 6 工艺参数
    python train_multiout.py --cv 5 --seed 42
产物（models/xgb_multiout/）：
    input_features.json   输入特征键（顺序）
    crack/model.json      裂纹分类器（XGBClassifier）
    geo/<target>/model.json  4 个几何回归器
    config.json / eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from openpyxl import load_workbook
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_XLSX = (
    Path(r"D:\SEproject\dataset\online\对激光焊接的钢-铜搭接接头的数据集")
    / "Definitive screening steel-copper lap joints V1.xlsx"
)

# 规范输入键
PARAM_KEYS = ["power", "speed", "gas_flow", "focal", "angular", "thickness"]
POS_KEY = "cross_section_mm"
# 几何输出键（V1 里可测的 4 个连续量）
GEO_TARGETS = {
    "steel_w":   ("weld seam width steel",   "钢侧熔宽(µm)"),
    "copper_w":  ("weld seam width copper",  "铜侧熔宽(µm)"),
    "copper_d":  ("weld depth copper",       "铜侧熔深(µm)"),
    "gap":       ("gap",                     "间隙(µm)"),
}
# 列名关键字 → 角色
ROLE_KEYWORDS = [
    ("power", "power"), ("speed", "speed"), ("gas flow", "gas_flow"),
    ("focal", "focal"), ("angular", "angular"), ("thickness", "thickness"),
    ("cracking", "label"), ("weld number", "group"),
    ("cross section", "pos"), ("steel", "g_steel_w"),
    ("weld seam width copper", "g_copper_w"), ("weld depth copper", "g_copper_d"),
    ("gap", "g_gap"),
]


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def load_records(xlsx_path: str) -> dict:
    """读 V1 → records 列表 + 可用性统计。"""
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = rows[0]

    col_role: dict[int, str] = {}
    for j, cell in enumerate(header):
        if cell is None:
            continue
        key = str(cell).strip().lower()
        for kw, role in ROLE_KEYWORDS:
            if kw in key and j not in col_role:
                col_role[j] = role
                break

    required = {"power", "speed", "gas_flow", "focal", "angular", "thickness",
                "label", "group"}
    missing = required - set(col_role.values())
    if missing:
        raise SystemExit(f"V1 缺少必要列: {missing}")

    records = []
    for row in rows[1:]:
        rec = {"params": {}, "pos": None, "label": None, "group": None, "geo": {}}
        ok = True
        for j, role in col_role.items():
            v = row[j]
            if role in ("power", "speed", "gas_flow", "focal", "angular", "thickness"):
                f = _to_float(v)
                if f is None:
                    ok = False
                    break
                rec["params"][role] = f
            elif role == "pos":
                rec["pos"] = _to_float(v)
            elif role == "label":
                text = str(v).strip().lower() if v is not None else ""
                if text in ("yes", "y", "1"):
                    rec["label"] = 1
                elif text in ("no", "n", "0"):
                    rec["label"] = 0
                else:
                    ok = False
            elif role == "group":
                g = _to_float(v)
                rec["group"] = int(g) if g is not None else None
            elif role in ("g_steel_w", "g_copper_w", "g_copper_d", "g_gap"):
                rec["geo"][role[2:]] = _to_float(v)   # g_steel_w → steel_w（与 GEO_TARGETS 键一致）
        if ok and rec["label"] is not None and len(rec["params"]) == len(PARAM_KEYS):
            records.append(rec)

    if not records:
        raise SystemExit(f"V1 没有可用记录: {xlsx_path}")

    n_pos = sum(r["label"] for r in records)
    n_geo_complete = sum(1 for r in records
                         if all(r["geo"].get(t) is not None for t in GEO_TARGETS))
    n_pos_missing = sum(1 for r in records if r["pos"] is None)
    return {
        "records": records,
        "n": len(records),
        "pos_rate": n_pos / len(records),
        "n_crack": n_pos,
        "n_geo_complete": n_geo_complete,
        "n_pos_missing": n_pos_missing,
        "welds": sorted({r["group"] for r in records}),
    }


def to_matrix(records: list[dict], with_pos: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """records → (X, y(label), groups, input_keys)。"""
    input_keys = list(PARAM_KEYS) + ([POS_KEY] if with_pos else [])
    X, y, g = [], [], []
    for r in records:
        row = [r["params"][k] for k in PARAM_KEYS]
        if with_pos:
            row.append(r["pos"] if r["pos"] is not None else float("nan"))
        X.append(row)
        y.append(r["label"])
        g.append(r["group"])
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32),
            np.asarray(g, dtype=np.int32), input_keys)


def geo_matrix(records: list[dict], target_key: str, with_pos: bool):
    """取某一几何目标的可用行。"""
    input_keys = list(PARAM_KEYS) + ([POS_KEY] if with_pos else [])
    X, y, g = [], [], []
    for r in records:
        val = r["geo"].get(target_key)
        if val is None:
            continue
        row = [r["params"][k] for k in PARAM_KEYS]
        if with_pos:
            row.append(r["pos"] if r["pos"] is not None else float("nan"))
        X.append(row)
        y.append(val)
        g.append(r["group"])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(g, dtype=np.int32)


# ---------------------------------------------------------------------------
# CV 评估（按 weld 分组）
# ---------------------------------------------------------------------------

def crack_group_cv(X, y, groups, cv: int, seed: int) -> dict:
    """GroupKFold 逐折训练，汇总 pooled 指标 + 按焊缝聚合准确率。"""
    kf = GroupKFold(n_splits=cv)
    y_all, p_all, g_all = [], [], []
    fold_auc = []
    for tr, te in kf.split(X, y, groups=groups):
        neg = int((y[tr] == 0).sum())
        pos = int((y[tr] == 1).sum())
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            scale_pos_weight=(neg / pos if pos > 0 else 1.0),
            random_state=seed, n_jobs=-1)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        y_all.extend(y[te].tolist())
        p_all.extend(p.tolist())
        g_all.extend(groups[te].tolist())
        if len(np.unique(y[te])) == 2:
            fold_auc.append(roc_auc_score(y[te], p))
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
        "roc_auc": round(float(np.mean(fold_auc)) if fold_auc else None, 4),
        "pr_auc": round(float(average_precision_score(y_all, p_all)), 4),
        "confusion_matrix": confusion_matrix(y_all, pred).tolist(),
    }
    # 按焊缝聚合：同一条焊缝的平均预测概率 vs 平均标签
    weld_acc = []
    for w in sorted(set(g_all.tolist())):
        mask = g_all == w
        if mask.sum() == 0:
            continue
        weld_acc.append(int((p_all[mask].mean() >= 0.5) == (y_all[mask].mean() >= 0.5)))
    m["weld_accuracy"] = round(float(np.mean(weld_acc)), 4) if weld_acc else None
    m["n_welds"] = len(weld_acc)
    return m, (y_all, p_all, g_all)


def geo_group_cv(X, y, groups, cv: int, seed: int) -> dict:
    kf = GroupKFold(n_splits=cv)
    y_all, p_all = [], []
    for tr, te in kf.split(X, y, groups=groups):
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=-1)
        model.fit(X[tr], y[tr])
        y_all.extend(y[te].tolist())
        p_all.extend(model.predict(X[te]).tolist())
    y_all = np.asarray(y_all)
    p_all = np.asarray(p_all)
    return {
        "n": int(len(y_all)),
        "mae": round(float(mean_absolute_error(y_all, p_all)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_all, p_all))), 3),
        "r2": round(float(r2_score(y_all, p_all)), 4),
        "y_mean": round(float(y_all.mean()), 2),
        "y_std": round(float(y_all.std()), 2),
    }


def _input_ranges(records: list[dict], with_pos: bool) -> dict:
    keys = list(PARAM_KEYS) + ([POS_KEY] if with_pos else [])
    out = {}
    for k in PARAM_KEYS:
        vals = [r["params"][k] for r in records]
        out[k] = [round(min(vals), 4), round(max(vals), 4)]
    if with_pos:
        vals = [r["pos"] for r in records if r["pos"] is not None]
        out[POS_KEY] = [round(min(vals), 4), round(max(vals), 4)]
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="V1 工艺→裂纹+几何 多输出质量模型")
    ap.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-position", action="store_true",
                    help="只用 6 个工艺参数，不加横截面位置特征")
    args = ap.parse_args()

    data = load_records(args.xlsx)
    with_pos = not args.no_position
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"[data] {data['n']} 行 / 裂纹 {data['n_crack']} ({data['pos_rate']:.1%}) / "
          f"weld 组数 {len(data['welds'])} / 几何完整行 {data['n_geo_complete']}")
    if with_pos:
        print(f"[data] 使用横截面位置特征；位置缺失行 {data['n_pos_missing']}")

    records = data["records"]
    X, y, g, input_keys = to_matrix(records, with_pos)
    if len(data["welds"]) < args.cv:
        raise SystemExit(f"weld 组数 {len(data['welds'])} < CV 折数 {args.cv}，请调小 --cv")
    print(f"[warn] 全数据集仅 {len(data['welds'])} 道焊缝：按 weld 分组的折数受限于此，"
          f"AUC/准确率估计方差较大，结论仅作趋势参考")

    # ---- 裂纹分类 CV ----
    print("\n========== 裂纹分类（GroupKFold 按 weld 分组） ==========")
    crack_cv, _ = crack_group_cv(X, y, g, args.cv, args.seed)
    print(f"n={crack_cv['n']} pos_rate={crack_cv['pos_rate']:.1%} "
          f"acc={crack_cv['accuracy']} bal_acc={crack_cv['balanced_accuracy']} "
          f"F1={crack_cv['f1']} AUC={crack_cv['roc_auc']} PR-AUC={crack_cv['pr_auc']}")
    print(f"按焊缝聚合准确率={crack_cv['weld_accuracy']}（{crack_cv['n_welds']} 条焊缝）")

    # ---- 几何回归 CV ----
    print("\n========== 几何量回归（GroupKFold 按 weld 分组） ==========")
    geo_cv = {}
    for tkey, (_, label) in GEO_TARGETS.items():
        Xg, yg, gg = geo_matrix(records, tkey, with_pos)
        mm = geo_group_cv(Xg, yg, gg, args.cv, args.seed)
        geo_cv[tkey] = mm
        print(f"{label:<12} n={mm['n']:>3} y={mm['y_mean']:.0f}±{mm['y_std']:.0f}  "
              f"MAE={mm['mae']} RMSE={mm['rmse']} R²={mm['r2']}")

    # ---- 保存（用全量数据重训的 serving 模型） ----
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "models" / "xgb_multiout")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input_features.json").write_text(
        json.dumps(input_keys, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "input_ranges.json").write_text(
        json.dumps(_input_ranges(records, with_pos), ensure_ascii=False, indent=2),
        encoding="utf-8")

    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    crack_final = xgb.XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss",
        scale_pos_weight=(neg / pos if pos > 0 else 1.0),
        random_state=args.seed, n_jobs=-1)
    crack_final.fit(X, y)
    (out_dir / "crack").mkdir(parents=True, exist_ok=True)
    crack_final.save_model(str(out_dir / "crack" / "model.json"))

    geo_models = {}
    for tkey, (_, label) in GEO_TARGETS.items():
        Xg, yg, _ = geo_matrix(records, tkey, with_pos)
        m = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                             random_state=args.seed, n_jobs=-1)
        m.fit(Xg, yg)
        d = out_dir / "geo" / tkey
        d.mkdir(parents=True, exist_ok=True)
        m.save_model(str(d / "model.json"))
        geo_models[tkey] = {"label": label, "n_train": int(len(yg))}

    report = {
        "dataset": args.xlsx,
        "input_features": input_keys,
        "with_position": with_pos,
        "n_rows": data["n"],
        "n_crack": data["n_crack"],
        "cv_folds": args.cv,
        "cv_note": "GroupKFold 按 weld number 分组；裂纹为 pooled 行级指标 + 按焊缝聚合准确率",
        "crack_cv": crack_cv,
        "geo_cv": geo_cv,
        "geo_models": geo_models,
        "input_ranges": _input_ranges(records, with_pos),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"model_type": "xgboost multi-output",
                    "note": "裂纹=XGBClassifier；几何=4×XGBRegressor；输入见 input_features.json",
                    "threshold": 0.5}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] serving 模型与评估报告已保存到: {out_dir}")


if __name__ == "__main__":
    main()
