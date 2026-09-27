# -*- coding: utf-8 -*-
"""
robot_monitor v0.9 — 结果一键复盘（任务 6 的执行工具）

聚合当前所有模型报告，输出复盘摘要；--out 时生成 docs/04_实验结果快照.md。
读取：models/xgb_multiout/{eval_report,comparison_report,model_selection}.json
      models/xgb_anomaly/model_selection.json
用法：
    python review_results.py             # 控制台摘要
    python review_results.py --out ../docs/04_实验结果快照.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
MULTI = ROOT / "models" / "xgb_multiout"
ANOMALY = ROOT / "models" / "xgb_anomaly"


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_md() -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# 04 · 实验结果快照（自动生成 {time.strftime('%Y-%m-%d %H:%M')}）\n")

    # 多输出模型（真实 V1）
    eval_r = _load(MULTI / "eval_report.json")
    if eval_r:
        c = eval_r.get("crack_cv", {})
        geo = eval_r.get("geo_cv", {})
        add("## 1. 多输出质量模型（V1 真实数据，GroupKFold 按 weld 分组）")
        if c:
            add(f"- 裂纹分类：acc {c.get('accuracy')} / bal {c.get('balanced_accuracy')} / "
                f"F1 {c.get('f1')} / AUC {c.get('roc_auc')} / PR-AUC {c.get('pr_auc')} / "
                f"焊缝级 {c.get('weld_accuracy')}（{c.get('n_welds')} 道）")
        for key, label in [("steel_w", "钢侧熔宽"), ("copper_w", "铜侧熔宽"),
                           ("copper_d", "铜侧熔深"), ("gap", "间隙")]:
            g = geo.get(key, {})
            if g:
                add(f"- 几何 {label}：MAE {g.get('mae')} / RMSE {g.get('rmse')} / R² {g.get('r2')}")
        add("")

    # 裂纹模型选型
    sel = _load(MULTI / "model_selection.json")
    if sel:
        add("## 2. 裂纹任务模型选型（5 模型同协议）")
        for name, m in sel.get("models", {}).items():
            add(f"- {name}: acc {m.get('accuracy')} / bal {m.get('balanced_accuracy')} / "
                f"F1 {m.get('f1')} / AUC {m.get('roc_auc')} / PR-AUC {m.get('pr_auc')}")
        add(f"- 建议：{sel.get('selected_model')}（差异 < 不确定度，建议沿用 XGBoost）\n")

    # 窗口质检模型选型
    selw = _load(ANOMALY / "model_selection.json")
    if selw:
        add("## 3. 窗口质检任务模型选型（32 帧窗口，合成数据，仅供参考）")
        for name, m in selw.get("models", {}).items():
            add(f"- {name}: acc {m.get('accuracy')} / bal {m.get('balanced_accuracy')} / "
                f"AUC {m.get('roc_auc')} / PR-AUC {m.get('pr_auc')} / 序列级 {m.get('group_accuracy')}")
        add(f"- 建议：{selw.get('selected_model')}（合成口径，待真实时序数据复评）\n")

    add("## 4. 结论")
    add("- 唯一硬约束：V1 仅 5 道焊缝 → 所有指标只作趋势参考（见 docs/01）")
    add("- 模型选择不是当前瓶颈；下一阶段核心 = P0-1 扩大标注焊缝（AUC std < 0.02 验收），P0-2 修 detect_demo 口径")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="写 markdown 快照路径")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    text = build_md()
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\n[ok] 快照已写入: {out}")


if __name__ == "__main__":
    main()
