"""
robot_monitor v0.3 — 采集器入口（带仪表盘）

v0.3 变更：
  - 新增 dashboard 模块，启动 Web 仪表盘
  - collector 数据同时推送到仪表盘
"""

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

from config import (
    LOG_DIR, LOG_LEVEL, LOG_ROTATION, LOG_RETENTION,
    USE_XGBOOST, XGB_MODEL_PATH, XGB_WINDOW_SIZE,
    SWITCH_IP, SWITCH_PORT,
)
from collector import Stm32Collector
from data_types import DataPoint
from quality_checker import RuleBasedQualityChecker
from xgb_checker import XGBoostQualityChecker
from dashboard import start_dashboard, hub                     # ← NEW

logger = logging.getLogger("main")


# ============================================================
# 日志初始化（同 v0.2）
# ============================================================

def _parse_log_rotation(cfg: str) -> tuple[str, int]:
    parts = cfg.strip().split(maxsplit=1)
    if len(parts) != 2:
        return "D", 1
    interval = int(parts[0])
    unit = parts[1].lower().rstrip("s")
    when_map = {"day": "D", "hour": "H", "minute": "M", "second": "S"}
    return when_map.get(unit, "D"), interval


def _parse_retention(cfg: str) -> int:
    parts = cfg.strip().split(maxsplit=1)
    return int(parts[0]) if parts else 7


def setup_logging() -> None:
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    when, interval = _parse_log_rotation(LOG_ROTATION)
    backup_count = _parse_retention(LOG_RETENTION)

    fh = TimedRotatingFileHandler(
        filename=log_dir / "collector.log",
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s"
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

    logger.info(f"日志初始化完成  level={LOG_LEVEL}  dir={log_dir.resolve()}")


# ============================================================
# 回调
# ============================================================

_main_queue: list[DataPoint] = []


def send_to_main(dp: DataPoint) -> None:
    """将质检合格的数据发送给主程序"""
    q = dp.quality

    if q is None:
        _main_queue.append(dp)
        return

    if q.label == "normal":
        _main_queue.append(dp)

    elif q.label == "suspicious":
        logger.warning(
            f"可疑数据仍放行 | flags={q.flags} | temp={dp.data.get('temperature')} "
            f"humi={dp.data.get('humidity')}"
        )
        _main_queue.append(dp)

    elif q.label == "anomaly":
        logger.error(
            f"异常数据已丢弃 | score={q.score:.2f} flags={q.flags} | "
            f"temp={dp.data.get('temperature')} humi={dp.data.get('humidity')} "
            f"details={q.details}"
        )

    while len(_main_queue) > 2000:
        _main_queue.pop(0)


def push_to_dashboard(dp: DataPoint) -> None:                  # ← NEW
    """将 DataPoint 推送给仪表盘（全量，不过滤）"""
    hub.push(dp)


def log_data_point(dp: DataPoint) -> None:
    logging.getLogger("collector").debug(str(dp))


# ============================================================
# 质检统计
# ============================================================

def quality_stats_reporter(checker: RuleBasedQualityChecker, interval: float = 30.0):
    while True:
        time.sleep(interval)
        stats = checker.stats
        logger.info(
            f"[质检统计] 总计:{stats['total']}  "
            f"正常:{stats['normal']}({stats['normal_rate']:.1%})  "
            f"可疑:{stats['suspicious']}  "
            f"异常:{stats['anomaly']}({stats['anomaly_rate']:.1%})"
        )


# ============================================================
# main
# ============================================================

def make_quality_checker():
    """Prefer the XGBoost checker; fall back to rule-based
    quality checking when the model is missing or fails to load."""
    rule_checker = RuleBasedQualityChecker(
        temp_range=(-20.0, 85.0),
        humi_range=(0.0, 100.0),
        max_temp_delta=5.0,
        max_humi_delta=10.0,
        freeze_window=8,
        freeze_temp_tol=0.2,
        freeze_humi_tol=0.5,
        history_size=64,
    )

    if not USE_XGBOOST:
        return rule_checker

    try:
        checker = XGBoostQualityChecker(
            model_path=XGB_MODEL_PATH,
            window_size=XGB_WINDOW_SIZE,
        )
        checker.load()
        logger.info(f"XGBoost quality model loaded: {XGB_MODEL_PATH}")
        return checker
    except Exception as e:
        logger.error(f"XGBoost checker unavailable, fallback to rules: {e}")
        return rule_checker


def main() -> None:
    setup_logging()

    # 1) 质检器
    quality_checker = make_quality_checker()

    # 2) 采集器
    collector = Stm32Collector()
    collector.set_quality_checker(quality_checker)

    # 3) 注册回调
    collector.add_callback(log_data_point)
    collector.add_callback(send_to_main)
    collector.add_callback(push_to_dashboard)          # ← NEW 推送到仪表盘

    # 4) 启动仪表盘 Web 服务                                    # ← NEW
    dashboard_thread = start_dashboard()
    logger.info(f"仪表盘地址: http://localhost:18080")

    # 5) 质检统计线程
    threading.Thread(
        target=quality_stats_reporter,
        args=(quality_checker, 30.0),
        daemon=True,
    ).start()

    # 6) 优雅退出
    def _graceful(sig, frame):
        logger.info(f"收到信号 {sig}，正在停止…")
        collector.stop()

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    logger.info("=" * 50)
    logger.info("robot_monitor v0.3 启动（带仪表盘）")
    logger.info(f"目标 {SWITCH_IP}:{SWITCH_PORT}")
    logger.info("=" * 50)

    collector.run()

    logger.info(f"最终质检统计: {quality_checker.stats}")
    logger.info("robot_monitor 已退出")
    sys.exit(0)


if __name__ == "__main__":
    main()