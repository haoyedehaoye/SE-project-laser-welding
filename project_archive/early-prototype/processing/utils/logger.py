"""
统一日志管理
使用 loguru，比标准 logging 更简洁强大
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str = "./logs", level: str = "INFO"):
    """初始化全局日志配置"""
    
    # 移除默认 handler
    logger.remove()
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 控制台输出（彩色）
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # 全量日志文件（按天轮转，保留30天）
    logger.add(
        log_path / "all_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
    )
    
    # 错误日志单独存
    logger.add(
        log_path / "error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        encoding="utf-8",
    )
    
    logger.info("日志系统初始化完成")
    return logger


# 创建默认 logger 实例
log = logger