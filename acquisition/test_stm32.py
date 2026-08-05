"""
测试 STM32 采集器（模拟模式）
"""
import sys
import time
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from acquisition.stm32_collector import STM32Collector

# 初始化日志
logger = setup_logger()

# 加载配置
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 使用模拟模式
stm32_config = config["stm32"]
stm32_config["simulated"] = True  # 强制模拟模式

# 创建采集器
collector = STM32Collector(stm32_config)

# 启动
collector.start()
logger.info("STM32 采集器已启动（模拟模式），按 Ctrl+C 停止...")

try:
    while True:
        time.sleep(2)
        latest = collector.get_latest(5)
        if latest:
            logger.info(f"最近5条数据:")
            for dp in latest:
                logger.info(f"  [{dp.timestamp:.3f}] V={dp.data['voltage']:.2f}V, I={dp.data['current']:.2f}A")
        logger.info(f"  健康状态: {collector.is_healthy()}, FPS: {collector.stats['fps']:.1f}")
except KeyboardInterrupt:
    logger.info("用户中断")
finally:
    collector.stop()
    logger.info("测试完成")