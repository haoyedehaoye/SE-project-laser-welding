"""
采集器抽象基类
所有数据采集器（STM32、温度相机、工业相机、机械臂）都继承此类
"""
import time
import threading
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from utils.logger import log


@dataclass
class DataPoint:
    """统一的数据点格式"""
    timestamp: float           # Unix时间戳（秒）
    source: str                # 数据来源: "stm32" / "thermal" / "camera" / "robot"
    data: Any                  # 实际数据
    metadata: dict = field(default_factory=dict)


class BaseCollector(ABC):
    """
    采集器抽象基类
    
    子类必须实现:
        _connect()       - 建立连接
        _disconnect()    - 断开连接
        _read_once()     - 单次读取，返回 DataPoint 或 None
    """
    
    def __init__(self, source_name: str, buffer_size: int = 2000):
        self.source_name = source_name
        self.buffer = deque(maxlen=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._error_count = 0
        self._max_errors = 10       # 连续错误超过此数触发重连
        self._lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            "total_reads": 0,
            "total_errors": 0,
            "last_read_time": 0,
            "fps": 0.0,
        }
    
    # ========== 子类必须实现 ==========
    
    @abstractmethod
    def _connect(self) -> bool:
        """建立与设备的连接，返回 True/False"""
        ...
    
    @abstractmethod
    def _disconnect(self):
        """断开连接，释放资源"""
        ...
    
    @abstractmethod
    def _read_once(self) -> Optional[DataPoint]:
        """单次读取，返回 DataPoint 或 None（读取失败时）"""
        ...
    
    # ========== 公共接口 ==========
    
    def start(self):
        """启动后台采集线程"""
        if self._running:
            log.warning(f"[{self.source_name}] 采集器已在运行")
            return
        
        log.info(f"[{self.source_name}] 正在启动采集器...")
        
        # 先尝试连接
        if not self._connect():
            log.error(f"[{self.source_name}] 初始连接失败，将在后台重试")
        
        self._running = True
        self._thread = threading.Thread(
            target=self._collect_loop,
            name=f"collector-{self.source_name}",
            daemon=True,
        )
        self._thread.start()
        log.info(f"[{self.source_name}] 采集器已启动")
    
    def stop(self):
        """停止采集"""
        log.info(f"[{self.source_name}] 正在停止采集器...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._disconnect()
        log.info(f"[{self.source_name}] 采集器已停止")
    
    def get_latest(self, n: int = 1) -> list:
        """获取最近 n 条数据（线程安全）"""
        with self._lock:
            items = list(self.buffer)[-n:] if n > 0 else list(self.buffer)
        return items
    
    def get_all(self) -> list:
        """获取缓冲区全部数据"""
        return self.get_latest(0)
    
    def is_healthy(self) -> bool:
        """健康检查"""
        return self._connected and self._error_count < self._max_errors
    
    # ========== 内部方法 ==========
    
    def _collect_loop(self):
        """采集主循环（在后台线程中运行）"""
        last_stats_time = time.time()
        read_count = 0
        
        while self._running:
            try:
                # 未连接则尝试重连
                if not self._connected:
                    log.info(f"[{self.source_name}] 尝试重新连接...")
                    self._connected = self._connect()
                    if not self._connected:
                        time.sleep(2.0)  # 重连间隔
                        continue
                    else:
                        self._error_count = 0
                
                # 读取数据
                dp = self._read_once()
                
                if dp is not None:
                    with self._lock:
                        self.buffer.append(dp)
                    self._error_count = 0
                    read_count += 1
                    self.stats["total_reads"] += 1
                    self.stats["last_read_time"] = dp.timestamp
                else:
                    self._error_count += 1
                    self.stats["total_errors"] += 1
                
                # 错误过多 → 断连重连
                if self._error_count >= self._max_errors:
                    log.error(f"[{self.source_name}] 连续 {self._max_errors} 次读取失败，断开重连")
                    self._disconnect()
                    self._connected = False
                    self._error_count = 0
                
                # 每秒更新 FPS 统计
                now = time.time()
                if now - last_stats_time >= 1.0:
                    self.stats["fps"] = read_count / (now - last_stats_time)
                    read_count = 0
                    last_stats_time = now
                    
            except Exception as e:
                log.error(f"[{self.source_name}] 采集异常: {e}")
                self._error_count += 1
                time.sleep(0.1)
    
    def _add_to_buffer(self, dp: DataPoint):
        """线程安全地添加数据"""
        with self._lock:
            self.buffer.append(dp)