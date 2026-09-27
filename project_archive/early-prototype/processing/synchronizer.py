"""
多模态时间同步器
将4种不同采样率的数据对齐到统一时间轴
"""
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from utils.logger import log


class MultiModalSynchronizer:
    """
    时间同步器
    
    策略：
    1. 以最高频数据（STM32 100Hz）为基准时间轴
    2. 其他低频数据用最近邻插值对齐
    3. 输出固定长度的时间窗口
    """
    
    def __init__(self, window_size_ms: int = 1000):
        """
        Args:
            window_size_ms: 滑动窗口大小（毫秒）
        """
        self.window_size_ms = window_size_ms
        self.window_size_sec = window_size_ms / 1000.0
        
        # 各模态的独立缓冲区
        self._buffers: Dict[str, deque] = {
            "stm32":   deque(maxlen=5000),
            "thermal": deque(maxlen=500),
            "camera":  deque(maxlen=1000),
            "robot":   deque(maxlen=500),
        }
        
        # 上次窗口输出时间
        self._last_window_time = 0.0
        self._window_interval = 0.2  # 每200ms输出一个窗口
    
    def feed(self, source: str, data_point):
        """喂入一个数据点"""
        if source in self._buffers:
            self._buffers[source].append(data_point)
    
    def get_window(self) -> Optional[Dict]:
        """
        获取当前对齐后的时间窗口数据
        
        Returns:
            {
                "timestamp": ...,
                "voltage": np.array,
                "current": np.array,
                "temp_features": {...},
                "frame": np.array,
                "speed": np.array,
            }
            或 None（数据不足时）
        """
        now = time.time()
        
        # 控制输出频率
        if now - self._last_window_time < self._window_interval:
            return None
        self._last_window_time = now
        
        # 检查各缓冲区是否有足够数据
        min_time = now - self.window_size_sec
        
        stm32_data = [dp for dp in self._buffers["stm32"] 
                      if dp.timestamp >= min_time]
        thermal_data = [dp for dp in self._buffers["thermal"] 
                        if dp.timestamp >= min_time]
        camera_data = [dp for dp in self._buffers["camera"] 
                       if dp.timestamp >= min_time]
        robot_data = [dp for dp in self._buffers["robot"] 
                      if dp.timestamp >= min_time]
        
        # 至少需要STM32数据
        if len(stm32_data) < 10:
            return None
        
        # 提取电压电流序列
        voltage = np.array([dp.data["voltage"] for dp in stm32_data])
        current = np.array([dp.data["current"] for dp in stm32_data])
        
        # 温度特征（取最新一个）
        temp_features = {}
        if thermal_data:
            latest = thermal_data[-1].data
            temp_features = {k: v for k, v in latest.items() 
                           if k != "temp_matrix"}
        
        # 视频帧（取最新一个）
        frame = None
        if camera_data:
            frame = camera_data[-1].data.get("frame")
        
        # 焊接速度序列
        speed = np.array([dp.data["speed"] for dp in robot_data]) if robot_data else np.array([])
        
        return {
            "timestamp": now,
            "voltage": voltage,
            "current": current,
            "temp_features": temp_features,
            "frame": frame,
            "speed": speed,
            "counts": {
                "stm32": len(stm32_data),
                "thermal": len(thermal_data),
                "camera": len(camera_data),
                "robot": len(robot_data),
            },
        }