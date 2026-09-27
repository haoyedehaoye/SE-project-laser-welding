"""
STM32 电压电流采集器
通过串口读取 STM32 上传的电压电流数据
支持：真实串口模式 / 模拟数据模式（无硬件时测试用）
"""
import time
import struct
import random
import math
from typing import Optional
import numpy as np

try:
    import serial
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False

from .base_collector import BaseCollector, DataPoint
from utils.logger import log


class STM32Collector(BaseCollector):
    """
    STM32 数据采集器
    
    通信协议（示例，需与你的STM32固件一致）：
        帧格式: 0xAA | voltage(float32,4B) | current(float32,4B) | checksum(1B) | 0x55
        总长度: 1 + 4 + 4 + 1 + 1 = 11 字节
        
    voltage: 焊接电压 (V)，范围 10-40V
    current: 焊接电流 (A)，范围 50-400A
    """
    
    # 帧格式常量
    FRAME_HEADER = 0xAA
    FRAME_FOOTER = 0x55
    FRAME_LENGTH = 11
    
    def __init__(self, config: dict):
        super().__init__(source_name="stm32", buffer_size=2000)
        self.config = config
        
        # 串口参数
        self.port = config.get("port", "COM3")
        self.baudrate = config.get("baudrate", 115200)
        self.timeout = config.get("timeout", 0.1)
        self._serial: Optional[serial.Serial] = None
        
        # 模拟模式（无硬件时）
        self._simulated = config.get("simulated", False)
        self._sim_voltage = 25.0
        self._sim_current = 200.0
        self._sim_time = 0.0
        
        # 接收缓冲区（处理粘包）
        self._rx_buffer = bytearray()
        
        # 时间基准
        self._start_time = time.time()
    
    def _connect(self) -> bool:
        """连接 STM32 串口"""
        if self._simulated:
            log.info(f"[{self.source_name}] 模拟模式，跳过真实串口连接")
            return True
        
        if not HAS_PYSERIAL:
            log.error("[{self.source_name}] pyserial 未安装，请执行: pip install pyserial")
            return False
        
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
            )
            if self._serial.is_open:
                log.info(f"[{self.source_name}] 串口 {self.port} 连接成功 (baud={self.baudrate})")
                self._rx_buffer.clear()
                return True
        except serial.SerialException as e:
            log.error(f"[{self.source_name}] 串口连接失败: {e}")
        
        return False
    
    def _disconnect(self):
        """断开串口连接"""
        if self._serial and self._serial.is_open:
            self._serial.close()
            log.info(f"[{self.source_name}] 串口已断开")
    
    def _read_once(self) -> Optional[DataPoint]:
        """读取一帧数据"""
        if self._simulated:
            return self._read_simulated()
        else:
            return self._read_serial()
    
    def _read_simulated(self) -> DataPoint:
        """
        生成模拟焊接数据
        
        模拟场景：
        - 正常焊接：电压~25V，电流~200A，小幅波动
        - 每30秒随机出现一次"微异常"：电压或电流波动增大
        - 每5分钟出现一次较明显的异常
        """
        now = time.time()
        self._sim_time += 0.01  # 模拟100Hz
        t = self._sim_time
        
        # 基础值 + 正常噪声
        voltage = 25.0 + random.gauss(0, 0.3)
        current = 200.0 + random.gauss(0, 2.0)
        
        # 周期性波动（模拟焊接过程中的自然变化）
        voltage += 1.5 * math.sin(t * 0.5)
        current += 10.0 * math.sin(t * 0.5 + 1.0)
        
        # 偶发异常：30秒一次微异常
        if int(t) % 30 == 0:
            voltage += random.gauss(0, 1.5)   # 波动增大5倍
            current += random.gauss(0, 10.0)
        
        # 偶发严重异常：300秒(5分钟)一次
        if int(t) % 300 == 0 and int(t) % 300 < 1:
            voltage += random.uniform(-5, 5)
            current += random.uniform(-40, 40)
        
        dp = DataPoint(
            timestamp=now,
            source=self.source_name,
            data={
                "voltage": round(voltage, 2),
                "current": round(current, 2),
            },
            metadata={"mode": "simulated"},
        )
        return dp
    
    def _read_serial(self) -> Optional[DataPoint]:
        """从串口读取并解析一帧"""
        if not self._serial or not self._serial.is_open:
            return None
        
        try:
            # 读取可用字节
            waiting = self._serial.in_waiting
            if waiting > 0:
                data = self._serial.read(waiting)
                self._rx_buffer.extend(data)
            
            # 在缓冲区中查找完整帧
            return self._parse_frame()
            
        except serial.SerialException as e:
            log.error(f"[{self.source_name}] 串口读取错误: {e}")
            return None
    
    def _parse_frame(self) -> Optional[DataPoint]:
        """从接收缓冲区解析一个完整帧"""
        while len(self._rx_buffer) >= self.FRAME_LENGTH:
            # 查找帧头
            if self._rx_buffer[0] != self.FRAME_HEADER:
                # 跳过无效字节直到找到帧头
                try:
                    header_idx = self._rx_buffer.index(self.FRAME_HEADER)
                    self._rx_buffer = self._rx_buffer[header_idx:]
                except ValueError:
                    self._rx_buffer.clear()
                    return None
            
            if len(self._rx_buffer) < self.FRAME_LENGTH:
                return None
            
            # 检查帧尾
            if self._rx_buffer[9] != self.FRAME_FOOTER:
                # 帧尾不对，丢掉第一个字节继续找
                self._rx_buffer.pop(0)
                continue
            
            # 提取字段
            raw = self._rx_buffer[:self.FRAME_LENGTH]
            
            # 校验和（前9字节的异或）
            checksum = raw[8]
            calc_checksum = 0
            for b in raw[:8]:
                calc_checksum ^= b
            
            if checksum != calc_checksum:
                log.warning(f"[{self.source_name}] 校验和不匹配")
                self._rx_buffer.pop(0)
                continue
            
            # 解析电压电流（小端序 float32）
            voltage = struct.unpack('<f', bytes(raw[1:5]))[0]
            current = struct.unpack('<f', bytes(raw[5:9]))[0]
            
            # 从缓冲区移除已解析的帧
            self._rx_buffer = self._rx_buffer[self.FRAME_LENGTH:]
            
            return DataPoint(
                timestamp=time.time(),
                source=self.source_name,
                data={"voltage": round(voltage, 2), "current": round(current, 2)},
            )
    