# ============================================================
# robot_monitor v0.1 — 全局配置
# ============================================================

# ---------- 设备连接 ----------
SWITCH_IP = "127.0.0.1"       # 本地测试用模拟器；硬件到了改这里
SWITCH_PORT = 8234            # 交换机透传端口
SOCKET_TIMEOUT = 0.1          # recv 超时（秒）

# ---------- STM32 帧协议 ----------
FRAME_HEAD  = 0xAA            # 帧头
FRAME_TAIL  = 0x55            # 帧尾
FRAME_MIN_LEN = 8             # 最小帧长：头(1)+温度(2)+湿度(2)+状态(1)+校验(1)+尾(1)

# ---------- 采集器 ----------
BUFFER_SIZE      = 2000       # DataPoint 环形缓冲容量
RECONNECT_INTERVAL = 2.0      # 断线重连间隔（秒）
MAX_ERRORS       = 10         # 连续读错 N 次触发重连
STATS_INTERVAL   = 1.0        # FPS 统计刷新间隔（秒）

# ---------- 日志 ----------
LOG_DIR      = "logs"         # 日志目录
LOG_LEVEL    = "INFO"         # DEBUG / INFO / WARNING / ERROR
LOG_ROTATION = "1 day"
LOG_RETENTION = "7 days"