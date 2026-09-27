import os
from pathlib import Path

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
FRAME_MIN_LEN = 8             # 最小帧长：头(1)+温度(2)+电压(2)+状态(1)+校验(1)+尾(1)

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
# ---------- XGBoost model ----------
USE_XGBOOST = True
# 注：xgb_anomaly 为保留的合成数据窗口模型（32 帧温度/电压/状态窗口）。
# 搭接接头 V1 数据只有工艺参数、没有时间序列，故未用其重训（见 README §7）。
XGB_MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_anomaly" / "model.json"
XGB_WINDOW_SIZE = 32
# ---------- 激光焊接问答（多模态 QA） ----------
QA_ENABLED = True                   # True 后仪表盘 /qa 页面可用
QA_BACKEND = "cloud"                # "cloud" 本机走云端；"local" 工作室本地模型
QA_MODEL = "qwen-vl-max"            # 云端视觉模型（阿里云百炼）；可换 qwen3-vl-plus 等
QA_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 云端服务商 base_url
QA_API_KEY = os.environ.get("QA_API_KEY", "")  # 推荐环境变量；也可直接填密钥
QA_TIMEOUT = 60
# ---------- 工业相机视频（仪表盘内的视频板块） ----------
VIDEO_ENABLED = True
VIDEO_SOURCE_TYPE = "hikrobot_mvs"
VIDEO_RTSP_URL = ""              # 保留旧字段；MVS工业相机不使用RTSP地址
VIDEO_FPS = 15
VIDEO_CAMERA_SERIAL = ""         # 留空时自动选择第一台相机
VIDEO_CAMERA_IP = ""             # 可填写相机固定IP
VIDEO_FRAME_TIMEOUT_MS = 1000
VIDEO_RECONNECT_SECONDS = 3.0
MVS_PYTHON_PATH = r"D:\海康机器人摄像头\MVS\Development\Samples\Python\MvImport"
