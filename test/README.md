# TEST 实验区

此目录用于持续开发、设备联调和模型实验。其接口与模型结论不会自动进入正式 `app/`。

| 方向 | 主要文件 | 当前用途 |
| --- | --- | --- |
| STM32 旧协议 | `simulator.py`、`collector.py`、`main.py`、`dashboard.py` | TCP 模拟、旧版数据质检和仪表盘；协议字段与目标电流/电压不同 |
| 热像仪 | `thermal_receiver.py`、`thermal_frames.py`、`thermal_config.toml`、`templates/thermal.html`、`test_thermal.py` | TCP 接收、原始矩阵解码、最新帧和热图；运行方式见 [`THERMAL_TEST.md`](THERMAL_TEST.md) |
| 工业相机 | `camera_service.py`、`templates/index.html` | 海康相机联调和预览 |
| XGBoost 研究 | `train_*.py`、`evaluate_*.py`、`predict_*.py`、`compare_models*.py` | 窗口异常、裂纹和焊缝几何研究，模型文件在项目根 `models/` |
| 问答和演示 | `qa_*.py`、`model_demo.py`、`detect_demo.py` | 可选研究功能 |

VS Code 任务“焊接监控: 一键启动整个系统”目前启动的是**TEST 温度联调**，不是正式 APP，也不启动 STM32 模拟器。正式系统使用项目根目录的 `scripts/run.ps1`，或 VS Code 任务“正式系统: 一键启动并打开相机监控”。

旧日志已归档到 `../project_archive/logs/test/`，旧 Python 缓存已归档到 `../project_archive/python-cache/`；运行 TEST 时会重新生成这些目录。`requirements.txt` 是实验依赖；根目录的 `requirements.txt` 是正式 APP 依赖。
