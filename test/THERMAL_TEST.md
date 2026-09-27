# 温度摄像头 TEST 联调

在 VS Code 运行 **焊接监控: 一键启动整个系统**。此任务只启动本机 TEST 后端、温度 TCP 接收器并打开包含二维热图的原仪表盘；STM32 模拟器和旧采集器不参与运行。后端页面为 `http://127.0.0.1:18080/`；独立热图页仍可在 `/thermal` 查看。

本机接收器设置在 `test/thermal_config.toml`：

- `bind_host`：监听的本机 IPv4。`0.0.0.0` 表示监听所有本机 IPv4 网卡；确定直连网卡后可以改为它的地址。
- `port`：接收工作机温度帧的 TCP 端口，默认 5000。
- `backend_url`：TEST 后端的原始矩阵接口，默认 `http://127.0.0.1:18080/api/data/temperature/raw`。

工作机 B 仍需单独运行 `IRToolProDetection.exe <摄像头IP> <本机直连网卡IP> <接收端口>`。工作机与本机的直连网卡必须同一网段；**不要把摄像头网卡改成传输网卡的地址**。原测试脚本中的 `192.168.2.20` 只是示例，实际使用时应改为本机直连网卡当前地址。

协议沿用 IRToolProDetection 的 TMP1 40 字节帧头和小端 Int16 原始矩阵。TEST 后端只在内存保留最新完整矩阵，页面每半秒读取一次缩小后的温度网格绘制热图，不保存录像或历史矩阵。页面中的“帧号缺口”可提示丢帧；接收端终端另显示后端转发成功和失败次数。

手动运行时，在项目根目录的两个终端分别执行：

```powershell
cd test
..\.venv\Scripts\python.exe -m uvicorn dashboard:app --host 127.0.0.1 --port 18080
```

```powershell
cd test
..\.venv\Scripts\python.exe thermal_receiver.py
```

温度矩阵作为原仪表盘上的独立区块展示；一键任务不启动 STM32 模拟器。
