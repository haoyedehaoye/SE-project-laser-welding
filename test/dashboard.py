"""
robot_monitor v0.3 — 实时仪表盘

基于 FastAPI + WebSocket + ECharts。
独立线程运行，不阻塞采集主循环。

数据流：
    Collector 回调 → Dashboard.push() → 缓冲队列
                                     → 批量推送线程 → WebSocket 广播
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from data_types import DataPoint
from thermal_frames import ThermalFrameStore
from config import (
    QA_API_BASE, QA_API_KEY, QA_BACKEND, QA_ENABLED, QA_MODEL, QA_TIMEOUT,
    MVS_PYTHON_PATH, VIDEO_CAMERA_IP, VIDEO_CAMERA_SERIAL, VIDEO_ENABLED,
    VIDEO_FPS, VIDEO_FRAME_TIMEOUT_MS, VIDEO_RECONNECT_SECONDS,
    VIDEO_RTSP_URL, VIDEO_SOURCE_TYPE,
)
from camera_service import HikCameraService
from qa_service import LaserWeldingQA

logger = logging.getLogger("dashboard")
thermal_frames = ThermalFrameStore()

# ============================================================
# 配置
# ============================================================
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 18080
MAX_PENDING = 500          # 待广播缓冲上限（防止无客户端时无限增长）
CLIENT_QUEUE_SIZE = 64     # 每个客户端发送队列上限（慢客户端丢旧保新）
BROADCAST_INTERVAL = 0.1   # 广播间隔（秒）


# ============================================================
# 数据缓冲 + 广播管理
# ============================================================

class DashboardHub:
    """
    WebSocket 广播中心。

    数据流（线程模型）：
      - push(): 采集器线程调用，把 payload 放入有界待广播缓冲 _pending。
      - _broadcast_loop(): 仪表盘事件循环内的【单一】分发任务，周期性取出
        _pending 并投递给【每一个】在线客户端的有界发送队列（Fan-out）。
        旧的“每个 WebSocket 连接各自 drain 全局 _pending”实现会让多个客户端
        互相抢数据——先 drain 的客户端独占整批，其他客户端只收到约 1/N 数据。
      - 每个客户端有独立的发送循环，从自己的队列取消息发送；慢客户端只会
        影响自己（队列满时丢旧保新），不会阻塞其他客户端。
      - 无客户端时 _pending 由分发任务周期性清空且 deque 本身有上限，
        不会无限增长。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[WebSocket, asyncio.Queue[dict]] = {}  # ws → 发送队列
        self._pending: deque[dict] = deque(maxlen=MAX_PENDING)    # 有界待广播缓冲
        self._stats = {"total_pushed": 0, "normal": 0, "suspicious": 0, "anomaly": 0}
        self._start_time = time.time()

    # ---- 采集器线程入口 ----

    def push(self, dp: DataPoint) -> None:
        """采集器回调入口（来自 collector 线程）"""
        q = dp.quality
        payload = {
            "timestamp": dp.timestamp,
            "source": dp.source,
            "data": dp.data,
            "quality": q.to_dict() if q else None,
        }

        with self._lock:
            self._pending.append(payload)  # deque(maxlen) 超限自动丢最旧，内存有界

            # 统计
            self._stats["total_pushed"] += 1
            if q:
                self._stats[q.label] = self._stats.get(q.label, 0) + 1

    # ---- 连接管理（事件循环线程） ----

    def register(self, ws: WebSocket, outbox: asyncio.Queue[dict]) -> None:
        with self._lock:
            self._clients[ws] = outbox

    def unregister(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.pop(ws, None)

    # ---- 分发（仅在仪表盘事件循环内调用，不持锁等待） ----

    def snapshot_batch(self) -> list[dict]:
        """取走当前全部待广播 payload"""
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()
        return batch

    def fanout(self, batch: list[dict]) -> None:
        """把同一批数据投递给所有客户端（广播）"""
        if not batch:
            return
        message = {"type": "data", "data": batch}
        with self._lock:
            outboxes = list(self._clients.values())
        for outbox in outboxes:
            self._put_latest(outbox, message)

    def send_to(self, ws: WebSocket, message: dict) -> None:
        """向单个客户端投递消息（如 stats 回复）"""
        with self._lock:
            outbox = self._clients.get(ws)
        if outbox is not None:
            self._put_latest(outbox, message)

    @staticmethod
    def _put_latest(outbox: asyncio.Queue[dict], message: dict) -> None:
        """队列满时先丢最旧消息再放入，保证每个客户端队列有界。"""
        if outbox.full():
            try:
                outbox.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            outbox.put_nowait(message)
        except asyncio.QueueFull:
            pass

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def stats(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time
            return {**self._stats, "uptime": round(uptime, 1), "clients": len(self._clients)}


async def _broadcast_loop() -> None:
    """单一分发任务：周期性取出待广播数据并投递给所有在线客户端。"""
    logger.info("仪表盘广播任务已启动")
    while True:
        await asyncio.sleep(BROADCAST_INTERVAL)
        batch = hub.snapshot_batch()
        if batch:
            hub.fanout(batch)


camera_service = HikCameraService(
    mvs_python_path=MVS_PYTHON_PATH,
    serial_number=VIDEO_CAMERA_SERIAL,
    ip_address=VIDEO_CAMERA_IP,
    fps=VIDEO_FPS,
    frame_timeout_ms=VIDEO_FRAME_TIMEOUT_MS,
    reconnect_seconds=VIDEO_RECONNECT_SECONDS,
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动/停止广播分发任务（随 uvicorn lifespan 自动执行）。"""
    task = asyncio.create_task(_broadcast_loop())
    if VIDEO_ENABLED:
        camera_service.start()
    try:
        yield
    finally:
        await asyncio.to_thread(camera_service.stop)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ============================================================
# FastAPI 应用
# ============================================================

hub = DashboardHub()
app = FastAPI(title="Robot Monitor Dashboard", version="0.3", lifespan=_lifespan)

# 模板目录
TEMPLATES_DIR = Path(__file__).parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def index():
    """仪表盘主页面"""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/thermal", response_class=HTMLResponse)
async def thermal_page():
    """Optional full-page view of the same latest thermal frame."""
    return HTMLResponse((TEMPLATES_DIR / "thermal.html").read_text(encoding="utf-8"))


@app.get("/api/stats")
async def api_stats():
    """REST API：获取统计信息"""
    return hub.stats


@app.post("/api/data/temperature/raw")
async def ingest_raw_temperature(
    request: Request,
    x_thermal_sequence: int = Header(...),
    x_thermal_device_index: int = Header(...),
    x_thermal_width: int = Header(...),
    x_thermal_height: int = Header(...),
    x_thermal_slope: int = Header(...),
    x_thermal_offset: int = Header(...),
    x_thermal_timestamp: int = Header(...),
):
    if int(request.headers.get("content-length", "0")) > 2_000_000:
        raise HTTPException(413, "温度帧超过 2 MB 限制")
    payload = await request.body()
    if len(payload) > 2_000_000:
        raise HTTPException(413, "温度帧超过 2 MB 限制")
    try:
        return thermal_frames.ingest(
            payload, sequence=x_thermal_sequence, device_index=x_thermal_device_index,
            width=x_thermal_width, height=x_thermal_height, slope=x_thermal_slope,
            offset=x_thermal_offset, camera_timestamp=x_thermal_timestamp,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/data/temperature/latest")
async def latest_temperature():
    return await asyncio.to_thread(thermal_frames.snapshot)


# ============================================================
# Laser welding multimodal QA
# ============================================================

_qa_instance = None


@app.get("/qa", response_class=HTMLResponse)
async def qa_page():
    """Multimodal QA chat page"""
    qa_html = TEMPLATES_DIR / "qa.html"
    if qa_html.exists():
        return HTMLResponse(qa_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>qa.html not found</h1>", status_code=404)


@app.post("/api/qa")
async def api_qa(payload: dict):
    """QA endpoint: {"question": "...", "image_base64": "..."}"""
    global _qa_instance
    if not QA_ENABLED:
        return JSONResponse({"error": "QA disabled; set QA_ENABLED=True in config.py"}, status_code=400)
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is empty"}, status_code=400)
    if _qa_instance is None:
        _qa_instance = LaserWeldingQA(
            backend=QA_BACKEND,
            model=QA_MODEL,
            api_base=QA_API_BASE,
            api_key=QA_API_KEY,
            timeout=QA_TIMEOUT,
        )
    try:
        answer = await asyncio.to_thread(
            _qa_instance.ask,
            question,
            image_base64=payload.get("image_base64"),
        )
        return {"answer": answer}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# Steel-copper lap joint crack predictor
# ============================================================

_lap_model = None


@app.get("/lap-joint", response_class=HTMLResponse)
async def lap_joint_page():
    """Crack prediction form page"""
    html = TEMPLATES_DIR / "lap_joint.html"
    if html.exists():
        return HTMLResponse(html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>lap_joint.html not found</h1>", status_code=404)


@app.post("/api/predict_lap_joint")
async def api_predict_lap_joint(payload: dict):
    """Crack prediction: {"power":..., "speed":..., ...}"""
    global _lap_model
    try:
        if _lap_model is None:
            from predict_lap_joint import load_model, predict_from_keys
            _lap_model = load_model()
        proba = predict_from_keys(_lap_model[0], _lap_model[1], payload)
        return {"probability": proba, "label": "crack" if proba >= 0.5 else "no_crack",
                "note": "判定阈值 0.5；概率越低越安全"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# Multi-output quality predictor (V1 real data: crack + geometry)
# ============================================================

_quality_predictor = None
_quality_predictor_lock = threading.Lock()


def _get_quality_predictor():
    """懒加载 models/xgb_multiout（首次调用时初始化，线程安全）。"""
    global _quality_predictor
    if _quality_predictor is None:
        with _quality_predictor_lock:
            if _quality_predictor is None:
                from predict_quality import MultiOutputPredictor
                _quality_predictor = MultiOutputPredictor().load()
                logger.info("多输出质量模型已加载: models/xgb_multiout")
    return _quality_predictor


@app.get("/quality", response_class=HTMLResponse)
async def quality_page():
    """多输出质量预测 + AI 解释页面"""
    html = TEMPLATES_DIR / "quality.html"
    if html.exists():
        return HTMLResponse(html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>quality.html not found</h1>", status_code=404)


@app.post("/api/predict_quality")
async def api_predict_quality(payload: dict):
    """工艺参数(+横截面位置) → 裂纹概率 + 几何量预测"""
    try:
        predictor = _get_quality_predictor()
        return predictor.predict(payload)
    except Exception:
        logger.exception("quality prediction failed")
        return JSONResponse({"error": "质量预测失败：参数格式不正确"}, status_code=400)


@app.post("/api/quality/ask")
async def api_quality_ask(payload: dict):
    """先做质量预测，再把预测摘要交给多模态 QA 解释（文字版）"""
    global _qa_instance
    if not QA_ENABLED:
        return JSONResponse({"error": "QA disabled; set QA_ENABLED=True in config.py"},
                            status_code=400)
    try:
        predictor = _get_quality_predictor()
        params = {k: v for k, v in payload.items() if k != "question"}
        summary = predictor.summary_text(params)
        question = (payload.get("question") or "").strip()
        prompt = (
            "你是激光焊接质量工程师。下面是一组工艺参数在钢-铜搭接接头数据集"
            "（V1，360 行真实数据，仅 5 道焊缝）上训练的多输出模型预测结果。"
            "请解释预测的工程含义、可能成因与参数调整建议，并明确说明模型局限"
            "（数据量小，结论仅作参考）。\n\n" + summary
            + ("\n\n用户问题：" + question if question else "")
        )
        if _qa_instance is None:
            _qa_instance = LaserWeldingQA(
                backend=QA_BACKEND,
                model=QA_MODEL,
                api_base=QA_API_BASE,
                api_key=QA_API_KEY,
                timeout=QA_TIMEOUT,
            )
        answer = await asyncio.to_thread(_qa_instance.ask, prompt)
        return {"answer": answer, "prediction_summary": summary}
    except Exception:
        logger.exception("quality QA failed")
        return JSONResponse({"error": "质量问答失败，请检查 QA 配置或稍后重试"}, status_code=500)


# ============================================================
# Industrial camera video panel
# ============================================================


@app.get("/api/video/status")
async def api_video_status():
    """Return current industrial-camera acquisition status."""
    status = camera_service.status()
    status["enabled"] = VIDEO_ENABLED
    status["source_type"] = VIDEO_SOURCE_TYPE
    status["fps_limit"] = VIDEO_FPS
    return status


@app.get("/api/video/devices")
async def api_video_devices():
    try:
        devices = await asyncio.to_thread(camera_service.discover)
        return {"devices": devices}
    except Exception as exc:
        return JSONResponse({"devices": [], "error": str(exc)}, status_code=503)


@app.post("/api/video/start")
async def api_video_start():
    if not VIDEO_ENABLED:
        return JSONResponse({"error": "相机在 test/config.py 中被禁用"}, status_code=409)
    await asyncio.to_thread(camera_service.start)
    return camera_service.status()


@app.post("/api/video/stop")
async def api_video_stop():
    await asyncio.to_thread(camera_service.stop)
    return camera_service.status()


@app.get("/api/video/stream")
async def api_video_stream(request: Request):
    """Multipart PNG stream displayed by the dashboard's <img> element."""
    boundary = "laserweldingframe"

    async def frames():
        sequence = 0
        while not await request.is_disconnected():
            sequence, png = await asyncio.to_thread(
                camera_service.wait_for_frame, sequence, 5.0
            )
            if png is None:
                await asyncio.sleep(0.2)
                continue
            yield (
                f"--{boundary}\r\nContent-Type: image/png\r\n"
                f"Content-Length: {len(png)}\r\n\r\n"
            ).encode("ascii") + png + b"\r\n"

    return StreamingResponse(
        frames(), media_type=f"multipart/x-mixed-replace; boundary={boundary}"
    )


@app.websocket("/ws/video")
async def video_ws(ws: WebSocket):
    """Compatibility status channel; image data uses /api/video/stream."""
    await ws.accept()

    def _status():
        return {
            "type": "video_status",
            "data": camera_service.status(),
        }

    await ws.send_json(_status())
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "status":
                await ws.send_json(_status())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass



@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket 端点：实时推送数据。

    每个客户端注册一个独立的有界发送队列；数据由单一广播任务
    (fanout) 投递给所有客户端，本端点只负责：
      1. reader 任务：接收客户端消息（如 "stats"），回复统计；
      2. sender 任务：从自己的发送队列取消息发送。
    慢客户端只阻塞自己的 sender，不会影响其他客户端。
    """
    await ws.accept()
    outbox: asyncio.Queue[dict] = asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE)
    hub.register(ws, outbox)

    # 发送初始统计快照
    await outbox.put({"type": "stats", "data": hub.stats})

    stop = asyncio.Event()

    async def _reader() -> None:
        """接收客户端消息（心跳/手动刷新用）"""
        try:
            while True:
                msg = await ws.receive_text()
                if msg == "stats":
                    hub.send_to(ws, {"type": "stats", "data": hub.stats})
        except Exception:
            pass  # 断开或异常 → 停止
        finally:
            stop.set()

    async def _sender() -> None:
        """从本客户端发送队列取消息并发送"""
        try:
            while True:
                message = await outbox.get()
                await ws.send_json(message)
        except Exception:
            pass
        finally:
            stop.set()

    tasks = [asyncio.create_task(_reader()), asyncio.create_task(_sender())]
    try:
        await stop.wait()
    except Exception as e:
        logger.warning(f"WebSocket 异常: {e}")
    finally:
        hub.unregister(ws)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ============================================================
# 独立线程启动器
# ============================================================

def _run_uvicorn():
    """在独立线程中运行 uvicorn"""
    uvicorn.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",
        access_log=False,
    )


def start_dashboard() -> threading.Thread:
    """
    启动仪表盘 Web 服务（非阻塞）。

    返回线程对象，主程序可 join 或 daemon。
    """
    t = threading.Thread(target=_run_uvicorn, daemon=True, name="dashboard")
    t.start()
    logger.info(f"仪表盘已启动 → http://localhost:{DASHBOARD_PORT}")
    return t
