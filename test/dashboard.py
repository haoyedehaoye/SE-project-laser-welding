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
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from data_types import DataPoint

logger = logging.getLogger("dashboard")

# ============================================================
# 配置
# ============================================================
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 18080
MAX_HISTORY = 200          # 每个客户端保留的历史点数
BROADCAST_INTERVAL = 0.1   # 广播间隔（秒）


# ============================================================
# 数据缓冲 + 广播管理
# ============================================================

class DashboardHub:
    """
    WebSocket 广播中心。

    特点：
      - 缓冲最新数据，批量推送（降低广播频率）
      - 为每个客户端独立维护历史队列
      - 线程安全
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[WebSocket, deque[dict]] = {}  # ws → 历史队列
        self._pending: list[dict] = []                     # 待广播数据
        self._stats = {"total_pushed": 0, "normal": 0, "suspicious": 0, "anomaly": 0}
        self._start_time = time.time()

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
            self._pending.append(payload)

            # 统计
            self._stats["total_pushed"] += 1
            if q:
                self._stats[q.label] = self._stats.get(q.label, 0) + 1

    def register(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients[ws] = deque(maxlen=MAX_HISTORY)

    def unregister(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.pop(ws, None)

    def drain(self) -> list[dict]:
        """取出待广播数据并分发到各客户端历史队列"""
        with self._lock:
            batch = self._pending
            self._pending = []

            for payload in batch:
                for history in self._clients.values():
                    history.append(payload)

        return batch

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def stats(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time
            return {**self._stats, "uptime": round(uptime, 1), "clients": len(self._clients)}


# ============================================================
# FastAPI 应用
# ============================================================

hub = DashboardHub()
app = FastAPI(title="Robot Monitor Dashboard", version="0.3")

# 模板目录
TEMPLATES_DIR = Path(__file__).parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def index():
    """仪表盘主页面"""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/api/stats")
async def api_stats():
    """REST API：获取统计信息"""
    return hub.stats


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket 端点：实时推送数据"""
    await ws.accept()
    hub.register(ws)

    # 发送初始统计快照
    await ws.send_json({"type": "stats", "data": hub.stats})

    try:
        while True:
            # 客户端发消息 → 回复统计（心跳/手动刷新用）
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=BROADCAST_INTERVAL)
                if msg == "stats":
                    await ws.send_json({"type": "stats", "data": hub.stats})
            except asyncio.TimeoutError:
                pass  # 超时正常，继续广播

            # 广播新数据
            batch = hub.drain()
            if batch:
                try:
                    await ws.send_json({"type": "data", "data": batch})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket 异常: {e}")
    finally:
        hub.unregister(ws)


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