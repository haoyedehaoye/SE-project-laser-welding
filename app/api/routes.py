"""HTTP/WebSocket transport; business logic remains in the five domain modules."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.core.models import DataMode, DeviceEvent, EventType, SourceKind, capture_timestamp
from app.data_analysis import PredictiveMaintenanceService, XGBoostModelRegistry
from app.data_collection import CurrentTelemetryCollector, InputChannels
from app.data_collection.video import CameraService
from app.data_collection.thermal_frames import ThermalFrameStore
from app.data_transport import EventBus


def build_router(
    camera: CameraService,
    current: CurrentTelemetryCollector,
    inputs: InputChannels,
    bus: EventBus,
    models: XGBoostModelRegistry,
    maintenance: PredictiveMaintenanceService,
    frontend_dir: Path,
    thermal_frames: ThermalFrameStore,
) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((frontend_dir / "index.html").read_text(encoding="utf-8"))

    @router.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "camera": camera.status()["state"],
            "current": current.status()["state"],
            "modules": [
                "data_collection", "data_quality", "data_analysis", "frontend", "data_transport"
            ],
        }

    @router.get("/api/data/latest")
    async def latest(source: str | None = None) -> dict:
        return {"item": bus.latest(source)}

    @router.get("/api/data/history")
    async def history(limit: int = 100) -> dict:
        return {"items": bus.snapshot(limit)}

    @router.get("/api/data/timestamps")
    async def data_timestamps() -> dict:
        sources = {}
        for source in ("stm32", "thermal", "robot", "camera"):
            latest_item = bus.latest(source)
            if latest_item is None:
                sources[source] = None
                continue
            event = latest_item["event"]
            sources[source] = {
                "device_id": event["device_id"],
                "captured_at": event["captured_at"],
                "captured_at_utc": event["captured_at_utc"],
                "received_at": event["received_at"],
                "received_at_utc": event["received_at_utc"],
                "timestamp_origin": event["metadata"].get("timestamp_origin", "host_received"),
                "source_timestamp_raw": event["metadata"].get("camera_timestamp_raw"),
            }
        return {"sources": sources}

    @router.post("/api/data/current")
    async def ingest_current(payload: dict[str, Any]) -> dict:
        try:
            event = DeviceEvent(
                device_id=str(payload.get("device_id", "stm32-api")),
                source=SourceKind.STM32,
                event_type=EventType.TELEMETRY,
                captured_at=capture_timestamp(payload.get("captured_at")),
                data={"current": float(payload["current"]), "voltage": float(payload["voltage"])},
                units={"current": "A", "voltage": "V"},
                mode=DataMode.REAL,
                metadata={
                    "timestamp_origin": "device_unix_seconds" if payload.get("captured_at") is not None else "host_received"
                },
            )
            return _envelope(current.publish(event))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, f"无效电流数据: {exc}") from exc

    @router.post("/api/data/temperature")
    async def ingest_temperature(payload: dict[str, Any]) -> dict:
        try:
            return _envelope(inputs.temperature_matrix(
                payload["temperature_matrix"], payload.get("captured_at"),
                str(payload.get("device_id", "thermal-01")),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, f"无效温度矩阵: {exc}") from exc

    @router.post("/api/data/temperature/raw")
    async def ingest_raw_temperature(
        request: Request,
        x_thermal_sequence: int = Header(...),
        x_thermal_device_index: int = Header(...),
        x_thermal_width: int = Header(...),
        x_thermal_height: int = Header(...),
        x_thermal_slope: int = Header(...),
        x_thermal_offset: int = Header(...),
        x_thermal_timestamp: int = Header(...),
    ) -> dict:
        try:
            summary = thermal_frames.ingest(
                await request.body(),
                sequence=x_thermal_sequence,
                device_index=x_thermal_device_index,
                width=x_thermal_width,
                height=x_thermal_height,
                slope=x_thermal_slope,
                offset=x_thermal_offset,
                camera_timestamp=x_thermal_timestamp,
            )
            event = _envelope(inputs.temperature_summary(summary))
            return {
                "received_sequence": x_thermal_sequence,
                "captured_at": event["event"]["captured_at"],
                "captured_at_utc": event["event"]["captured_at_utc"],
                "received_at": event["event"]["received_at"],
                "received_at_utc": event["event"]["received_at_utc"],
                "camera_timestamp_raw": x_thermal_timestamp,
            }
        except ValueError as exc:
            raise HTTPException(422, f"无效温度帧: {exc}") from exc

    @router.post("/api/data/robot")
    async def ingest_robot(payload: dict[str, Any]) -> dict:
        try:
            return _envelope(inputs.robot_motion(
                float(payload["speed"]), float(payload["elapsed_time"]),
                payload.get("captured_at"), str(payload.get("device_id", "robot-01")),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, f"无效机械臂数据: {exc}") from exc

    @router.websocket("/ws/events")
    async def events(websocket: WebSocket) -> None:
        await websocket.accept()
        sequence = 0
        try:
            while True:
                items = await asyncio.to_thread(bus.wait_after, sequence, 5.0)
                for item in items:
                    sequence = max(sequence, item["sequence"])
                    await websocket.send_json(item)
                if not items:
                    await websocket.send_json({"type": "heartbeat", "sequence": sequence})
        except WebSocketDisconnect:
            return

    @router.get("/api/models")
    async def model_list() -> dict:
        descriptions = models.describe()
        return {
            "models": descriptions,
            "underlying_model_count": sum(item["model_count"] for item in descriptions),
        }

    @router.post("/api/analysis/process")
    async def process_analysis(payload: dict[str, Any]) -> dict:
        return await _model_call(models.predict_process, payload)

    @router.post("/api/analysis/anomaly")
    async def anomaly_analysis(payload: dict[str, Any]) -> dict:
        return await _model_call(models.predict_anomaly, payload.get("samples", []))

    @router.post("/api/analysis/all")
    async def all_analysis(payload: dict[str, Any]) -> dict:
        return await _model_call(
            models.predict_all, payload.get("parameters", {}), payload.get("samples")
        )

    @router.get("/api/maintenance/readiness")
    async def maintenance_readiness() -> dict:
        result = maintenance.readiness()
        result["modalities"]["weld_video"] = camera.status()["details"].get(
            "frames_received", 0
        ) > 0
        result["available_modalities"] = sum(result["modalities"].values())
        return result

    @router.get("/api/current/status")
    async def current_status() -> dict:
        return current.status()

    @router.get("/api/camera/status")
    async def camera_status() -> dict:
        return camera.status()

    @router.get("/api/camera/devices")
    async def camera_devices():
        try:
            devices = await asyncio.to_thread(camera.discover)
            return {"devices": devices}
        except Exception as exc:
            return JSONResponse({"devices": [], "error": str(exc)}, status_code=503)

    @router.post("/api/camera/start")
    async def camera_start() -> dict:
        await asyncio.to_thread(camera.start)
        return camera.status()

    @router.post("/api/camera/stop")
    async def camera_stop() -> dict:
        await asyncio.to_thread(camera.stop)
        return camera.status()

    @router.get("/api/camera/stream")
    async def camera_stream(request: Request) -> StreamingResponse:
        boundary = "laserweldingframe"

        async def frames():
            sequence = 0
            while not await request.is_disconnected():
                sequence, png = await asyncio.to_thread(camera.wait_for_frame, sequence, 5.0)
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

    return router


async def _model_call(function, *args) -> dict:
    try:
        return await asyncio.to_thread(function, *args)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        raise HTTPException(503, f"模型不可用: {exc}") from exc


def _envelope(value: object) -> dict:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise HTTPException(500, "数据管线返回了未知类型")
