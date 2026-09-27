"""Formal ASGI entry point for the monitoring system."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from app import __version__
from app.api.routes import build_router
from app.config import AppSettings, load_settings
from app.core.logging import configure_logging, shutdown_logging
from app.core.service import ServiceManager
from app.data_analysis import PredictiveMaintenanceService, XGBoostModelRegistry
from app.data_collection import CurrentTelemetryCollector, InputChannels
from app.data_collection.video import CameraService
from app.data_collection.thermal_frames import ThermalFrameStore
from app.data_quality import DataQualityService
from app.data_transport import DataPipeline, EventBus


def create_app(settings: AppSettings | None = None, camera: CameraService | None = None) -> FastAPI:
    settings = settings or load_settings()
    bus = EventBus()
    pipeline = DataPipeline(bus, DataQualityService())
    camera = camera or CameraService(settings.camera, publish=pipeline.ingest)
    current = CurrentTelemetryCollector(settings.current, pipeline.ingest)
    inputs = InputChannels(pipeline.ingest)
    thermal_frames = ThermalFrameStore()
    models = XGBoostModelRegistry(settings.analysis.model_dir)
    maintenance = PredictiveMaintenanceService(bus, models)
    manager = ServiceManager()
    manager.register(camera)
    manager.register(current)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        configure_logging(settings.system.log_dir, settings.system.log_level)
        logging.getLogger("app").info("正式监控系统启动，版本 %s", __version__)
        manager.start_all()
        try:
            yield
        finally:
            manager.stop_all()
            logging.getLogger("app").info("正式监控系统已停止")
            shutdown_logging()

    application = FastAPI(title=settings.system.name, version=__version__, lifespan=lifespan)
    application.state.settings = settings
    application.state.camera = camera
    application.state.current = current
    application.state.event_bus = bus
    application.state.thermal_frames = thermal_frames
    application.state.models = models
    application.include_router(
        build_router(
            camera, current, inputs, bus, models, maintenance, Path(__file__).parent / "frontend",
            thermal_frames,
        )
    )
    return application


app = create_app()


def main() -> None:
    settings = load_settings()
    uvicorn.run("app.main:app", host=settings.web.host, port=settings.web.port, reload=False)


if __name__ == "__main__":
    main()
