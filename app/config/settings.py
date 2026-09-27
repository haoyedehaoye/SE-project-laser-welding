"""TOML settings with deployment environment overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app.toml"


@dataclass(slots=True)
class SystemSettings:
    name: str = "激光焊接设备监控系统"
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"


@dataclass(slots=True)
class WebSettings:
    host: str = "0.0.0.0"
    port: int = 18080


@dataclass(slots=True)
class CameraSettings:
    enabled: bool = True
    device_id: str = "camera-01"
    backend: str = "hikrobot_mvs"
    serial_number: str = ""
    ip_address: str = ""
    mvs_python_path: str = ""
    frame_timeout_ms: int = 1000
    reconnect_interval_seconds: float = 3.0
    preview_max_fps: float = 15.0
    png_compression: int = 3
    packet_size_auto: bool = True


@dataclass(slots=True)
class CurrentSettings:
    enabled: bool = False
    device_id: str = "stm32-01"
    port: str = "COM3"
    baudrate: int = 115200
    timeout_seconds: float = 0.1
    reconnect_interval_seconds: float = 2.0
    simulated: bool = False
    sample_rate_hz: float = 100.0


@dataclass(slots=True)
class AnalysisSettings:
    enabled: bool = True
    model_dir: Path = PROJECT_ROOT / "models"


@dataclass(slots=True)
class StorageSettings:
    recording_dir: Path = PROJECT_ROOT / "recordings"
    retention_days: int = 3


@dataclass(slots=True)
class AppSettings:
    system: SystemSettings = field(default_factory=SystemSettings)
    web: WebSettings = field(default_factory=WebSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    current: CurrentSettings = field(default_factory=CurrentSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)


def _path(value: str | Path, base: Path) -> Path:
    result = Path(value)
    return result if result.is_absolute() else base / result


def load_settings(path: str | Path | None = None) -> AppSettings:
    config_path = Path(path or os.getenv("LASER_WELDING_CONFIG", DEFAULT_CONFIG_PATH))
    data: dict = {}
    if config_path.exists():
        with config_path.open("rb") as stream:
            data = tomllib.load(stream)

    base = config_path.parent.parent if config_path.parent.name == "config" else PROJECT_ROOT
    system_data = data.get("system", {})
    web_data = data.get("web", {})
    camera_data = data.get("camera", {})
    current_data = data.get("current", {})
    analysis_data = data.get("analysis", {})
    storage_data = data.get("storage", {})
    return AppSettings(
        system=SystemSettings(
            name=system_data.get("name", "激光焊接设备监控系统"),
            log_level=os.getenv("LASER_WELDING_LOG_LEVEL", system_data.get("log_level", "INFO")),
            log_dir=_path(system_data.get("log_dir", "logs"), base),
        ),
        web=WebSettings(
            host=os.getenv("LASER_WELDING_HOST", web_data.get("host", "0.0.0.0")),
            port=int(os.getenv("LASER_WELDING_PORT", web_data.get("port", 18080))),
        ),
        camera=CameraSettings(
            enabled=bool(camera_data.get("enabled", True)),
            device_id=camera_data.get("device_id", "camera-01"),
            backend=camera_data.get("backend", "hikrobot_mvs"),
            serial_number=os.getenv("HIK_CAMERA_SERIAL", camera_data.get("serial_number", "")),
            ip_address=os.getenv("HIK_CAMERA_IP", camera_data.get("ip_address", "")),
            mvs_python_path=os.getenv("MVS_PYTHON_PATH", camera_data.get("mvs_python_path", "")),
            frame_timeout_ms=int(camera_data.get("frame_timeout_ms", 1000)),
            reconnect_interval_seconds=float(camera_data.get("reconnect_interval_seconds", 3.0)),
            preview_max_fps=float(camera_data.get("preview_max_fps", 15.0)),
            png_compression=int(camera_data.get("png_compression", 3)),
            packet_size_auto=bool(camera_data.get("packet_size_auto", True)),
        ),
        current=CurrentSettings(
            enabled=bool(current_data.get("enabled", False)),
            device_id=current_data.get("device_id", "stm32-01"),
            port=current_data.get("port", "COM3"),
            baudrate=int(current_data.get("baudrate", 115200)),
            timeout_seconds=float(current_data.get("timeout_seconds", 0.1)),
            reconnect_interval_seconds=float(
                current_data.get("reconnect_interval_seconds", 2.0)
            ),
            simulated=bool(current_data.get("simulated", False)),
            sample_rate_hz=float(current_data.get("sample_rate_hz", 100.0)),
        ),
        analysis=AnalysisSettings(
            enabled=bool(analysis_data.get("enabled", True)),
            model_dir=_path(analysis_data.get("model_dir", "models"), base),
        ),
        storage=StorageSettings(
            recording_dir=_path(storage_data.get("recording_dir", "recordings"), base),
            retention_days=int(storage_data.get("retention_days", 3)),
        ),
    )
