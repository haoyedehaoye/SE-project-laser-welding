from __future__ import annotations

import tempfile
import unittest
import zlib
from pathlib import Path

from app.config import load_settings
from app.core.models import DataMode, DeviceEvent, EventType, SourceKind
from app.core.service import ManagedService, ServiceManager
from app.services.png import PNG_SIGNATURE, encode_png


class _Service(ManagedService):
    def __init__(self, name: str, calls: list[str], fail: bool = False) -> None:
        self._name = name
        self.calls = calls
        self.fail = fail

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        self.calls.append(f"start:{self.name}")
        if self.fail:
            raise RuntimeError("expected")

    def stop(self) -> None:
        self.calls.append(f"stop:{self.name}")


class CoreTests(unittest.TestCase):
    def test_device_event_is_serializable(self) -> None:
        event = DeviceEvent(
            device_id="camera-01",
            source=SourceKind.CAMERA,
            event_type=EventType.STATUS,
            data={"connected": True},
            mode=DataMode.REAL,
        )
        self.assertEqual(event.to_dict()["source"], "camera")
        self.assertEqual(event.to_dict()["data"], {"connected": True})

    def test_settings_load_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            config_file = config_dir / "app.toml"
            config_file.write_text(
                '[web]\nport=19090\n[camera]\nserial_number="ABC"\n[storage]\nretention_days=3\n',
                encoding="utf-8",
            )
            settings = load_settings(config_file)
            self.assertEqual(settings.web.port, 19090)
            self.assertEqual(settings.camera.serial_number, "ABC")
            self.assertEqual(settings.storage.retention_days, 3)
            self.assertEqual(settings.system.log_dir, root / "logs")

    def test_service_manager_rolls_back(self) -> None:
        calls: list[str] = []
        manager = ServiceManager()
        manager.register(_Service("one", calls))
        manager.register(_Service("two", calls, fail=True))
        with self.assertRaises(RuntimeError):
            manager.start_all()
        self.assertEqual(calls, ["start:one", "start:two", "stop:one"])

    def test_png_encoder(self) -> None:
        png = encode_png(2, 1, 1, b"\x00\xff")
        self.assertTrue(png.startswith(PNG_SIGNATURE))
        self.assertIn(b"IHDR", png)
        self.assertIn(b"IDAT", png)


if __name__ == "__main__":
    unittest.main()
