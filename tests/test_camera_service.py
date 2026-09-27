from __future__ import annotations

import time
import unittest

from app.config.settings import CameraSettings
from app.devices.camera.base import CameraAdapter, CameraDevice, CameraFrame
from app.services.camera_service import CameraService
from app.data_quality import DataQualityService
from app.data_transport import DataPipeline, EventBus


class _TestAdapter(CameraAdapter):
    """Unit-test double only; it is not exposed as a product camera source."""

    def __init__(self) -> None:
        self.closed = False
        self.counter = 0

    def enumerate_devices(self) -> list[CameraDevice]:
        return [CameraDevice(0, "gige", "MV-CS050-10GM", "TEST", "169.254.1.2")]

    def open(self, serial_number: str = "", ip_address: str = "") -> CameraDevice:
        return self.enumerate_devices()[0]

    def read(self, timeout_ms: int) -> CameraFrame | None:
        time.sleep(0.005)
        self.counter += 1
        return CameraFrame(2, 2, 1, bytes([0, 64, 128, 255]), time.time(), self.counter)

    def close(self) -> None:
        self.closed = True


class CameraServiceTests(unittest.TestCase):
    def test_frame_metadata_enters_timestamped_event_bus(self) -> None:
        adapter = _TestAdapter()
        bus = EventBus()
        pipeline = DataPipeline(bus, DataQualityService())
        settings = CameraSettings(preview_max_fps=30, reconnect_interval_seconds=0.01)
        service = CameraService(settings, adapter_factory=lambda: adapter, publish=pipeline.ingest)
        service.start()
        try:
            items = bus.wait_after(0, 1.0)
            self.assertTrue(items)
            event = items[0]["event"]
            self.assertEqual(event["source"], "camera")
            self.assertEqual(event["metadata"]["timestamp_origin"], "host_frame_read")
            self.assertTrue(event["captured_at_utc"].endswith("Z"))
            self.assertTrue(event["received_at"] >= event["captured_at"])
        finally:
            service.stop()

    def test_service_publishes_preview_and_stops(self) -> None:
        adapter = _TestAdapter()
        settings = CameraSettings(preview_max_fps=30, reconnect_interval_seconds=0.01)
        service = CameraService(settings, adapter_factory=lambda: adapter)
        service.start()
        sequence, png = service.wait_for_frame(0, 1.0)
        self.assertGreaterEqual(sequence, 1)
        self.assertIsNotNone(png)
        self.assertTrue(service.status()["connected"])
        service.stop()
        self.assertTrue(adapter.closed)
        self.assertEqual(service.status()["state"], "stopped")
        self.assertFalse(service.status()["details"]["requested_on"])
        _, png_after_stop = service.wait_for_frame(sequence, 0.01)
        self.assertIsNone(png_after_stop)

    def test_service_can_be_stopped_and_started_again(self) -> None:
        adapters: list[_TestAdapter] = []

        def factory() -> _TestAdapter:
            adapter = _TestAdapter()
            adapters.append(adapter)
            return adapter

        settings = CameraSettings(preview_max_fps=30, reconnect_interval_seconds=0.01)
        service = CameraService(settings, adapter_factory=factory)
        service.start()
        first_sequence, first_png = service.wait_for_frame(0, 1.0)
        self.assertIsNotNone(first_png)
        service.stop()
        stopped_sequence, _ = service.wait_for_frame(first_sequence, 0.01)

        service.start()
        second_sequence, second_png = service.wait_for_frame(stopped_sequence, 1.0)
        self.assertGreater(second_sequence, stopped_sequence)
        self.assertIsNotNone(second_png)
        self.assertTrue(service.status()["details"]["requested_on"])
        service.stop()
        self.assertGreaterEqual(len(adapters), 2)


if __name__ == "__main__":
    unittest.main()
