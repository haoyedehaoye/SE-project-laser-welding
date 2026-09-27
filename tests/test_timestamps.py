from __future__ import annotations

import struct
import time
import unittest

from fastapi.testclient import TestClient

from app.core.models import DataMode, DeviceEvent, EventType, SourceKind
from app.data_quality import DataQualityService
from app.data_transport import DataPipeline, EventBus
from app.main import create_app


class TimestampTests(unittest.TestCase):
    def test_pipeline_stamps_host_receive_time_and_serializes_utc(self) -> None:
        bus = EventBus()
        pipeline = DataPipeline(bus, DataQualityService())
        event = DeviceEvent(
            device_id="robot-01",
            source=SourceKind.ROBOT,
            event_type=EventType.TELEMETRY,
            captured_at=1_700_000_000.125,
            received_at=1.0,
            data={"speed": 1.0, "elapsed_time": 2.0},
            mode=DataMode.REAL,
        )
        before = time.time()
        item = pipeline.ingest(event).to_dict()["event"]
        after = time.time()
        self.assertEqual(item["captured_at_utc"], "2023-11-14T22:13:20.125Z")
        self.assertTrue(before <= item["received_at"] <= after)
        self.assertTrue(item["received_at_utc"].endswith("Z"))

    def test_api_preserves_device_time_and_exposes_all_source_times(self) -> None:
        client = TestClient(create_app())
        captured_at = time.time() - 0.1
        current = client.post(
            "/api/data/current",
            json={"current": 12.5, "voltage": 24.0, "captured_at": captured_at},
        )
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["event"]["captured_at"], captured_at)
        self.assertEqual(current.json()["event"]["metadata"]["timestamp_origin"], "device_unix_seconds")

        robot = client.post(
            "/api/data/robot",
            json={"speed": 1.2, "elapsed_time": 10.0},
        )
        self.assertEqual(robot.status_code, 200)
        self.assertEqual(robot.json()["event"]["metadata"]["timestamp_origin"], "host_received")

        thermal = client.post(
            "/api/data/temperature/raw",
            content=struct.pack("<hhhh", 1, 2, 3, 4),
            headers={
                "X-Thermal-Sequence": "1",
                "X-Thermal-Device-Index": "0",
                "X-Thermal-Width": "2",
                "X-Thermal-Height": "2",
                "X-Thermal-Slope": "10",
                "X-Thermal-Offset": "0",
                "X-Thermal-Timestamp": "123456",
            },
        )
        self.assertEqual(thermal.status_code, 200)
        self.assertEqual(thermal.json()["camera_timestamp_raw"], 123456)
        self.assertTrue(thermal.json()["received_at_utc"].endswith("Z"))

        sources = client.get("/api/data/timestamps").json()["sources"]
        self.assertEqual(sources["stm32"]["captured_at"], captured_at)
        self.assertEqual(sources["robot"]["timestamp_origin"], "host_received")
        self.assertEqual(sources["thermal"]["source_timestamp_raw"], 123456)
        self.assertIsNone(sources["camera"])


if __name__ == "__main__":
    unittest.main()
