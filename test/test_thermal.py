from __future__ import annotations

import struct
import unittest

from thermal_frames import ThermalFrameStore
from thermal_receiver import HEADER, MAGIC, parse_header


class ThermalTests(unittest.TestCase):
    def test_sender_header_matches_receiver(self) -> None:
        header = HEADER.pack(MAGIC, 1, 40, 12, 3, 2, 2, 10, -5, 0, 123456, 8)
        fields, size = parse_header(header)
        self.assertEqual(size, 8)
        self.assertEqual(fields["X-Thermal-Sequence"], "12")
        self.assertEqual(fields["X-Thermal-Offset"], "-5")

    def test_store_keeps_latest_and_reports_gap(self) -> None:
        store = ThermalFrameStore()
        payload = struct.pack("<hhhh", 0, 10, 20, 30)
        for sequence in (1, 3):
            store.ingest(payload, sequence=sequence, device_index=0,
                         width=2, height=2, slope=10, offset=5, camera_timestamp=0)
        snapshot = store.snapshot()
        self.assertEqual(snapshot["received_frames"], 2)
        self.assertEqual(snapshot["sequence_gaps"], 1)
        self.assertEqual(snapshot["frame"]["average_celsius"], 6.5)
        self.assertEqual(snapshot["frame"]["grid"], [[5.0, 6.0], [7.0, 8.0]])

    def test_store_rejects_wrong_payload(self) -> None:
        with self.assertRaises(ValueError):
            ThermalFrameStore().ingest(b"\0", sequence=1, device_index=0,
                                       width=2, height=2, slope=10, offset=0,
                                       camera_timestamp=0)


if __name__ == "__main__":
    unittest.main()
