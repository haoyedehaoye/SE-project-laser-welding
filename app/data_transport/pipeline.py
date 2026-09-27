"""The single path from device collection through validation to consumers."""

from __future__ import annotations

import time

from app.core.models import DeviceEvent
from app.data_quality import DataQualityService
from app.data_transport.event_bus import EventBus, EventEnvelope


class DataPipeline:
    def __init__(self, bus: EventBus, quality: DataQualityService) -> None:
        self.bus = bus
        self.quality = quality

    def ingest(self, event: DeviceEvent) -> EventEnvelope:
        # The host receive time belongs to the pipeline, not the device clock.
        event.received_at = time.time()
        report = self.quality.check(event)
        return self.bus.publish(event, report.to_dict())

