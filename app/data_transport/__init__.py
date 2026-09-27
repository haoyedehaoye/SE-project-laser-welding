"""Transport primitives for normalized device events."""

from .event_bus import EventBus
from .pipeline import DataPipeline

__all__ = ["DataBus", "DataPipeline", "EventBus"]

# Compatibility alias for integrations that describe this component as a data bus.
DataBus = EventBus
