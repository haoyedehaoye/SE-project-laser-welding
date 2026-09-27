"""Lifecycle contracts for long-running application services."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ManagedService(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class ServiceManager:
    """Starts services in order and always stops them in reverse order."""

    def __init__(self) -> None:
        self._services: list[ManagedService] = []
        self._started: list[ManagedService] = []

    def register(self, service: ManagedService) -> None:
        self._services.append(service)

    def start_all(self) -> None:
        try:
            for service in self._services:
                service.start()
                self._started.append(service)
        except Exception:
            self.stop_all()
            raise

    def stop_all(self) -> None:
        for service in reversed(self._started):
            try:
                service.stop()
            except Exception:
                pass
        self._started.clear()
