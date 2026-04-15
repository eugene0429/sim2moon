"""
Base sensor interface.

All sensors implement this abstract interface so they can be managed
uniformly by the SensorManager.
"""

from abc import ABC, abstractmethod
from typing import Any


class Sensor(ABC):
    """Abstract base for all sensor types."""

    def __init__(self, name: str, prim_path: str) -> None:
        self._name = name
        self._prim_path = prim_path
        self._initialized = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def prim_path(self) -> str:
        return self._prim_path

    @property
    def initialized(self) -> bool:
        return self._initialized

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the sensor on the USD stage."""
        ...

    @abstractmethod
    def get_reading(self) -> Any:
        """Return the latest sensor reading."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, path={self._prim_path!r})"
