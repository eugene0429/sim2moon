from abc import ABC, abstractmethod

from core.enums import SimulatorMode


class BaseEnvironment(ABC):
    """Abstract base class for all simulation environments.

    Concrete subclasses must implement: build_scene, load, instantiate_scene,
    update, and reset. deform_terrain and apply_terramechanics are optional
    (default no-op).
    """

    def __init__(self, stage, mode: SimulatorMode) -> None:
        self._stage = stage
        self._mode = mode

    @abstractmethod
    def build_scene(self) -> None:
        """Build or load the USD scene."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load environment assets (terrain, rocks, etc.)."""
        ...

    @abstractmethod
    def instantiate_scene(self) -> None:
        """Post-renderer initialization."""
        ...

    @abstractmethod
    def update(self, dt: float) -> None:
        """Per-frame environment update."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset environment to initial state."""
        ...

    def deform_terrain(self) -> None:
        """Apply terrain deformation. Override in environments that support it."""
        pass

    def apply_terramechanics(self) -> None:
        """Apply terramechanics forces. Override in environments that support it."""
        pass
