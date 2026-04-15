"""Thermal simulation model for a six-faced rover body.

Models six exterior faces (+X, -X, +Y, -Y, +Z, -Z) plus one interior node.
Each face temperature follows a first-order response toward a sigmoid-derived
target that depends on the sun view factor. The interior node is the average
of all exterior faces.

Ported from OmniLRS src/robots/ThermalModel.py with improved structure.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np

FACE_NORMALS: Dict[str, np.ndarray] = {
    "+X": np.array((1.0, 0.0, 0.0)),
    "-X": np.array((-1.0, 0.0, 0.0)),
    "+Y": np.array((0.0, 1.0, 0.0)),
    "-Y": np.array((0.0, -1.0, 0.0)),
    "+Z": np.array((0.0, 0.0, 1.0)),
    "-Z": np.array((0.0, 0.0, -1.0)),
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class ThermalModel:
    """Six-face thermal model with sigmoid target temperatures.

    Usage:
        1. Set inputs: rover position, yaw, sun position
        2. Call step(dt) to advance simulation
        3. Read outputs: temperatures()
    """

    min_temp: float = -50.0
    max_temp: float = 100.0
    time_constant: float = 600.0  # Seconds to reach ~63% of target delta.
    sigmoid_gain: float = 8.0
    faces: Iterable[str] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    node_temps: Dict[str, float] = field(default_factory=dict)
    initial_temp: float = 20.0
    rover_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rover_yaw_deg: float = 0.0
    sun_position: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    measurement_noise_std: float = 0.5

    def __post_init__(self) -> None:
        self.faces = tuple(self.faces)
        if not self.node_temps:
            self.node_temps = {face: self.initial_temp for face in self.faces}
            self.node_temps["interior"] = self.initial_temp
        else:
            for face in self.faces:
                self.node_temps.setdefault(face, self.initial_temp)
            self.node_temps.setdefault("interior", self.initial_temp)

    # -- Input setters --

    def set_rover_position(self, position: Tuple[float, float, float]) -> None:
        self.rover_position = position

    def set_rover_yaw(self, yaw_deg: float) -> None:
        self.rover_yaw_deg = yaw_deg

    def set_sun_position(self, position: Tuple[float, float, float]) -> None:
        self.sun_position = position

    # -- Computation --

    def step(self, dt: float) -> None:
        """Advance all node temperatures by dt seconds."""
        view_factors = self.compute_view_factors(self.rover_position, self.sun_position)

        for face in self.faces:
            exposure = _clamp(view_factors.get(face, 0.0), 0.0, 1.0)
            target = self._target_temperature(exposure)
            current = self.node_temps[face]
            delta = (target - current) * (dt / self.time_constant)
            self.node_temps[face] = current + delta

        self.node_temps["interior"] = sum(
            self.node_temps[face] for face in self.faces
        ) / len(self.faces)

    def compute_view_factors(
        self,
        rover_position: Tuple[float, float, float],
        sun_position: Tuple[float, float, float],
    ) -> Dict[str, float]:
        """Compute cosine-based per-face view factors."""
        rover = np.asarray(rover_position, dtype=float)
        sun = np.asarray(sun_position, dtype=float)
        vector = sun - rover
        magnitude = np.linalg.norm(vector)
        if magnitude == 0.0:
            return {face: 0.0 for face in self.faces}

        unit = vector / magnitude
        yaw_rad = math.radians(self.rover_yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        rotation = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        return {
            face: float(
                _clamp(float(np.dot(unit, rotation @ FACE_NORMALS[face])), 0.0, 1.0)
            )
            for face in self.faces
        }

    # -- Outputs --

    def temperatures(self) -> Dict[str, float]:
        """Return noisy temperature readings for all nodes."""
        if self.measurement_noise_std <= 0:
            return dict(self.node_temps)
        return {
            name: value + random.gauss(0.0, self.measurement_noise_std)
            for name, value in self.node_temps.items()
        }

    # -- Internal --

    def _target_temperature(self, view_factor: float) -> float:
        """Sigmoid-based target temperature for a given view factor."""
        midpoint = 0.5
        sigmoid = 1.0 / (1.0 + math.exp(-self.sigmoid_gain * (view_factor - midpoint)))
        return self.min_temp + (self.max_temp - self.min_temp) * sigmoid
