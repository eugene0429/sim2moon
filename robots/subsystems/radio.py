"""Radio signal strength (RSSI) simulation model.

Models received signal strength indicator between rover and lander
using quadratic distance falloff with configurable noise.

Ported from OmniLRS src/robots/RadioModel.py with improved structure.
"""

import math
import random
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RadioModel:
    """RSSI model with quadratic distance falloff.

    Usage:
        1. Set inputs: rover_position, lander_position
        2. Read outputs: rssi(), distance()
    """

    lander_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rover_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    best_rssi: float = -90.0  # Strongest signal at zero separation (dBm).
    worst_rssi: float = -30.0  # Weakest reading at reference distance (dBm).
    reference_distance: float = 100.0  # Distance (m) at which worst_rssi applies.
    noise_std: float = 1.0  # Gaussian noise standard deviation (dB).

    def set_rover_position(self, position: Tuple[float, float, float]) -> None:
        self.rover_position = position

    def set_lander_position(self, position: Tuple[float, float, float]) -> None:
        self.lander_position = position

    def distance(self) -> float:
        """Euclidean distance between rover and lander."""
        dx = self.rover_position[0] - self.lander_position[0]
        dy = self.rover_position[1] - self.lander_position[1]
        dz = self.rover_position[2] - self.lander_position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def rssi(self) -> float:
        """Compute RSSI with quadratic falloff and noise."""
        dist = self.distance()
        norm = dist / self.reference_distance
        mean_rssi = self.best_rssi + (self.worst_rssi - self.best_rssi) * (norm ** 2)
        return mean_rssi + random.gauss(0.0, self.noise_std)

    def rssi_no_noise(self) -> float:
        """Compute RSSI without measurement noise (for testing)."""
        dist = self.distance()
        norm = dist / self.reference_distance
        return self.best_rssi + (self.worst_rssi - self.best_rssi) * (norm ** 2)
