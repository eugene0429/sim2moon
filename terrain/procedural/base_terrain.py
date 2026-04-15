"""
Base terrain generator using interpolated noise.

Generates a DEM with low-frequency and high-frequency random noise,
simulating large-scale and small-scale terrain features.
"""

import numpy as np
import cv2

from terrain.config import BaseTerrainGeneratorConf


class BaseTerrainGenerator:
    """Generates a random base terrain DEM from interpolated noise."""

    def __init__(self, cfg: BaseTerrainGeneratorConf) -> None:
        self._min_elevation = cfg.min_elevation
        self._max_elevation = cfg.max_elevation
        self._x_pixels = int(cfg.x_size / cfg.resolution)
        self._y_pixels = int(cfg.y_size / cfg.resolution)
        self._z_scale = cfg.z_scale
        self._rng = np.random.default_rng(cfg.seed)

    def generate(self, is_lab: bool = False, is_yard: bool = False) -> np.ndarray:
        """
        Generate a random terrain DEM.

        For lab mode, borders are set to zero to align with catwalks.
        For yard mode, outer ring is zeroed for smooth edges.

        Args:
            is_lab: Whether the terrain is for an indoor lab environment.
            is_yard: Whether the terrain is for an outdoor yard environment.

        Returns:
            DEM array (float32) with shape (x_pixels, y_pixels).
        """
        dem = np.zeros((self._x_pixels, self._y_pixels), dtype=np.float32)

        # Low-frequency noise for large-scale features
        if is_lab:
            lr_noise = np.zeros((4, 4))
            lr_noise[:-1, 1:] = self._rng.uniform(
                self._min_elevation, self._max_elevation, (3, 3)
            )
        elif is_yard:
            lr_noise = np.zeros((7, 7))
            lr_noise[1:-1, 1:-1] = self._rng.uniform(
                self._min_elevation, self._max_elevation, (5, 5)
            )
        else:
            lr_noise = self._rng.uniform(
                self._min_elevation, self._max_elevation, (4, 4)
            )

        hr_noise = cv2.resize(
            lr_noise, (self._y_pixels, self._x_pixels), interpolation=cv2.INTER_CUBIC
        )
        dem += hr_noise

        # High-frequency noise for small-scale features
        lr_hf = self._rng.uniform(
            self._min_elevation * 0.01, self._max_elevation * 0.01, (100, 100)
        )
        hr_hf = cv2.resize(
            lr_hf, (self._y_pixels, self._x_pixels), interpolation=cv2.INTER_CUBIC
        )
        dem += hr_hf

        return dem * self._z_scale
