"""
Footprint profile generation for wheel-terrain interaction.

Generates the 2D footprint grid representing the contact patch
of a wheel on the terrain surface.
"""

from typing import Tuple

import numpy as np

from terrain.config import DeformConstrainConf, FootprintConf


class FootprintProfileGenerator:
    """Generates the 2D footprint profile for terrain deformation."""

    def __init__(
        self,
        footprint_conf: FootprintConf,
        deform_constrain_conf: DeformConstrainConf,
        terrain_resolution: float,
    ) -> None:
        self._resolution = terrain_resolution
        self._width = footprint_conf.width
        self._height = footprint_conf.height
        self._x_offset = deform_constrain_conf.x_deform_offset
        self._y_offset = deform_constrain_conf.y_deform_offset
        self.profile: np.ndarray = np.empty((0, 2))
        self.profile_px_width: int = 0
        self.profile_px_height: int = 0

    def create_profile(self) -> Tuple[np.ndarray, int, int]:
        """
        Create the footprint profile grid in the local frame.

        Returns:
            Tuple of (profile [M, 2], pixel_width, pixel_height).
        """
        x = (
            np.linspace(
                -self._height / 2,
                self._height / 2,
                int(self._height / self._resolution) + 1,
            )
            + self._x_offset
        )
        y = (
            np.linspace(
                -self._width / 2,
                self._width / 2,
                int(self._width / self._resolution) + 1,
            )
            + self._y_offset
        )
        xx, yy = np.meshgrid(x, y)
        self.profile = np.column_stack([xx.flatten(), yy.flatten()])
        self.profile_px_width = xx.shape[0]
        self.profile_px_height = yy.shape[1]
        return self.profile, self.profile_px_width, self.profile_px_height
