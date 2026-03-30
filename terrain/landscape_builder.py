"""
SouthPole DEM-based landscape builder.

Loads a real lunar DEM, crops it around the main terrain area,
cuts a rectangular hole for the main procedural terrain,
downsamples for rendering performance, and builds a USD mesh.
"""

import logging
import os
from typing import Tuple

import numpy as np
import yaml

from terrain.config import LandscapeConf

logger = logging.getLogger(__name__)


class LandscapeBuilder:
    """Builds an outer landscape mesh from SouthPole DEM data."""

    def __init__(
        self,
        conf: LandscapeConf,
        main_terrain_size: Tuple[float, float] = (40.0, 40.0),
        main_terrain_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._conf = conf
        self._main_size = main_terrain_size
        self._main_pos = main_terrain_position

    def load_dem(self) -> Tuple[np.ndarray, dict]:
        """Load DEM array and metadata from the configured path.

        Returns:
            Tuple of (dem_array, metadata_dict).

        Raises:
            FileNotFoundError: If dem.npy does not exist at the path.
        """
        dem_file = os.path.join(self._conf.dem_path, "dem.npy")
        meta_file = os.path.join(self._conf.dem_path, "dem.yaml")

        if not os.path.isfile(dem_file):
            raise FileNotFoundError(f"DEM file not found: {dem_file}")

        dem = np.load(dem_file)
        meta = {}
        if os.path.isfile(meta_file):
            with open(meta_file) as f:
                meta = yaml.safe_load(f)

        logger.info("Loaded DEM %s: shape=%s", self._conf.dem_path, dem.shape)
        return dem, meta

    def crop_dem(self, dem: np.ndarray, dem_resolution: float) -> np.ndarray:
        """Crop DEM to crop_size around center.

        Args:
            dem: Full DEM array.
            dem_resolution: Meters per pixel of the DEM.

        Returns:
            Cropped DEM array.
        """
        crop_px = int(self._conf.crop_size / dem_resolution)
        h, w = dem.shape
        half = crop_px // 2

        cy, cx = h // 2, w // 2
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)

        return dem[y0:y1, x0:x1].copy()

    def cut_hole(
        self, dem: np.ndarray, dem_resolution: float, dem_extent: float
    ) -> np.ndarray:
        """Cut a rectangular hole in the DEM where the main terrain sits.

        Args:
            dem: Cropped DEM array (centered at main terrain position).
            dem_resolution: Meters per pixel.
            dem_extent: Total extent of the DEM in meters (assumes square).

        Returns:
            DEM with NaN values in the hole region.
        """
        result = dem.copy()
        h, w = result.shape
        cy, cx = h // 2, w // 2

        half_x = (self._main_size[0] / 2 + self._conf.hole_margin) / dem_resolution
        half_y = (self._main_size[1] / 2 + self._conf.hole_margin) / dem_resolution

        y0 = max(0, int(cy - half_y))
        y1 = min(h, int(cy + half_y))
        x0 = max(0, int(cx - half_x))
        x1 = min(w, int(cx + half_x))

        result[y0:y1, x0:x1] = np.nan
        return result

    def downsample(self, dem: np.ndarray, source_resolution: float) -> np.ndarray:
        """Downsample DEM to target_resolution for rendering performance.

        Uses block averaging, treating NaN as missing (NaN blocks stay NaN).

        Args:
            dem: DEM array (may contain NaN for hole).
            source_resolution: Current meters per pixel.

        Returns:
            Downsampled DEM array.
        """
        factor = int(self._conf.target_resolution / source_resolution)
        if factor <= 1:
            return dem.copy()

        h, w = dem.shape
        new_h = h // factor
        new_w = w // factor
        trimmed = dem[:new_h * factor, :new_w * factor]

        # Reshape into blocks and average, preserving NaN
        blocks = trimmed.reshape(new_h, factor, new_w, factor)
        with np.errstate(invalid="ignore"):
            result = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)

        return result
