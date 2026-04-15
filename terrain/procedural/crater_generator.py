"""
Crater shape generator using spline profiles.

Generates individual craters from pre-computed half-profile splines, with
randomized ellipticity, surface marks, and rotation. Craters are then
stamped onto a DEM.
"""

import dataclasses
import pickle
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.ndimage import rotate

from terrain.config import CraterGeneratorConf


@dataclasses.dataclass
class CraterData:
    """Metadata for a single generated crater."""

    deformation_spline: Optional[CubicSpline] = None
    marks_spline: Optional[CubicSpline] = None
    marks_intensity: float = 0.0
    size: int = 0
    crater_profile_id: int = 0
    xy_deformation_factor: Tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    coord: Tuple[float, float] = (0.0, 0.0)


class CraterGenerator:
    """Generates crater DEMs from spline profiles with random deformation."""

    def __init__(self, cfg: CraterGeneratorConf) -> None:
        self._profiles_path = cfg.profiles_path
        self._resolution = cfg.resolution
        self._min_xy_ratio = cfg.min_xy_ratio
        self._max_xy_ratio = cfg.max_xy_ratio
        self._z_scale = cfg.z_scale
        self._random_rotation = cfg.random_rotation
        self._pad_size = cfg.pad_size
        self._rng = np.random.default_rng(cfg.seed)
        self._profiles: list = []

        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load half-crater spline profiles from a pickle file.

        Uses PPoly to reconstruct from raw coefficients, avoiding scipy
        version mismatch issues with pickled CubicSpline objects.
        """

        class _RawSpline:
            pass

        class _SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "CubicSpline":
                    return _RawSpline
                return super().find_class(module, name)

        with open(self._profiles_path, "rb") as f:
            raw_profiles = _SafeUnpickler(f).load()

        self._profiles = [PPoly(p.c, p.x) for p in raw_profiles]

    @staticmethod
    def _saturated_gaussian(
        x: np.ndarray, mu1: float, mu2: float, std: float
    ) -> np.ndarray:
        """
        Gaussian that saturates to its maximum between mu1 and mu2.
        Used to smoothly blend surface marks toward crater center.
        """
        shape = x.shape
        x = x.flatten()
        x[x < mu1] = np.exp(-0.5 * ((x[x < mu1] - mu1) / std) ** 2)
        x[x > mu2] = np.exp(-0.5 * ((x[x > mu2] - mu2) / std) ** 2)
        x[(x >= mu1) & (x <= mu2)] = 1.0
        x = x / (std * np.sqrt(2 * np.pi))
        return x.reshape(shape)

    def _centered_distance_matrix(self, cd: CraterData) -> np.ndarray:
        """
        Build a deformed distance matrix for a crater.

        Applies elliptical deformation, surface marks, and rotation to
        produce the radial distance field used by the crater profile.
        """
        n = cd.size

        # Angular coordinate grid
        x_lin, y_lin = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        theta = np.arctan2(y_lin, x_lin)
        fac = cd.deformation_spline(theta / (2 * np.pi) + 0.5)

        # Surface marks
        marks = cd.marks_spline(theta / (2 * np.pi) + 0.5) * n / 2 * cd.marks_intensity

        # Euclidean distance with elliptical deformation
        x_idx, y_idx = np.meshgrid(range(n), range(n))
        dist = np.sqrt(
            ((x_idx - n / 2 + 1) / cd.xy_deformation_factor[0]) ** 2
            + ((y_idx - n / 2 + 1) / cd.xy_deformation_factor[1]) ** 2
        )

        # Apply angular deformation
        dist = dist * fac

        # Smooth marks blend (strongest near mid-radius, fading at edges)
        sat = self._saturated_gaussian(
            dist, 0.15 * n / 2, 0.45 * n / 2, 0.05 * n / 2
        )
        sat = (sat - sat.min()) / (sat.max() - sat.min() + 1e-12)
        dist = dist + marks * sat

        # Rotate
        dist = rotate(dist, cd.rotation, reshape=False, cval=n / 2)

        # Clamp to crater radius
        dist[dist > n / 2] = n / 2

        return dist

    def _apply_profile(self, distance: np.ndarray, cd: CraterData) -> np.ndarray:
        """Apply the selected spline profile to the distance matrix."""
        return self._profiles[cd.crater_profile_id](2 * distance / cd.size)

    def randomize_parameters(self, index: int, size: int) -> CraterData:
        """
        Generate randomized parameters for a single crater.

        Args:
            index: Profile index (-1 for random).
            size: Crater diameter in pixels.

        Returns:
            CraterData with all randomized fields set.
        """
        cd = CraterData()
        cd.size = size + (size % 2 == 0)  # ensure odd

        # Angular deformation spline
        deform_vals = self._rng.uniform(0.95, 1.0, 9)
        deform_vals = np.concatenate([deform_vals, [deform_vals[0]]])
        t = np.linspace(0, 1, deform_vals.shape[0])
        cd.deformation_spline = CubicSpline(t, deform_vals, bc_type=((1, 0.0), (1, 0.0)))

        # Surface marks spline
        marks_vals = self._rng.uniform(0.0, 0.01, 45)
        marks_vals = np.concatenate([marks_vals, [marks_vals[0]]])
        t_marks = np.linspace(0, 1, marks_vals.shape[0])
        cd.marks_spline = CubicSpline(t_marks, marks_vals, bc_type=((1, 0.0), (1, 0.0)))
        cd.marks_intensity = self._rng.uniform(0, 1)

        # Ellipticity
        sx = self._rng.uniform(self._min_xy_ratio, self._max_xy_ratio)
        cd.xy_deformation_factor = (sx, 1.0)

        # Rotation
        cd.rotation = float(int(self._rng.uniform(0, 360)))

        # Profile selection
        if index == -1:
            index = int(self._rng.integers(0, len(self._profiles)))
        elif index >= len(self._profiles):
            raise ValueError(f"Profile index {index} out of range (max {len(self._profiles) - 1})")
        cd.crater_profile_id = index

        return cd

    def generate_single(
        self,
        size: Optional[int] = None,
        index: int = -1,
        crater_data: Optional[CraterData] = None,
    ) -> Tuple[np.ndarray, CraterData]:
        """
        Generate a single crater DEM patch.

        Args:
            size: Crater diameter in pixels (ignored if crater_data provided).
            index: Profile index (-1 for random, ignored if crater_data provided).
            crater_data: Pre-existing CraterData to regenerate from.

        Returns:
            Tuple of (crater DEM patch, CraterData).
        """
        if crater_data is None:
            crater_data = self.randomize_parameters(index, size)

        distance = self._centered_distance_matrix(crater_data)
        crater = (
            self._apply_profile(distance, crater_data)
            * crater_data.size / 2.0
            * self._z_scale
            * self._resolution
        )
        return crater, crater_data

    def generate_craters(
        self,
        dem: np.ndarray,
        coords: Optional[np.ndarray] = None,
        radii: Optional[np.ndarray] = None,
        craters_data: Optional[List[CraterData]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[CraterData]]:
        """
        Stamp multiple craters onto a DEM.

        Args:
            dem: Base DEM to add craters to.
            coords: [N, 2] crater center positions in meters.
            radii: [N] crater radii in meters.
            craters_data: Pre-existing crater data for regeneration.

        Returns:
            Tuple of (modified DEM, mask, list of CraterData).
        """
        pad = self._pad_size
        h, w = dem.shape
        dem_padded = np.zeros((pad * 2 + h, pad * 2 + w))
        mask_padded = np.ones_like(dem_padded)
        dem_padded[pad:-pad, pad:-pad] = dem

        if craters_data is None:
            craters_data = []

        ph, pw = dem_padded.shape

        if len(craters_data) == 0:
            # Generate new craters from coords/radii
            for coord, rad in zip(coords, radii):
                size_px = int(rad * 2 / self._resolution)
                coord_px = coord / self._resolution
                crater_patch, cd = self.generate_single(int(size_px))
                cd.coord = (coord[0], coord[1])

                center_px = (coord_px + pad).astype(np.int64)
                origin_px = (coord_px - cd.size / 2 + pad).astype(np.int64)

                # Clip to padded DEM bounds
                r0 = max(origin_px[0], 0)
                c0 = max(origin_px[1], 0)
                r1 = min(origin_px[0] + cd.size, ph)
                c1 = min(origin_px[1] + cd.size, pw)
                pr0 = r0 - origin_px[0]
                pc0 = c0 - origin_px[1]
                pr1 = pr0 + (r1 - r0)
                pc1 = pc0 + (c1 - c0)

                if r1 > r0 and c1 > c0:
                    dem_padded[r0:r1, c0:c1] += crater_patch[pr0:pr1, pc0:pc1]

                mask_padded = cv2.circle(
                    mask_padded,
                    (int(center_px[1]), int(center_px[0])),
                    int(cd.size / 4),
                    0,
                    -1,
                )
                craters_data.append(cd)
        else:
            # Regenerate from existing crater data
            for cd in craters_data:
                coord_px = np.array(cd.coord) / self._resolution
                crater_patch, cd = self.generate_single(crater_data=cd)

                center_px = (coord_px + pad).astype(np.int64)
                origin_px = (coord_px - cd.size / 2 + pad).astype(np.int64)

                r0 = max(origin_px[0], 0)
                c0 = max(origin_px[1], 0)
                r1 = min(origin_px[0] + cd.size, ph)
                c1 = min(origin_px[1] + cd.size, pw)
                pr0 = r0 - origin_px[0]
                pc0 = c0 - origin_px[1]
                pr1 = pr0 + (r1 - r0)
                pc1 = pc0 + (c1 - c0)

                if r1 > r0 and c1 > c0:
                    dem_padded[r0:r1, c0:c1] += crater_patch[pr0:pr1, pc0:pc1]

                mask_padded = cv2.circle(
                    mask_padded,
                    (int(center_px[1]), int(center_px[0])),
                    int(cd.size / 4),
                    0,
                    -1,
                )

        # Zero out padding region in mask
        mask_padded[:pad + 1, :] = 0
        mask_padded[:, :pad + 1] = 0
        mask_padded[-pad - 1:, :] = 0
        mask_padded[:, -pad - 1:] = 0

        return (
            dem_padded[pad:-pad, pad:-pad],
            mask_padded[pad:-pad, pad:-pad],
            craters_data,
        )
