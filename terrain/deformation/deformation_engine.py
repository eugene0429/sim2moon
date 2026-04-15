"""
Terrain deformation engine for wheel sinkage simulation.

Models the terrain response to wheel contact forces using:
- Footprint profile generation
- Force-depth regression (linear)
- Depth distribution (uniform/sinusoidal/trapezoidal grouser pattern)
- Boundary distribution (uniform/parabolic/trapezoidal angle of repose)
- Exponential decay for repeated passes
"""

from typing import Tuple

import numpy as np

from terrain.config import (
    BoundaryDistributionConf,
    DeformationEngineConf,
    DepthDistributionConf,
    FootprintConf,
)
from terrain.deformation.footprint import FootprintProfileGenerator


# --------------------------------------------------------------------------- #
# Depth distribution generators
# --------------------------------------------------------------------------- #

def _uniform_depth(profile_height: int) -> np.ndarray:
    return np.ones(profile_height, dtype=np.float32)


def _sinusoidal_depth(wave_freq: float, profile_height: int) -> np.ndarray:
    return np.cos(wave_freq * np.pi * np.linspace(-1, 1, profile_height))


def _trapezoidal_depth(wave_freq: float, profile_height: int) -> np.ndarray:
    t = np.linspace(-1, 1, profile_height)
    period = 2 / wave_freq
    rise = 0.25 * period
    high = 0.25 * period
    fall = 0.25 * period

    t_shifted = t + 0.5 * period
    phase = t_shifted % period

    y = np.zeros_like(phase)
    # Rising edge
    mask_rise = phase < rise
    y[mask_rise] = phase[mask_rise] / rise
    # High plateau
    mask_high = (~mask_rise) & (phase < rise + high)
    y[mask_high] = 1.0
    # Falling edge
    mask_fall = (~mask_rise) & (~mask_high) & (phase < rise + high + fall)
    y[mask_fall] = 1.0 - (phase[mask_fall] - rise - high) / fall

    return -1 + y * 2


def _build_depth_distribution(
    cfg: DepthDistributionConf,
    footprint_cfg: FootprintConf,
    px_width: int,
    px_height: int,
) -> np.ndarray:
    """Build the full 2D depth distribution mask."""
    if cfg.distribution == "uniform":
        y_slice = _uniform_depth(px_height)
    elif cfg.distribution == "sinusoidal":
        y_slice = _sinusoidal_depth(cfg.wave_frequency, px_height)
    elif cfg.distribution == "trapezoidal":
        y_slice = _trapezoidal_depth(cfg.wave_frequency, px_height)
    else:
        raise ValueError(f"Unknown depth distribution: {cfg.distribution}")

    return np.tile(y_slice[None, :], (px_width, 1)).reshape(-1)


# --------------------------------------------------------------------------- #
# Boundary distribution generators
# --------------------------------------------------------------------------- #

def _uniform_boundary(px_width: int) -> np.ndarray:
    return -np.ones(px_width, dtype=np.float32)


def _parabolic_boundary(px_width: int) -> np.ndarray:
    y = np.linspace(-1, 1, px_width)
    return y ** 2 - 1


def _trapezoidal_boundary(angle_of_repose: float, px_width: int) -> np.ndarray:
    tan = np.tan(angle_of_repose)
    y = np.linspace(-1, 1, px_width)
    mask = (np.abs(y) >= 1 - (1 / tan)).astype(np.float32)
    return mask * (tan * np.abs(y) - tan + 1) - 1


def _build_boundary_distribution(
    cfg: BoundaryDistributionConf,
    px_width: int,
    px_height: int,
) -> np.ndarray:
    """Build the full 2D boundary distribution mask."""
    if cfg.distribution == "uniform":
        x_slice = _uniform_boundary(px_width)
    elif cfg.distribution == "parabolic":
        x_slice = _parabolic_boundary(px_width)
    elif cfg.distribution == "trapezoidal":
        x_slice = _trapezoidal_boundary(cfg.angle_of_repose, px_width)
    else:
        raise ValueError(f"Unknown boundary distribution: {cfg.distribution}")

    return np.tile(x_slice[None, :], (px_height, 1)).reshape(-1)


# --------------------------------------------------------------------------- #
# Deformation engine
# --------------------------------------------------------------------------- #

class DeformationEngine:
    """
    Simulates terrain deformation from wheel contact.

    Pipeline per frame:
    1. Project footprint into world coordinates
    2. Compute sinkage depth from contact forces (linear regression)
    3. Apply depth distribution (grouser pattern)
    4. Apply boundary distribution (angle of repose)
    5. Modify DEM pixels with exponential decay
    """

    def __init__(self, cfg: DeformationEngineConf) -> None:
        self._resolution = cfg.terrain_resolution
        self._sim_width = cfg.terrain_width / cfg.terrain_resolution
        self._sim_height = cfg.terrain_height / cfg.terrain_resolution
        self._num_links = cfg.num_links
        self._force_depth = cfg.force_depth_regression
        self._decay_ratio = cfg.deform_constrain.deform_decay_ratio

        # Build footprint profile
        fp_gen = FootprintProfileGenerator(
            cfg.footprint, cfg.deform_constrain, cfg.terrain_resolution
        )
        self._profile, self._px_width, self._px_height = fp_gen.create_profile()

        # Build depth and boundary distributions
        self._depth_dist = _build_depth_distribution(
            cfg.depth_distribution, cfg.footprint, self._px_width, self._px_height
        )
        self._boundary_dist = _build_boundary_distribution(
            cfg.boundary_distribution, self._px_width, self._px_height
        )

        # Pre-allocate buffers
        n_pts = self._profile.shape[0]
        self._headings = np.zeros((cfg.num_links, 2), dtype=np.float32)
        self._profile_global = np.zeros((cfg.num_links * n_pts, 2), dtype=np.float32)
        self._depth = np.zeros(cfg.num_links * n_pts, dtype=np.float32)

    def _project_footprint(
        self, world_positions: np.ndarray, world_orientations: np.ndarray
    ) -> None:
        """Transform footprint from local to world coordinates."""
        n_links = world_positions.shape[0]
        # Resize buffers if actual wheel count differs from config
        if self._headings.shape[0] != n_links:
            self._headings = np.zeros((n_links, 2), dtype=np.float32)

        # Extract heading from quaternion (yaw only)
        q = world_orientations
        self._headings[:, 0] = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
        self._headings[:, 1] = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)

        proj = np.zeros((world_positions.shape[0], self._profile.shape[0], 2))
        proj[:, :, 0] = (
            self._profile[:, 0] * self._headings[:, 1, None]
            - self._profile[:, 1] * self._headings[:, 0, None]
            + world_positions[:, 0][:, None]
        )
        proj[:, :, 1] = (
            self._profile[:, 0] * self._headings[:, 0, None]
            + self._profile[:, 1] * self._headings[:, 1, None]
            + world_positions[:, 1][:, None]
        )
        self._profile_global = proj.reshape(-1, 2)

    def _compute_depth(self, normal_forces: np.ndarray) -> None:
        """Compute sinkage depth from contact forces via linear regression."""
        fdr = self._force_depth
        amplitude = fdr.amplitude_slope * normal_forces + fdr.amplitude_intercept
        mean_val = fdr.mean_slope * normal_forces + fdr.mean_intercept

        depth = self._boundary_dist[None, :] * (
            amplitude[:, None] / 2.0 * self._depth_dist[None, :] - mean_val[:, None]
        )
        self._depth = depth.reshape(-1)

    def deform(
        self,
        dem_delta: np.ndarray,
        num_pass: np.ndarray,
        world_positions: np.ndarray,
        world_orientations: np.ndarray,
        normal_forces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply deformation to the DEM delta.

        Args:
            dem_delta: Accumulated deformation delta array.
            num_pass: Per-pixel pass count for decay calculation.
            world_positions: Link positions [N, 3].
            world_orientations: Link quaternions [N, 4].
            normal_forces: Normal (Z) forces per link [N].

        Returns:
            Updated (dem_delta, num_pass).
        """
        self._project_footprint(world_positions, world_orientations)
        self._compute_depth(normal_forces)

        # Vectorised pixel update
        xs = (self._profile_global[:, 0] / self._resolution).astype(np.intp)
        ys = (self._sim_height - self._profile_global[:, 1] / self._resolution).astype(np.intp)

        h, w = dem_delta.shape
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[valid]
        ys = ys[valid]
        depths = self._depth[valid]

        # Apply deformation with decay (sequential for correct num_pass accumulation)
        decay = self._decay_ratio
        for i in range(len(xs)):
            yi, xi = ys[i], xs[i]
            dem_delta[yi, xi] += depths[i] * (decay ** num_pass[yi, xi])
            num_pass[yi, xi] += 1

        return dem_delta, num_pass
