"""
Rock distribution algorithms for lunar simulation.

Provides power-law size sampling, crater-aware ejecta placement,
wall debris, background scatter, and clustered scatter.
KDTree-based overlap rejection ensures rocks don't intersect.

Extracted from OmniLRS src/environments/rock_manager.py for clean separation.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


@dataclass
class CraterData:
    """Crater info for rock placement. Matches terrain module's CraterData."""
    coord: Tuple[float, float]  # (x_m, y_m) in DEM space
    size: float  # diameter in pixels


@dataclass
class RockPlacement:
    """Result of a rock distribution: 2D positions and scale diameters."""
    positions: np.ndarray   # (N, 2) XY in world meters
    diameters: np.ndarray   # (N,) scale multipliers


class RockDistributor:
    """
    Generates rock placements using various distribution methods.

    All methods produce RockPlacement objects with 2D positions and
    diameter scale multipliers. The RockManager handles 3D positioning
    (Z from DEM) and USD instancing.

    Interface contract:
        distribute(method, rng, terrain_width, terrain_height, resolution, ...) -> RockPlacement
        remove_overlaps(placement, min_gap, native_rock_size) -> RockPlacement
    """

    # -- Power-law sampling --

    @staticmethod
    def sample_power_law(
        rng: np.random.Generator,
        d_min: float,
        d_max: float,
        alpha: float,
        n: int,
    ) -> np.ndarray:
        """
        Sample diameters from a power-law distribution: N(>d) ~ d^(-alpha).

        Args:
            rng: NumPy random generator.
            d_min: Minimum diameter.
            d_max: Maximum diameter.
            alpha: Power-law exponent (must be > 1).
            n: Number of samples.

        Returns:
            Array of sampled diameters.
        """
        u = rng.uniform(0, 1, n)
        exponent = 1.0 - alpha
        return (d_min**exponent + u * (d_max**exponent - d_min**exponent)) ** (1.0 / exponent)

    # -- Distribution methods --

    @staticmethod
    def crater_ejecta(
        rng: np.random.Generator,
        terrain_width: float,
        terrain_height: float,
        resolution: float,
        craters: List[CraterData],
        rocks_per_crater: int = 15,
        rim_inner_ratio: float = 0.8,
        rim_outer_ratio: float = 2.5,
        decay_power: float = 2.0,
        d_min: float = 3.0,
        d_max: float = 8.0,
        alpha: float = 2.2,
        min_crater_radius: float = 1.0,
    ) -> RockPlacement:
        """
        Place rocks around crater rims using ejecta pattern.

        Rocks are concentrated near the rim with radial power-law falloff.
        Rocks closer to the rim get larger maximum diameters.

        Args:
            rng: Random generator.
            terrain_width: Terrain width in meters.
            terrain_height: Terrain height in meters.
            resolution: DEM resolution in meters/pixel.
            craters: List of CraterData from terrain system.
            rocks_per_crater: Mean rocks per crater (Poisson).
            rim_inner_ratio: Inner radius as fraction of crater radius.
            rim_outer_ratio: Outer radius as fraction of crater radius.
            decay_power: Radial decay exponent.
            d_min: Minimum diameter scale.
            d_max: Maximum diameter scale.
            alpha: Power-law size exponent.
            min_crater_radius: Skip craters smaller than this (meters).

        Returns:
            RockPlacement with positions and diameters.
        """
        tw, th = terrain_width, terrain_height
        all_pos, all_diam = [], []

        # Filter eligible craters
        eligible = [
            (c, c.size * resolution / 2)
            for c in craters
            if c.size * resolution / 2 >= min_crater_radius
        ]
        logger.debug(
            "crater_ejecta: %d/%d craters with radius >= %.1fm",
            len(eligible), len(craters), min_crater_radius,
        )

        for crater, cr in eligible:
            # Transform DEM coords to mesh world space
            # DEM row -> mesh Y (flipped), DEM col -> mesh X
            cx = crater.coord[1]
            cy = th - crater.coord[0]

            n = rng.poisson(rocks_per_crater)
            if n == 0:
                continue

            r_min = rim_inner_ratio * cr
            r_max = rim_outer_ratio * cr

            # Inverse CDF for power-law radial distribution
            u = rng.uniform(0, 1, n)
            p = 1.0 - decay_power
            if abs(p) < 1e-6:
                r = r_min * np.exp(u * np.log(r_max / r_min))
            else:
                r = (r_min**p + u * (r_max**p - r_min**p)) ** (1.0 / p)

            theta = rng.uniform(0, 2 * np.pi, n)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)

            valid = (x >= 0) & (x < tw) & (y >= 0) & (y < th)
            x, y = x[valid], y[valid]
            if len(x) == 0:
                continue

            # Size gradient: closer to rim -> larger max diameter
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            dist_ratio = np.clip((dist - r_min) / (r_max - r_min + 1e-6), 0, 1)
            local_d_max = np.clip(d_max * (1.0 - 0.6 * dist_ratio), d_min, d_max)
            # Vectorised power-law sampling with per-rock d_max
            u = rng.uniform(0, 1, len(local_d_max))
            exponent = 1.0 - alpha
            diam = (d_min**exponent + u * (local_d_max**exponent - d_min**exponent)) ** (1.0 / exponent)

            all_pos.append(np.column_stack([x, y]))
            all_diam.append(diam)

        if all_pos:
            return RockPlacement(np.vstack(all_pos), np.concatenate(all_diam))
        return RockPlacement(np.zeros((0, 2)), np.zeros(0))

    @staticmethod
    def crater_wall_debris(
        rng: np.random.Generator,
        terrain_width: float,
        terrain_height: float,
        resolution: float,
        craters: List[CraterData],
        rocks_per_crater: int = 20,
        inner_radius_ratio: float = 0.1,
        outer_radius_ratio: float = 0.9,
        wall_bias: float = 2.0,
        d_min: float = 1.0,
        d_max: float = 5.0,
        alpha: float = 2.5,
        min_crater_radius: float = 1.0,
    ) -> RockPlacement:
        """
        Place collapsed wall debris inside craters.

        Rocks are biased toward the wall base (where debris accumulates).
        Rocks near the wall are larger (block falls); toward centre, smaller.
        """
        tw, th = terrain_width, terrain_height
        all_pos, all_diam = [], []

        eligible = [
            (c, c.size * resolution / 2)
            for c in craters
            if c.size * resolution / 2 >= min_crater_radius
        ]

        for crater, cr in eligible:
            cx = crater.coord[1]
            cy = th - crater.coord[0]

            n = rng.poisson(rocks_per_crater)
            if n == 0:
                continue

            r_min = inner_radius_ratio * cr
            r_max = outer_radius_ratio * cr

            # Radial distribution biased toward wall: r ~ U^(1/wall_bias)
            u = rng.uniform(0, 1, n)
            r = r_min + (r_max - r_min) * (u ** (1.0 / wall_bias))

            theta = rng.uniform(0, 2 * np.pi, n)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)

            valid = (x >= 0) & (x < tw) & (y >= 0) & (y < th)
            x, y = x[valid], y[valid]
            if len(x) == 0:
                continue

            # Size gradient: larger near wall, smaller toward centre
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            wall_proximity = np.clip((dist - r_min) / (r_max - r_min + 1e-6), 0, 1)
            local_d_max = np.clip(
                d_min + (d_max - d_min) * wall_proximity, d_min, d_max
            )
            # Vectorised power-law sampling with per-rock d_max
            u = rng.uniform(0, 1, len(local_d_max))
            exponent = 1.0 - alpha
            diam = (d_min**exponent + u * (local_d_max**exponent - d_min**exponent)) ** (1.0 / exponent)

            all_pos.append(np.column_stack([x, y]))
            all_diam.append(diam)

        if all_pos:
            return RockPlacement(np.vstack(all_pos), np.concatenate(all_diam))
        return RockPlacement(np.zeros((0, 2)), np.zeros(0))

    @staticmethod
    def background_scatter(
        rng: np.random.Generator,
        terrain_width: float,
        terrain_height: float,
        resolution: float,
        mask: Optional[np.ndarray] = None,
        density: float = 0.5,
        d_min: float = 0.02,
        d_max: float = 0.5,
        alpha: float = 3.0,
    ) -> RockPlacement:
        """
        Uniform random scatter with power-law sizes, respecting terrain mask.

        Args:
            rng: Random generator.
            terrain_width: Width in meters.
            terrain_height: Height in meters.
            resolution: DEM resolution (m/pixel).
            mask: Optional terrain validity mask.
            density: Rocks per square meter.
            d_min: Minimum diameter scale.
            d_max: Maximum diameter scale.
            alpha: Power-law exponent.
        """
        tw, th = terrain_width, terrain_height
        n = rng.poisson(density * tw * th)
        if n == 0:
            return RockPlacement(np.zeros((0, 2)), np.zeros(0))

        x = rng.uniform(0, tw, n)
        y = rng.uniform(0, th, n)

        # Apply mask filter
        if mask is not None:
            h, w = mask.shape
            xi = np.clip((x / resolution).astype(int), 0, w - 1)
            yi = np.clip(h - 1 - (y / resolution).astype(int), 0, h - 1)
            valid = mask[yi, xi] > 0
            x, y = x[valid], y[valid]

        pos = np.column_stack([x, y])
        diam = RockDistributor.sample_power_law(rng, d_min, d_max, alpha, len(pos))
        return RockPlacement(pos, diam)

    @staticmethod
    def clustered_scatter(
        rng: np.random.Generator,
        terrain_width: float,
        terrain_height: float,
        resolution: float,
        mask: Optional[np.ndarray] = None,
        lambda_parent: float = 0.3,
        lambda_daughter: int = 5,
        sigma: float = 8.0,
        d_min: float = 0.05,
        d_max: float = 1.0,
        alpha: float = 2.8,
    ) -> RockPlacement:
        """
        Thomas cluster process with power-law sizes and mask filtering.

        Parent points are scattered uniformly, then each spawns daughter
        points with Gaussian offset.
        """
        tw, th = terrain_width, terrain_height
        n_parents = rng.poisson(lambda_parent * tw * th)
        if n_parents == 0:
            return RockPlacement(np.zeros((0, 2)), np.zeros(0))

        px = rng.uniform(0, tw, n_parents)
        py = rng.uniform(0, th, n_parents)

        all_x, all_y = [], []
        for ppx, ppy in zip(px, py):
            nd = rng.poisson(lambda_daughter)
            all_x.append(ppx + rng.normal(0, sigma, nd))
            all_y.append(ppy + rng.normal(0, sigma, nd))

        if not all_x:
            return RockPlacement(np.zeros((0, 2)), np.zeros(0))

        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        valid = (x >= 0) & (x < tw) & (y >= 0) & (y < th)
        x, y = x[valid], y[valid]

        # Mask filter
        if mask is not None:
            h, w = mask.shape
            xi = np.clip((x / resolution).astype(int), 0, w - 1)
            yi = np.clip(h - 1 - (y / resolution).astype(int), 0, h - 1)
            mask_valid = mask[yi, xi] > 0
            x, y = x[mask_valid], y[mask_valid]

        pos = np.column_stack([x, y])
        diam = RockDistributor.sample_power_law(rng, d_min, d_max, alpha, len(pos))
        return RockPlacement(pos, diam)

    # -- Overlap rejection --

    @staticmethod
    def remove_overlaps(
        placement: RockPlacement,
        min_gap: float = 0.0,
        native_rock_size: float = 0.3,
    ) -> RockPlacement:
        """
        Remove overlapping rocks, preserving larger ones first.

        Uses KDTree for efficient spatial queries.

        Args:
            placement: Input rock positions and diameters.
            min_gap: Minimum gap between rock edges (meters).
            native_rock_size: Approximate native rock model size (meters).

        Returns:
            Filtered RockPlacement with overlaps removed.
        """
        positions = placement.positions
        diameters = placement.diameters

        if len(positions) <= 1:
            return placement

        actual_radii = diameters * native_rock_size / 2
        order = np.argsort(-actual_radii)  # largest first

        # Build KDTree once with all positions, then greedily accept/reject
        all_tree = KDTree(positions)
        max_rad = actual_radii.max()
        max_search = 2 * max_rad + min_gap

        accepted = np.zeros(len(positions), dtype=bool)
        accepted_radii = actual_radii.copy()

        for idx in order:
            pos = positions[idx]
            rad = actual_radii[idx]

            # Query only previously accepted neighbours within max possible distance
            neighbours = all_tree.query_ball_point(pos, rad + max_rad + min_gap)
            conflict = False
            for ni in neighbours:
                if ni != idx and accepted[ni]:
                    d = np.linalg.norm(pos - positions[ni])
                    if d < rad + accepted_radii[ni] + min_gap:
                        conflict = True
                        break
            if not conflict:
                accepted[idx] = True

        kept = np.where(accepted)[0]
        n_before = len(positions)
        logger.debug(
            "overlap rejection: %d -> %d rocks (min_gap=%.3f)",
            n_before, len(kept), min_gap,
        )
        return RockPlacement(positions[kept], diameters[kept])
