"""
Crater distribution using a hardcore Poisson process.

Places craters of multiple size classes across the terrain, ensuring
that larger craters take priority and overlap is prevented via
hardcore rejection sampling.
"""

from typing import List, Optional, Tuple

import numpy as np

from terrain.config import CraterDistributionConf


class CraterDistributor:
    """Distributes craters on a DEM using a hardcore Poisson process."""

    def __init__(self, cfg: CraterDistributionConf) -> None:
        self._x_max = cfg.x_size
        self._y_max = cfg.y_size
        self._area = self._x_max * self._y_max
        self._densities = cfg.densities
        self._radius_ranges = cfg.radius
        self._num_repeat = cfg.num_repeat
        self._rng = np.random.default_rng(cfg.seed)

    def _sample_poisson(
        self, density: float, r_minmax: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample crater positions from a Poisson process."""
        num_points = self._rng.poisson(self._area * density)
        radii = self._rng.uniform(r_minmax[0], r_minmax[1], num_points)
        x = self._rng.uniform(0, self._x_max, num_points)
        y = self._rng.uniform(0, self._y_max, num_points)
        return np.stack([x, y]).T, radii

    def _hardcore_rejection(
        self, coords: np.ndarray, radii: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove overlapping craters via age-based hardcore rejection."""
        n = coords.shape[0]
        if n == 0:
            return coords, radii

        from scipy.spatial.distance import cdist
        dist_matrix = cdist(coords, coords)  # (n, n)

        ages = self._rng.uniform(0, 1, n)
        # For each pair (i, j), i overlaps j if dist < radii[i] and i != j
        np.fill_diagonal(dist_matrix, np.inf)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            in_disk = dist_matrix[i] < radii[i]
            # Among overlapping neighbours, only keep if youngest
            if np.any(in_disk):
                younger = in_disk & (ages < ages[i])
                if np.any(younger):
                    keep[i] = False

        return coords[keep], radii[keep]

    def _check_previous(
        self,
        new_coords: np.ndarray,
        new_radii: np.ndarray,
        prev: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reject new craters that overlap with previously placed ones."""
        if prev is None or new_coords.shape[0] == 0:
            return new_coords, new_radii

        prev_coords, prev_radii = prev
        if prev_coords.shape[0] == 0:
            return new_coords, new_radii

        from scipy.spatial.distance import cdist
        # (n_prev, n_new) distance matrix
        dist_matrix = cdist(prev_coords, new_coords)
        # For each prev crater i, mark new craters within its radius
        overlap = dist_matrix < prev_radii[:, None]
        keep = ~np.any(overlap, axis=0)

        return new_coords[keep], new_radii[keep]

    def _simulate_hc_poisson(
        self,
        density: float,
        r_minmax: List[float],
        prev: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one hardcore Poisson process for a single size class."""
        coords, radii = self._sample_poisson(density, r_minmax)

        for _ in range(self._num_repeat):
            coords, radii = self._hardcore_rejection(coords, radii)
            new_c, new_r = self._sample_poisson(density, r_minmax)
            coords = np.concatenate([coords, new_c])
            radii = np.concatenate([radii, new_r])
            coords, radii = self._check_previous(coords, radii, prev)

        coords, radii = self._hardcore_rejection(coords, radii)
        coords, radii = self._check_previous(coords, radii, prev)
        return coords, radii

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the full crater distribution pipeline.

        Processes size classes from largest to smallest, ensuring
        that smaller craters do not overlap with larger ones.

        Returns:
            Tuple of (coords [N, 2] in meters, radii [N] in meters).
        """
        prev = None
        for density, r_minmax in zip(self._densities, self._radius_ranges):
            new_coords, new_radii = self._simulate_hc_poisson(density, r_minmax, prev)
            if prev is not None:
                prev = (
                    np.concatenate([prev[0], new_coords], axis=0),
                    np.concatenate([prev[1], new_radii], axis=0),
                )
            else:
                prev = (new_coords, new_radii)
        return prev
