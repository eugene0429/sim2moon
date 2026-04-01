"""
Moonyard generator combining base terrain, crater distribution, and craters.

This is the top-level procedural terrain pipeline that produces a complete
lunar surface DEM with craters and supports terrain deformation.
"""

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from terrain.config import MoonYardConf
from terrain.deformation.deformation_engine import DeformationEngine
from terrain.procedural.base_terrain import BaseTerrainGenerator
from terrain.procedural.crater_distribution import CraterDistributor
from terrain.procedural.crater_generator import CraterData, CraterGenerator
from terrain.procedural.realistic_crater_generator import RealisticCraterGenerator


class MoonyardGenerator:
    """
    Full procedural moonyard terrain pipeline.

    Combines:
    1. Base terrain generation (interpolated noise)
    2. Crater distribution (hardcore Poisson)
    3. Crater shape generation (spline profiles)
    4. Terrain deformation (wheel sinkage)
    """

    def __init__(self, cfg: MoonYardConf, seed: Optional[int] = None) -> None:
        if seed is not None:
            cfg = self._override_seeds(cfg, seed)
        self._base_gen = BaseTerrainGenerator(cfg.base_terrain_generator)
        self._distributor = CraterDistributor(cfg.crater_distribution)
        if cfg.crater_generator.crater_mode == "realistic":
            self._crater_gen = RealisticCraterGenerator(
                cfg.crater_generator, cfg.realistic_crater
            )
        else:
            self._crater_gen = CraterGenerator(cfg.crater_generator)
        self._deform_engine = DeformationEngine(cfg.deformation_engine)

        self._is_lab = cfg.is_lab
        self._is_yard = cfg.is_yard
        self._cfg = cfg

        # State
        self._dem_init: Optional[np.ndarray] = None
        self._dem_delta: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._num_pass: Optional[np.ndarray] = None

    def randomize(self) -> Tuple[np.ndarray, np.ndarray, List[CraterData]]:
        """
        Generate a fully randomized terrain with craters.

        Returns:
            Tuple of (DEM, mask, list of CraterData).
        """
        dem = self._base_gen.generate(is_lab=self._is_lab, is_yard=self._is_yard)
        coords, radii = self._distributor.run()
        dem, mask, craters_data = self._crater_gen.generate_craters(dem, coords, radii)

        self._dem_init = dem
        self._dem_delta = np.zeros_like(dem)
        self._mask = mask
        self._num_pass = np.zeros_like(mask)

        return dem, mask, craters_data

    def augment(
        self, dem: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[CraterData]]:
        """
        Add craters to an existing DEM (e.g., a pre-generated or loaded DEM).

        Returns:
            Tuple of (modified DEM, combined mask, list of CraterData).
        """
        coords, radii = self._distributor.run()
        dem, new_mask, craters_data = self._crater_gen.generate_craters(dem, coords, radii)
        mask = mask * new_mask

        self._dem_init = dem
        self._dem_delta = np.zeros_like(dem)
        self._mask = mask
        self._num_pass = np.zeros_like(mask)

        return dem, mask, craters_data

    def register_terrain(self, dem: np.ndarray, mask: np.ndarray) -> None:
        """Register an externally loaded DEM/mask for deformation tracking."""
        self._dem_init = dem
        self._dem_delta = np.zeros_like(dem)
        self._mask = mask
        self._num_pass = np.zeros_like(mask)

    def deform(
        self,
        world_positions: np.ndarray,
        world_orientations: np.ndarray,
        contact_forces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply wheel-terrain deformation to the DEM.

        Args:
            world_positions: World positions of robot links [N, 3].
            world_orientations: World orientations of robot links [N, 4] (quaternion).
            contact_forces: Contact forces on robot links [N, 3].

        Returns:
            Tuple of (deformed DEM, mask).
        """
        self._dem_delta, self._num_pass = self._deform_engine.deform(
            self._dem_delta,
            self._num_pass,
            world_positions,
            world_orientations,
            contact_forces[:, 2],
        )
        return self._dem_init + self._dem_delta, self._mask

    @staticmethod
    def _override_seeds(cfg: MoonYardConf, seed: int) -> MoonYardConf:
        """Return a copy of cfg with all sub-generator seeds replaced."""
        return dataclasses.replace(
            cfg,
            base_terrain_generator=dataclasses.replace(cfg.base_terrain_generator, seed=seed),
            crater_distribution=dataclasses.replace(cfg.crater_distribution, seed=seed),
            crater_generator=dataclasses.replace(cfg.crater_generator, seed=seed),
        )
