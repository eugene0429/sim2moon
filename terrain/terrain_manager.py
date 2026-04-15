"""
Unified terrain manager.

Provides a single interface for all terrain operations:
- Procedural terrain generation (craters + base noise)
- Pre-generated DEM loading
- DEM-to-mesh rendering
- Collision mesh management
- Terrain deformation from wheel contact
- Point queries (height, normal)

This is the primary interface that other agents/modules interact with.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from terrain.config import TerrainManagerConf
from terrain.mesh.terrain_mesh import TerrainMeshBuilder
from terrain.materials.terrain_materials import TerrainMaterialManager
from terrain.procedural.crater_generator import CraterData
from terrain.procedural.moonyard_generator import MoonyardGenerator


class TerrainManager:
    """
    Unified terrain management interface.

    Interface contract for other agents:
        generate_terrain(seed) -> None
        get_dem() -> np.ndarray
        get_mask() -> np.ndarray
        get_craters_data() -> List[CraterData]
        render_mesh() -> None
        update_collider() -> None
        deform(positions, orientations, forces) -> None
        get_height_at(x, y) -> float
        get_normal_at(x, y) -> np.ndarray
    """

    def __init__(self, cfg: TerrainManagerConf, assets_path: str = "", pxr_utils=None) -> None:
        """
        Args:
            cfg: Full terrain configuration.
            assets_path: Base path for asset files.
            pxr_utils: USD utility module (injected for testability).
        """
        self._cfg = cfg
        self._pxr_utils = pxr_utils
        self._dems_path = os.path.join(assets_path, cfg.dems_path) if assets_path else cfg.dems_path
        self._texture_path = cfg.texture_path
        self._augmentation = cfg.augmentation

        # Pixel dimensions
        self._sim_width_px = int(cfg.sim_width / cfg.resolution)
        self._sim_length_px = int(cfg.sim_length / cfg.resolution)
        self._resolution = cfg.resolution

        # Procedural generator
        self._generator = MoonyardGenerator(cfg.moon_yard)

        # Mesh builder
        self._mesh_builder = TerrainMeshBuilder(
            sim_width_px=self._sim_width_px,
            sim_length_px=self._sim_length_px,
            grid_size=cfg.resolution,
            root_path=cfg.root_path,
            mesh_position=cfg.mesh_position,
            mesh_orientation=cfg.mesh_orientation,
            mesh_scale=cfg.mesh_scale,
            pxr_utils=pxr_utils,
        )

        # Material manager
        self._material_mgr = TerrainMaterialManager(pxr_utils=pxr_utils)

        # State
        self._dem: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._craters_data: Optional[List[CraterData]] = None
        self._pre_generated_dems: Dict[str, Tuple[str, Optional[str]]] = {}

        # Index pre-generated DEMs
        self._fetch_pre_generated_dems()

    def initialize_stage(self, stage=None) -> None:
        """Initialize USD stage for mesh builder and material manager."""
        self._mesh_builder.initialize_stage(stage)
        self._material_mgr.initialize(stage)

    # ------------------------------------------------------------------ #
    # Pre-generated DEM management
    # ------------------------------------------------------------------ #

    def _fetch_pre_generated_dems(self) -> None:
        """Index available pre-generated DEM folders."""
        if not os.path.isdir(self._dems_path):
            return

        for name in os.listdir(self._dems_path):
            folder = os.path.join(self._dems_path, name)
            if not os.path.isdir(folder):
                continue
            dem_file = os.path.join(folder, "dem.npy")
            mask_file = os.path.join(folder, "mask.npy")
            if not os.path.isfile(dem_file):
                warnings.warn(f"dem.npy not found in {name}, skipping")
                continue
            mask_path = mask_file if os.path.isfile(mask_file) else None
            self._pre_generated_dems[name] = (dem_file, mask_path)

    def load_dem_by_name(self, name: str) -> None:
        """
        Load a pre-generated DEM by folder name.

        Args:
            name: Folder name in the DEMs directory.
        """
        if name not in self._pre_generated_dems:
            raise KeyError(f"DEM '{name}' not found. Available: {list(self._pre_generated_dems.keys())}")

        dem_path, mask_path = self._pre_generated_dems[name]
        raw_dem = np.load(dem_path)
        raw_mask = np.load(mask_path) if mask_path else np.ones_like(raw_dem, dtype=bool)

        self._dem = np.zeros((self._sim_length_px, self._sim_width_px), dtype=np.float32)
        self._mask = np.zeros((self._sim_length_px, self._sim_width_px), dtype=np.float32)
        self._dem[:raw_dem.shape[0], :raw_dem.shape[1]] = raw_dem[:self._sim_length_px, :self._sim_width_px]
        self._mask[:raw_mask.shape[0], :raw_mask.shape[1]] = raw_mask[:self._sim_length_px, :self._sim_width_px]

        if self._augmentation:
            self._dem, self._mask, self._craters_data = self._generator.augment(self._dem, self._mask)
        else:
            self._generator.register_terrain(self._dem, self._mask)
        self._apply_tilt()

    def load_dem_by_index(self, idx: int) -> None:
        """Load a pre-generated DEM by index."""
        names = list(self._pre_generated_dems.keys())
        if idx >= len(names):
            raise IndexError(f"DEM index {idx} out of range (max {len(names) - 1})")
        self.load_dem_by_name(names[idx])

    # ------------------------------------------------------------------ #
    # Tilt / slope
    # ------------------------------------------------------------------ #

    def _apply_tilt(self) -> None:
        """Add a linear slope gradient to the DEM based on tilt config.

        The gradient is computed from ``tilt_angle`` (degrees from horizontal)
        and ``tilt_direction`` (compass bearing in degrees, 0 = +Y, 90 = +X).
        The slope is highest in the tilt direction and the gradient is centred
        so the middle of the terrain stays at its original height.
        """
        angle_deg = self._cfg.tilt_angle
        if angle_deg == 0.0 or self._dem is None:
            return

        direction_rad = np.deg2rad(self._cfg.tilt_direction)
        slope = np.tan(np.deg2rad(angle_deg))

        rows, cols = self._dem.shape
        # Pixel coordinates centred on the middle of the DEM
        cx = (cols - 1) / 2.0
        cy = (rows - 1) / 2.0
        x_idx = np.arange(cols) - cx  # pixel offsets
        y_idx = np.arange(rows) - cy

        # World-space offsets (metres)
        x_m = x_idx * self._resolution
        y_m = y_idx * self._resolution

        # Gradient components along X and Y
        gx = np.sin(direction_rad) * slope
        gy = np.cos(direction_rad) * slope

        # 2-D ramp via outer product: ramp[r, c] = gy*y_m[r] + gx*x_m[c]
        ramp = gy * y_m[:, np.newaxis] + gx * x_m[np.newaxis, :]

        self._dem = self._dem + ramp.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Procedural generation
    # ------------------------------------------------------------------ #

    def generate_terrain(self, seed: Optional[int] = None) -> None:
        """
        Generate a fully procedural terrain with craters.

        Args:
            seed: If provided, override the RNG seed for this generation.
        """
        if seed is not None:
            self._generator = MoonyardGenerator(self._cfg.moon_yard, seed=seed)
        self._dem, self._mask, self._craters_data = self._generator.randomize()
        self._apply_tilt()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def render_mesh(self, update_collider: bool = False) -> None:
        """
        Update the USD mesh from the current DEM.

        Args:
            update_collider: If True, rebuild the collision mesh and re-apply material.
        """
        if self._dem is None:
            raise RuntimeError("No DEM loaded. Call generate_terrain() or load_dem_by_name() first.")

        self._mesh_builder.update_heights(self._dem)

        if update_collider:
            self._mesh_builder.render(update_topology=True, new_prim=True)
            self._mesh_builder.update_collider()
            self._mesh_builder.apply_semantic_label("ground")
            self._mesh_builder.apply_material(self._texture_path)
        else:
            self._mesh_builder.render()

    def update_collider(self) -> None:
        """Rebuild the collision mesh for the current terrain."""
        self._mesh_builder.update_collider()

    def randomize_terrain(self) -> None:
        """Generate new procedural terrain and render with collider update."""
        self.generate_terrain()
        self.render_mesh(update_collider=True)

    def load_terrain_by_name(self, name: str) -> None:
        """Load a pre-generated DEM and render with collider update."""
        self.load_dem_by_name(name)
        self.render_mesh(update_collider=True)

    def load_terrain_by_index(self, idx: int) -> None:
        """Load a pre-generated DEM by index and render with collider update."""
        self.load_dem_by_index(idx)
        self.render_mesh(update_collider=True)

    # ------------------------------------------------------------------ #
    # Deformation
    # ------------------------------------------------------------------ #

    def deform(
        self,
        world_positions: np.ndarray,
        world_orientations: np.ndarray,
        contact_forces: np.ndarray,
    ) -> None:
        """
        Apply wheel-terrain deformation and update mesh.

        Args:
            world_positions: Wheel world positions [N, 3].
            world_orientations: Wheel world orientations [N, 4] (quaternion).
            contact_forces: Contact forces per wheel [N, 3].
        """
        self._dem, self._mask = self._generator.deform(
            world_positions, world_orientations, contact_forces
        )
        self.render_mesh(update_collider=False)

    # ------------------------------------------------------------------ #
    # Point queries
    # ------------------------------------------------------------------ #

    def get_height_at(self, x: float, y: float) -> float:
        """
        Query terrain height at a world (x, y) position.

        Args:
            x: World X coordinate in meters.
            y: World Y coordinate in meters.

        Returns:
            Height in meters at the queried position.
        """
        if self._dem is None:
            return 0.0
        px = int(x / self._resolution)
        py = int(y / self._resolution)
        h, w = self._dem.shape
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        return float(self._dem[py, px])

    def get_normal_at(self, x: float, y: float) -> np.ndarray:
        """
        Estimate terrain surface normal at a world (x, y) position.

        Uses finite differences on neighboring DEM pixels.

        Args:
            x: World X coordinate in meters.
            y: World Y coordinate in meters.

        Returns:
            Unit normal vector [3] (pointing up).
        """
        if self._dem is None:
            return np.array([0.0, 0.0, 1.0])

        px = int(x / self._resolution)
        py = int(y / self._resolution)
        h, w = self._dem.shape
        px = max(1, min(px, w - 2))
        py = max(1, min(py, h - 2))

        # Central differences
        dz_dx = (self._dem[py, px + 1] - self._dem[py, px - 1]) / (2 * self._resolution)
        dz_dy = (self._dem[py + 1, px] - self._dem[py - 1, px]) / (2 * self._resolution)

        normal = np.array([-dz_dx, -dz_dy, 1.0])
        return normal / np.linalg.norm(normal)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_dem(self) -> Optional[np.ndarray]:
        """Return the current DEM height map."""
        return self._dem

    def get_mask(self) -> Optional[np.ndarray]:
        """Return the current valid-area mask."""
        return self._mask

    def get_craters_data(self) -> Optional[List[CraterData]]:
        """Return crater metadata from the last generation (for rock placement)."""
        return self._craters_data

    @property
    def resolution(self) -> float:
        """Terrain resolution in meters per pixel."""
        return self._resolution

    @property
    def available_dems(self) -> List[str]:
        """List of available pre-generated DEM names."""
        return list(self._pre_generated_dems.keys())
