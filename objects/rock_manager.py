"""
Rock manager for lunar simulation.

Manages rock asset instancing via USD PointInstancer.
Uses realistic distribution (power-law + crater-aware) via RockDistributor.

Reference: OmniLRS src/environments/rock_manager.py
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from objects.config import RockGroupConf, RockManagerConf
from objects.rock_distribution import CraterData, RockDistributor, RockPlacement

logger = logging.getLogger(__name__)

try:
    import omni.usd
    from pxr import Sdf, UsdGeom
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


class PointInstancer:
    """
    USD PointInstancer wrapper for efficient rock rendering.

    Creates a PointInstancer prim with cached prototypes and sets
    instance positions, orientations, and scales.
    """

    def __init__(
        self,
        instancer_path: str,
        asset_list: List[str],
        seed: int = 42,
    ) -> None:
        self._path = instancer_path
        self._prototypes = asset_list
        self._rng = np.random.default_rng(seed=seed)
        self._instancer = None

        if _HAS_USD:
            stage = omni.usd.get_context().get_stage()
            self._instancer = UsdGeom.PointInstancer.Define(stage, self._path)
            # Cache prototypes
            proto_container = stage.DefinePrim(
                os.path.join(self._path, "Prototypes"), "Scope"
            )
            proto_targets = []
            for i, asset_path in enumerate(self._prototypes):
                proto_prim_path = os.path.join(
                    self._path, "Prototypes", f"proto_{i}"
                )
                proto_prim = stage.DefinePrim(proto_prim_path, "Xform")
                ref = proto_prim.GetReferences()
                ref.AddReference(asset_path)
                proto_targets.append(proto_prim.GetPath())
            rel = self._instancer.GetPrototypesRel()
            rel.SetTargets(proto_targets)

    def set_instances(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        """
        Set instance transforms.

        Args:
            position: (N, 3) positions.
            orientation: (N, 4) quaternions (x, y, z, w).
            scale: (N, 3) scales.
        """
        if not _HAS_USD or self._instancer is None:
            logger.warning("PointInstancer: USD not available")
            return

        n = position.shape[0]
        ids = self._rng.integers(0, len(self._prototypes), n)

        from pxr import Gf, Vt

        self._instancer.GetProtoIndicesAttr().Set(ids.tolist())
        self._instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in position])
        )
        self._instancer.GetOrientationsAttr().Set(
            Vt.QuathArray([Gf.Quath(float(q[3]), float(q[0]), float(q[1]), float(q[2])) for q in orientation])
        )
        self._instancer.GetScalesAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(s[0]), float(s[1]), float(s[2])) for s in scale])
        )


class RockManager:
    """
    Manages rock placement in the simulation environment.

    Uses realistic distribution: power-law sizes with crater-aware placement
    (configured via realistic_distribution in RockGroupConf).

    Interface contract (exported to other agents):
        build(dem, mask) -> None
        set_craters_data(craters) -> None
        randomize() -> None
        set_visible(flag) -> None
    """

    def __init__(self, cfg: RockManagerConf) -> None:
        self._cfg = cfg
        self._instancers: Dict[str, PointInstancer] = {}
        self._craters_data: List[CraterData] = []
        self._dem: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._mesh_position: tuple = (0.0, 0.0, 0.0)
        self._terrain_resolution: float = 0.0  # actual DEM m/px
        self._distributor = RockDistributor()

    def set_terrain_info(self, mesh_position, terrain_resolution: float) -> None:
        """Set terrain geometry info for correct rock world-space placement.

        Args:
            mesh_position: (x, y, z) terrain mesh offset in world space.
            terrain_resolution: Terrain DEM resolution in meters/pixel.
        """
        mp = [float(v) for v in mesh_position]
        self._mesh_position = (mp[0], mp[1], mp[2])
        self._terrain_resolution = float(terrain_resolution)
        logger.info("RockManager: mesh_position=(%.2f, %.2f, %.2f), terrain_res=%.4f",
                     mp[0], mp[1], mp[2], self._terrain_resolution)

    def build(self, dem: np.ndarray, mask: np.ndarray) -> None:
        """
        Build rock instancers and prepare for placement.

        Args:
            dem: Height map array.
            mask: Valid area mask array.
        """
        if not self._cfg.enable:
            return

        self._dem = dem
        self._mask = mask
        self._create_instancers()
        logger.info("RockManager: built with %d groups", len(self._cfg.rocks_settings))

    def set_craters_data(self, craters: List[CraterData]) -> None:
        """
        Receive crater data from TerrainManager for realistic rock placement.

        Args:
            craters: List of CraterData objects from terrain generation.
        """
        self._craters_data = craters
        logger.info("RockManager: received %d craters", len(craters))

    def randomize(self) -> None:
        """Generate and place rocks according to each group's distribution config."""
        if not self._cfg.enable:
            return

        for name, group_cfg in self._cfg.rocks_settings.items():
            self._generate_realistic(name, group_cfg)

    def set_visible(self, flag: bool) -> None:
        """Set visibility of all rock instancers."""
        if not (_HAS_USD and self._cfg.enable):
            return

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._cfg.instancers_path)
        if prim:
            prim.GetAttribute("visibility").Set(
                "visible" if flag else "invisible"
            )

    # -- Internal --

    def _create_instancers(self) -> None:
        """Create USD instancers for each rock group."""
        if not _HAS_USD:
            return

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.DefinePrim(self._cfg.instancers_path, "Xform")

        # Apply terrain mesh_position as a USD transform on the rock root prim
        # so all rock instances are automatically offset to match the terrain.
        mp = self._mesh_position
        if mp[0] != 0.0 or mp[1] != 0.0 or mp[2] != 0.0:
            from pxr import UsdGeom, Gf
            xformable = UsdGeom.Xformable(root_prim)
            xformable.AddTranslateOp().Set(
                Gf.Vec3d(mp[0], mp[1], mp[2])
            )
            logger.info("RockManager: instancer root offset set to (%.2f, %.2f, %.2f)",
                         mp[0], mp[1], mp[2])

        for name, group_cfg in self._cfg.rocks_settings.items():
            rock_assets = self._collect_assets(group_cfg.collections)
            if not rock_assets:
                logger.warning("RockManager: no assets found for group '%s'", name)
                continue

            group_path = os.path.join(self._cfg.instancers_path, name)
            stage.DefinePrim(group_path, "Xform")

            instancer_path = os.path.join(group_path, "instancer")
            self._instancers[name] = PointInstancer(
                instancer_path, rock_assets, seed=group_cfg.seed,
            )

    def _collect_assets(self, collections: List[str]) -> List[str]:
        """Collect USD asset paths from named collections."""
        assets = []
        for collection in collections:
            root = os.path.join(self._cfg.assets_root, collection)
            if not os.path.isdir(root):
                logger.warning("RockManager: collection dir not found: %s", root)
                continue
            for entry in os.listdir(root):
                entry_path = os.path.join(root, entry)
                if os.path.isdir(entry_path):
                    for f in os.listdir(entry_path):
                        if f.endswith(".usd") or f.endswith(".usdz"):
                            assets.append(os.path.join(entry_path, f))
                elif entry.endswith(".usd") or entry.endswith(".usdz"):
                    assets.append(entry_path)
        return assets

    def _generate_realistic(self, name: str, group_cfg: RockGroupConf) -> None:
        """Generate rocks using realistic distribution."""
        if self._dem is None:
            logger.warning("RockManager: DEM not set, skipping %s", name)
            return

        dist_cfg = group_cfg.realistic_distribution
        rng = np.random.default_rng(seed=group_cfg.seed)
        resolution = dist_cfg.resolution
        terrain_h, terrain_w = self._dem.shape
        # Use terrain DEM resolution for actual world dimensions so that
        # rock distribution covers the full terrain, not a compressed subset.
        terrain_res = self._terrain_resolution if self._terrain_resolution > 0 else resolution
        tw = terrain_w * terrain_res
        th = terrain_h * terrain_res

        # Select distribution method
        # Pass terrain_res so crater size conversion and mask indexing use the
        # correct DEM resolution (crater sizes are stored in DEM pixels).
        method = dist_cfg.method
        if method == "crater_ejecta":
            placement = RockDistributor.crater_ejecta(
                rng, tw, th, terrain_res, self._craters_data,
                rocks_per_crater=dist_cfg.rocks_per_crater,
                rim_inner_ratio=dist_cfg.rim_inner_ratio,
                rim_outer_ratio=dist_cfg.rim_outer_ratio,
                decay_power=dist_cfg.decay_power,
                d_min=dist_cfg.d_min, d_max=dist_cfg.d_max,
                alpha=dist_cfg.alpha,
                min_crater_radius=dist_cfg.min_crater_radius,
            )
        elif method == "crater_wall_debris":
            placement = RockDistributor.crater_wall_debris(
                rng, tw, th, terrain_res, self._craters_data,
                rocks_per_crater=dist_cfg.rocks_per_crater,
                inner_radius_ratio=dist_cfg.inner_radius_ratio,
                outer_radius_ratio=dist_cfg.outer_radius_ratio,
                wall_bias=dist_cfg.wall_bias,
                d_min=dist_cfg.d_min, d_max=dist_cfg.d_max,
                alpha=dist_cfg.alpha,
                min_crater_radius=dist_cfg.min_crater_radius,
            )
        elif method == "background_scatter":
            placement = RockDistributor.background_scatter(
                rng, tw, th, terrain_res, self._mask,
                density=dist_cfg.density,
                d_min=dist_cfg.d_min, d_max=dist_cfg.d_max,
                alpha=dist_cfg.alpha,
            )
        elif method == "clustered_scatter":
            placement = RockDistributor.clustered_scatter(
                rng, tw, th, terrain_res, self._mask,
                lambda_parent=dist_cfg.lambda_parent,
                lambda_daughter=dist_cfg.lambda_daughter,
                sigma=dist_cfg.sigma,
                d_min=dist_cfg.d_min, d_max=dist_cfg.d_max,
                alpha=dist_cfg.alpha,
            )
        else:
            logger.error("RockManager: unknown method '%s' for group '%s'", method, name)
            return

        # Apply overlap rejection
        placement = RockDistributor.remove_overlaps(
            placement,
            min_gap=dist_cfg.min_gap,
            native_rock_size=dist_cfg.native_rock_size,
        )

        n = len(placement.positions)
        if n == 0:
            logger.info("RockManager: %s: 0 rocks generated", name)
            return

        # Build 3D positions with Z from DEM
        pos_3d = self._positions_to_3d(placement.positions)

        # Random yaw rotations
        yaws = rng.uniform(0, 2 * np.pi, n)
        orientations = np.zeros((n, 4), dtype=np.float32)
        orientations[:, 2] = np.sin(yaws / 2)  # qz
        orientations[:, 3] = np.cos(yaws / 2)  # qw

        # Uniform scale from diameters
        scales = np.column_stack([
            placement.diameters, placement.diameters, placement.diameters,
        ]).astype(np.float32)

        logger.info(
            "RockManager: %s: %d rocks (%s, d=%.2f~%.2f)",
            name, n, method,
            placement.diameters.min(), placement.diameters.max(),
        )

        if name in self._instancers:
            self._instancers[name].set_instances(pos_3d, orientations, scales)

    def _positions_to_3d(
        self, positions_2d: np.ndarray,
    ) -> np.ndarray:
        """Convert 2D world-space positions to 3D by sampling Z from DEM.

        Positions are in world meters (matching terrain mesh vertex coords).
        DEM lookup uses terrain_resolution to convert meters → pixel index.
        """
        dem = self._dem
        h, w = dem.shape
        terrain_res = self._terrain_resolution if self._terrain_resolution > 0 else 1.0

        xi = np.clip((positions_2d[:, 0] / terrain_res).astype(int), 0, w - 1)
        yi = np.clip((positions_2d[:, 1] / terrain_res).astype(int), 0, h - 1)
        yi_flipped = h - 1 - yi
        z = dem[yi_flipped, xi]

        pos_3d = np.zeros((len(positions_2d), 3), dtype=np.float32)
        pos_3d[:, 0] = positions_2d[:, 0]
        pos_3d[:, 1] = positions_2d[:, 1]
        pos_3d[:, 2] = z
        return pos_3d
