"""
LunarYard outdoor procedural environment.

Composes terrain, celestial, rendering, and objects modules into a
concrete BaseEnvironment that can be run by SimulationManager.
"""

import logging
import os
from typing import Optional

import numpy as np

from core.enums import SimulatorMode
from core.simulation_manager import register_environment
from environments.base_environment import BaseEnvironment
from environments.lunar_yard_config import LunarYardConf

logger = logging.getLogger(__name__)


class LunarYardEnvironment(BaseEnvironment):
    """Outdoor procedural lunar environment.

    Composes: TerrainManager, StellarEngine, SunController,
    EarthController, RockManager, Renderer, PostProcessingManager.
    """

    def __init__(self, stage, mode: SimulatorMode, cfg: dict = None) -> None:
        super().__init__(stage, mode, cfg)
        cfg = cfg or {}
        self._seed = cfg.get("seed", 42)

        # Extract sub-configs
        ly = cfg.get("lunaryard_settings")
        if isinstance(ly, LunarYardConf):
            self._lunaryard_conf = ly
        elif isinstance(ly, dict):
            self._lunaryard_conf = LunarYardConf(**ly)
        else:
            self._lunaryard_conf = LunarYardConf()

        self._root_path = self._lunaryard_conf.root_path

        # Sub-managers — created during build_scene/load
        self._terrain_manager = None
        self._stellar_engine = None
        self._sun_controller = None
        self._earth_controller = None
        self._rock_manager = None
        self._renderer = None
        self._post_processing = None
        self._starfield = None
        self._earthshine = None
        self._robot_manager = None

        # Terramechanics solver (Bekker-Janosi model)
        # Configured from YAML terramechanics_settings; disabled by default
        # (falls back to PhysX rigid-body friction).
        from physics.terramechanics import TerramechanicsSolver
        from physics.terramechanics_parameters import RobotParameter, TerrainMechanicalParameter
        import math

        tm_cfg = cfg.get("terramechanics_settings") or {}
        if isinstance(tm_cfg, dict):
            self._terramechanics_enabled = tm_cfg.get("enable", False)
        else:
            self._terramechanics_enabled = getattr(tm_cfg, "enable", False)
            tm_cfg = {k: getattr(tm_cfg, k) for k in dir(tm_cfg)
                      if not k.startswith("_")}

        robot_cfg = tm_cfg.get("robot", {})
        soil_cfg = tm_cfg.get("soil", {})

        self._robot_param = RobotParameter(
            mass=robot_cfg.get("mass", 20.0),
            num_wheels=robot_cfg.get("num_wheels", 4),
            wheel_radius=robot_cfg.get("wheel_radius", 0.09),
            wheel_width=robot_cfg.get("wheel_width", 0.1),
            wheel_lug_height=robot_cfg.get("wheel_lug_height", 0.02),
            wheel_lug_count=robot_cfg.get("wheel_lug_count", 16),
        )
        self._terrain_mech_param = TerrainMechanicalParameter(
            k_c=soil_cfg.get("k_c", 1400.0),
            k_phi=soil_cfg.get("k_phi", 820000.0),
            n=soil_cfg.get("n", 1.0),
            c=soil_cfg.get("c", 170.0),
            phi=math.radians(soil_cfg.get("phi_deg", 35.0)),
            K=soil_cfg.get("K", 0.018),
            rho=soil_cfg.get("rho", 1500.0),
            gravity=soil_cfg.get("gravity", 1.625),
            slip_sinkage_coeff=soil_cfg.get("slip_sinkage_coeff", 0.5),
            heterogeneity=soil_cfg.get("heterogeneity", 0.0),
            heterogeneity_scale=soil_cfg.get("heterogeneity_scale", 2.0),
            heterogeneity_seed=soil_cfg.get("heterogeneity_seed", 42),
            physx_mu=soil_cfg.get("physx_mu", 0.0),
        )
        self._terramechanics_solver = TerramechanicsSolver(
            robot_param=self._robot_param,
            terrain_param=self._terrain_mech_param,
        )
        # EMA filter for PhysX contact forces to reduce jitter in sinkage computation.
        # alpha = 0 means no filtering (raw PhysX), alpha close to 1 means heavy smoothing.
        self._contact_ema_alpha = float(tm_cfg.get("contact_force_ema", 0.7))
        self._contact_ema = {}  # per-robot EMA state: {robot_name: np.ndarray}
        # Robots listed here skip terramechanics (PhysX friction only).
        self._terramechanics_skip_robots = set(tm_cfg.get("skip_robots", []))
        if self._terramechanics_enabled:
            logger.info("Terramechanics enabled (Bekker-Janosi)")
        else:
            logger.info("Terramechanics disabled (PhysX friction only)")

    def build_scene(self) -> None:
        """Create USD scene structure and celestial controllers."""
        logger.info("Building LunarYard scene at %s", self._root_path)

        # Create root Xform
        if self._stage is not None:
            self._create_root_xform()

        # Load materials (MDL files for terrain texturing)
        if self._stage is not None:
            self._load_materials()

        # Stellar engine
        stellar_cfg = self._cfg.get("stellar_engine_settings")
        if stellar_cfg is not None:
            from celestial.config import StellarEngineConf

            sec = StellarEngineConf(**stellar_cfg) if isinstance(stellar_cfg, dict) else stellar_cfg
            if sec.enable:
                from celestial.stellar_engine import StellarEngine

                self._stellar_engine = StellarEngine(stellar_cfg)
                coords = self._lunaryard_conf.coordinates
                self._stellar_engine.set_coordinates(coords.latitude, coords.longitude)
                logger.info("StellarEngine initialized at lat=%.1f, lon=%.1f",
                            coords.latitude, coords.longitude)
            else:
                logger.info("StellarEngine disabled — sun stays at static azimuth/elevation")

        # Sun controller
        sun_cfg = self._cfg.get("sun_settings")
        if sun_cfg is not None:
            from celestial.sun_controller import SunController

            self._sun_controller = SunController(sun_cfg)
            self._sun_controller.build(self._root_path)
            logger.info("SunController built")

        # Earth controller
        earth_cfg = self._cfg.get("earth_settings")
        if earth_cfg is not None:
            from celestial.earth_controller import EarthController

            self._earth_controller = EarthController(earth_cfg)
            self._earth_controller.build(self._root_path)
            logger.info("EarthController built")

        # Renderer
        rendering_cfg = self._cfg.get("rendering")
        if rendering_cfg is not None:
            from rendering.renderer import Renderer

            renderer_sub = getattr(rendering_cfg, "renderer", rendering_cfg)
            self._renderer = Renderer(renderer_sub)
            self._renderer.apply()
            logger.info("Renderer applied")

        # Starfield
        if self._stage is not None:
            from effects.starfield import Starfield
            from effects.config import StarfieldConf

            starfield_cfg = self._cfg.get("starfield_settings")
            if isinstance(starfield_cfg, dict):
                sf_conf = StarfieldConf(**starfield_cfg)
            elif isinstance(starfield_cfg, StarfieldConf):
                sf_conf = starfield_cfg
            else:
                sf_conf = StarfieldConf()  # use defaults

            self._starfield = Starfield(sf_conf)
            self._starfield.generate()
            self._starfield.setup(self._stage)
            logger.info("Starfield created")

        # Earthshine
        if self._stage is not None:
            from effects.earthshine import Earthshine
            from effects.config import EarthshineConf

            earthshine_cfg = self._cfg.get("earthshine_settings")
            if isinstance(earthshine_cfg, dict):
                es_conf = EarthshineConf(**earthshine_cfg)
            elif isinstance(earthshine_cfg, EarthshineConf):
                es_conf = earthshine_cfg
            else:
                es_conf = EarthshineConf()  # use defaults

            self._earthshine = Earthshine(es_conf)
            self._earthshine.setup(self._stage)
            logger.info("Earthshine created")

    def load(self) -> None:
        """Generate terrain, render mesh, and place rocks."""
        logger.info("Loading LunarYard assets")

        # Terrain
        terrain_cfg = self._cfg.get("terrain_manager")
        if terrain_cfg is not None:
            from terrain.terrain_manager import TerrainManager
            import core.pxr_utils as pxr_utils

            self._terrain_manager = TerrainManager(terrain_cfg, pxr_utils=pxr_utils)
            self._terrain_manager.initialize_stage(self._stage)
            self._terrain_manager.generate_terrain(self._seed)
            self._terrain_manager.render_mesh(update_collider=True)
            logger.info("Terrain generated and rendered")

        # Landscape (outer terrain from SouthPole DEM)
        landscape_cfg = self._cfg.get("landscape_settings")
        if landscape_cfg is not None:
            from terrain.landscape_builder import LandscapeBuilder
            from terrain.config import LandscapeConf

            if isinstance(landscape_cfg, dict):
                ls_conf = LandscapeConf(**landscape_cfg)
            elif isinstance(landscape_cfg, LandscapeConf):
                ls_conf = landscape_cfg
            else:
                ls_conf = None

            if ls_conf and ls_conf.enable:
                import core.pxr_utils as pxr_utils
                from assets import get_assets_path

                # Resolve DEM path relative to assets
                if not os.path.isabs(ls_conf.dem_path):
                    ls_conf.dem_path = os.path.join(get_assets_path(), ls_conf.dem_path)

                tm_cfg = self._cfg.get("terrain_manager", {})
                main_size = (
                    tm_cfg.get("sim_length", 40.0) if isinstance(tm_cfg, dict)
                    else getattr(tm_cfg, "sim_length", 40.0),
                    tm_cfg.get("sim_width", 40.0) if isinstance(tm_cfg, dict)
                    else getattr(tm_cfg, "sim_width", 40.0),
                )
                main_pos = (
                    tm_cfg.get("mesh_position", [0, 0, 0]) if isinstance(tm_cfg, dict)
                    else getattr(tm_cfg, "mesh_position", (0, 0, 0))
                )

                landscape_builder = LandscapeBuilder(
                    ls_conf,
                    main_terrain_size=main_size,
                    main_terrain_position=main_pos,
                )
                texture = ls_conf.texture_path or self._cfg.get(
                    "terrain_manager", {}
                ).get("texture_path", "")
                landscape_builder.render(self._stage, pxr_utils, texture)
                logger.info("SouthPole landscape loaded")

                # Align main terrain Z to landscape hole boundary max height,
                # then build a curved transition mesh bridging the gap.
                if self._terrain_manager is not None:
                    main_dem = self._terrain_manager.get_dem()
                    main_res = (
                        tm_cfg.get("resolution", 0.02) if isinstance(tm_cfg, dict)
                        else getattr(tm_cfg, "resolution", 0.02)
                    )
                    if main_dem is not None:
                        z_off = landscape_builder.get_hole_edge_max_z()
                        # Shift actual terrain mesh Z to sit at hole boundary max.
                        # terrain_mesh and terrain_mesh_0 are siblings (not
                        # parent-child), so we must target the Mesh prim directly.
                        root_path = (
                            tm_cfg.get("root_path", "/LunarYard")
                            if isinstance(tm_cfg, dict)
                            else getattr(tm_cfg, "root_path", "/LunarYard")
                        )
                        from pxr import UsdGeom, Gf
                        terrain_parent = self._stage.GetPrimAtPath(
                            f"{root_path}/Terrain"
                        )
                        for child in terrain_parent.GetChildren():
                            if child.GetTypeName() == "Mesh":
                                child_xform = UsdGeom.Xform(child)
                                child_ops = child_xform.GetOrderedXformOps()
                                if child_ops:
                                    cur = child_ops[0].Get()
                                    px = float(cur[0]) if cur else 0.0
                                    py = float(cur[1]) if cur else 0.0
                                    child_ops[0].Set(
                                        Gf.Vec3d(px, py, float(z_off))
                                    )
                        logger.info(
                            "Main terrain Z offset set to %.2f m", z_off
                        )

                        landscape_builder.render_transition(
                            self._stage, pxr_utils,
                            main_dem, main_res, texture,
                            z_offset=z_off,
                        )
                        logger.info("Transition mesh loaded")

        # Rocks
        rocks_cfg = self._cfg.get("rocks_settings")
        if rocks_cfg is not None:
            from objects.rock_manager import RockManager

            self._rock_manager = RockManager(rocks_cfg)
            # Get DEM and mask from terrain
            dem = self._terrain_manager.get_dem() if self._terrain_manager else None
            mask = self._terrain_manager.get_mask() if self._terrain_manager else None
            if dem is not None and mask is not None:
                self._rock_manager.build(dem, mask)
                # Pass crater data from terrain if available
                craters = self._terrain_manager.get_craters_data()
                if craters:
                    self._rock_manager.set_craters_data(craters)
                self._rock_manager.randomize()
            else:
                logger.warning("No DEM/mask available, skipping rock placement")
            logger.info("Rocks placed")

        # Static assets (lander, background landscape, etc.)
        static_cfg = self._cfg.get("static_assets_settings")
        if static_cfg is not None and self._stage is not None:
            self._load_static_assets(static_cfg)

    def _resolve_static_asset_usd_path(self, asset: dict, assets_root: str = "") -> str:
        """Return the USD path for an asset, applying crop_scale suffix if available.

        If ``crop_scale`` is set and the suffixed file exists, returns the
        cropped variant path.  Falls back to the original path with a warning
        if the file is not found.

        Args:
            asset: Asset config dict from static_assets_settings.parameters.

        Returns:
            Resolved USD file path (raw, before joining with assets_root).
        """
        import os
        usd_path = asset["usd_path"]
        crop_scale = asset.get("crop_scale")
        if crop_scale is not None:
            stem, suffix = os.path.splitext(usd_path)
            candidate = f"{stem}_{int(crop_scale)}x{suffix}"
            abs_candidate = os.path.join(assets_root, candidate)
            if os.path.isfile(abs_candidate):
                logger.info(
                    "Using cropped landscape (scale %dx): %s", int(crop_scale), abs_candidate
                )
                return candidate
            logger.warning(
                "Cropped USD not found for scale %dx: %s — falling back to original",
                int(crop_scale), abs_candidate,
            )
        return usd_path

    def _load_static_assets(self, cfg) -> None:
        """Load static USD assets (lander, background landscape, etc.)."""
        import os
        from assets import get_assets_path, resolve_path
        from core.pxr_utils import addDefaultOps, setDefaultOps

        try:
            from pxr import UsdGeom, UsdPhysics, Gf
        except ImportError:
            logger.warning("USD not available, skipping static assets")
            return

        root_path = cfg.get("root_path", "/StaticAssets") if isinstance(cfg, dict) else "/StaticAssets"
        self._stage.DefinePrim(root_path, "Xform")

        parameters = cfg.get("parameters", []) if isinstance(cfg, dict) else []
        assets_root = get_assets_path()

        for asset in parameters:
            name = asset["asset_name"]
            prim_path = f"{root_path}/{name}"
            raw_usd_path = self._resolve_static_asset_usd_path(asset, assets_root)
            usd_path = os.path.join(assets_root, raw_usd_path)

            # Create prim and add USD reference
            prim = self._stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)

            # Set pose
            pose = asset.get("pose", {})
            pos = pose.get("position", [0, 0, 0])
            orient = pose.get("orientation", [0, 0, 0, 1])

            xformable = UsdGeom.Xformable(prim)
            # Find or create translate/orient ops, respecting existing precision
            translate_op = None
            orient_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    orient_op = op

            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            if orient_op is None:
                orient_op = xformable.AddOrientOp()

            # Set translate (match precision)
            if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                translate_op.Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
            else:
                translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

            # Set orient (match precision)
            if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                orient_op.Set(Gf.Quatf(float(orient[3]), float(orient[0]), float(orient[1]), float(orient[2])))
            else:
                orient_op.Set(Gf.Quatd(float(orient[3]), float(orient[0]), float(orient[1]), float(orient[2])))

            # Set collision
            if asset.get("collision", True):
                try:
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                except Exception:
                    pass

            # Override material if specified in config
            material_override = asset.get("material_override")
            if material_override:
                self._apply_material_override(prim, material_override)

            # Transition strip (for elevated main terrain → background landscape)
            transition_cfg = asset.get("transition_strip")
            if transition_cfg and transition_cfg.get("enabled", False):
                self._build_background_transition(
                    root_path,
                    prim_path,
                    transition_cfg,
                    material_override or "",
                )

            logger.info("Static asset '%s' loaded from %s", name, usd_path)

    def _build_background_transition(
        self,
        root_path: str,
        bg_prim_path: str,
        transition_cfg: dict,
        material_path: str,
    ) -> None:
        """Build and render a Hermite transition strip between main terrain and background.

        Args:
            root_path: USD root path for static assets (e.g. "/StaticAssets").
            bg_prim_path: USD path of the loaded background landscape prim.
            transition_cfg: Dict with optional keys: band_width (float, default 10.0),
                n_subdivisions (int, default 8).
            material_path: USD material path to apply to the transition mesh.
        """
        import core.pxr_utils as pxr_utils
        from terrain.static_transition import sample_background_z, render_static_transition

        if self._terrain_manager is None:
            logger.warning("No terrain manager available; skipping background transition strip")
            return

        main_dem = self._terrain_manager.get_dem()
        if main_dem is None:
            logger.warning("Terrain DEM not available; skipping background transition strip")
            return

        tm_cfg = self._cfg.get("terrain_manager", {})
        main_res = (
            tm_cfg.get("resolution", 0.02)
            if isinstance(tm_cfg, dict)
            else getattr(tm_cfg, "resolution", 0.02)
        )
        mesh_position = (
            tm_cfg.get("mesh_position", [0.0, 0.0, 0.0])
            if isinstance(tm_cfg, dict)
            else list(getattr(tm_cfg, "mesh_position", [0.0, 0.0, 0.0]))
        )
        sim_length = (
            tm_cfg.get("sim_length", 40.0)
            if isinstance(tm_cfg, dict)
            else getattr(tm_cfg, "sim_length", 40.0)
        )
        sim_width = (
            tm_cfg.get("sim_width", 40.0)
            if isinstance(tm_cfg, dict)
            else getattr(tm_cfg, "sim_width", 40.0)
        )

        main_pos = (float(mesh_position[0]), float(mesh_position[1]), float(mesh_position[2]))
        main_size = (float(sim_width), float(sim_length))

        x_min, x_max = main_pos[0], main_pos[0] + main_size[0]
        y_min, y_max = main_pos[1], main_pos[1] + main_size[1]

        outer_z = sample_background_z(
            self._stage, bg_prim_path, x_min, x_max, y_min, y_max
        )

        render_static_transition(
            self._stage,
            pxr_utils,
            root_path,
            main_dem,
            float(main_res),
            main_pos,
            main_size,
            outer_z,
            band_width=float(transition_cfg.get("band_width", 10.0)),
            n_subdivisions=int(transition_cfg.get("n_subdivisions", 8)),
            material_path=material_path,
        )
        logger.info("Background transition strip rendered (outer_z=%.3f m)", outer_z)

    def _apply_material_override(self, prim, material_path: str) -> None:
        """Override all Mesh descendants of a prim with the given material.

        Args:
            prim: The root USD prim whose descendant meshes will be rebound.
            material_path: USD stage path to the material, e.g.
                ``/LunarYard/Looks/LunarRegolith8k_landscape``.
        """
        try:
            from pxr import UsdShade, UsdGeom, Usd
        except ImportError:
            return

        mtl_prim = self._stage.GetPrimAtPath(material_path)
        if not mtl_prim.IsValid():
            logger.warning("Material override path not found: %s", material_path)
            return

        material = UsdShade.Material(mtl_prim)
        count = 0
        for descendant in Usd.PrimRange(prim):
            if descendant.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI(descendant).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants)
                count += 1

        logger.info("Material override applied to %d meshes under %s -> %s",
                     count, prim.GetPath(), material_path)

    def instantiate_scene(self) -> None:
        """Post-renderer initialization: post-processing and colliders."""
        logger.info("Instantiating LunarYard scene")

        # Post-processing
        rendering_cfg = self._cfg.get("rendering")
        if rendering_cfg is not None:
            from rendering.post_processing import PostProcessingManager

            self._post_processing = PostProcessingManager()
            self._post_processing.apply(rendering_cfg)
            logger.info("Post-processing applied")

        # Update terrain collider
        if self._terrain_manager is not None:
            self._terrain_manager.update_collider()
            logger.info("Terrain collider updated")

    def update(self, dt: float) -> None:
        """Per-frame update: advance stellar engine and update celestial bodies."""
        if self._stellar_engine is not None:
            changed = self._stellar_engine.update(dt)
            if changed:
                if self._sun_controller is not None:
                    self._sun_controller.update_from_stellar(self._stellar_engine)
                if self._earth_controller is not None:
                    self._earth_controller.update_from_stellar(self._stellar_engine)
                sun_alt, sun_az, _ = self._stellar_engine.get_sun_alt_az()
                if self._starfield is not None:
                    self._starfield.update(sun_alt)
                if self._earthshine is not None:
                    earth_alt, earth_az, _ = self._stellar_engine.get_earth_alt_az()
                    self._earthshine.update(sun_alt, earth_alt, earth_az, sun_az)

    def reset(self) -> None:
        """Re-generate terrain and rocks with a new seed."""
        self._seed += 1
        logger.info("Resetting LunarYard with seed %d", self._seed)

        if self._terrain_manager is not None:
            self._terrain_manager.generate_terrain(self._seed)
            self._terrain_manager.render_mesh()
            self._terrain_manager.update_collider()

        if self._rock_manager is not None:
            if self._terrain_manager is not None:
                craters = self._terrain_manager.get_craters_data()
                if craters:
                    self._rock_manager.set_craters_data(craters)
            self._rock_manager.randomize()

    def set_robot_manager(self, robot_manager) -> None:
        """Register robot manager for terramechanics integration."""
        self._robot_manager = robot_manager

        # Update terramechanics solver num_wheels to match actual robot
        from physics.terramechanics import TerramechanicsSolver
        for rrg in robot_manager.rigid_groups.values():
            n_wheels = len(rrg._target_links)
            self._robot_param.num_wheels = n_wheels
            self._terramechanics_solver = TerramechanicsSolver(
                robot_param=self._robot_param,
                terrain_param=self._terrain_mech_param,
            )
            logger.info("Terramechanics: %d wheels for robot %s",
                         n_wheels, rrg._robot_name)
            break  # Use first robot's wheel count

        logger.info("RobotManager registered for terramechanics (%d robots)",
                     robot_manager.num_robots)

    def apply_terramechanics(self) -> None:
        """Compute and apply Bekker-Janosi terramechanics forces to all robot wheels."""
        if not self._terramechanics_enabled or self._robot_manager is None:
            return

        for rrg in self._robot_manager.rigid_groups.values():
            robot_name = rrg._robot_name

            # Skip robots that should use PhysX friction only (no terramechanics)
            if robot_name in self._terramechanics_skip_robots:
                continue

            # Only apply forces to wheels that are in contact with the ground
            raw_contact = rrg.get_net_contact_forces()  # (N, 3)

            # Per-robot EMA filter: smooth contact forces to reduce PhysX jitter.
            # EMA(t) = alpha * EMA(t-1) + (1 - alpha) * raw(t)
            # Initialize to zero so landing impact spikes ramp in gradually.
            alpha = self._contact_ema_alpha
            if robot_name not in self._contact_ema or self._contact_ema[robot_name].shape != raw_contact.shape:
                self._contact_ema[robot_name] = np.zeros_like(raw_contact)
            self._contact_ema[robot_name] = alpha * self._contact_ema[robot_name] + (1.0 - alpha) * raw_contact

            contact_forces = self._contact_ema[robot_name]
            contact_magnitude = np.linalg.norm(contact_forces, axis=1)
            in_contact = contact_magnitude > 1.0  # threshold: 1 N minimum

            if not np.any(in_contact):
                continue

            linear_velocities, angular_velocities = rrg.get_velocities()
            n_wheels = linear_velocities.shape[0]

            # Get wheel orientations to project velocities into local frame.
            # get_pose() returns pitch-corrected orientations (heading only),
            # so forward direction lies in the ground plane.
            positions, orientations = rrg.get_pose()

            # Build per-wheel forward and axle directions in world frame
            forward_dirs = np.zeros((n_wheels, 3))
            axle_dirs = np.zeros((n_wheels, 3))
            for i in range(n_wheels):
                # orientations are (w, x, y, z)
                w, x, y, z = orientations[i]
                # Rotate local X=[1,0,0] by quaternion -> forward direction
                forward_dirs[i] = [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y + w * z),
                    2 * (x * z - w * y),
                ]
                # Rotate local Y=[0,1,0] by quaternion -> axle direction
                axle_dirs[i] = [
                    2 * (x * y - w * z),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z + w * x),
                ]

            # Build per-wheel lateral direction (perpendicular to forward in ground plane)
            # lateral = cross(forward, up) where up = [0, 0, 1]
            lateral_dirs = np.zeros((n_wheels, 3))
            lateral_dirs[:, 0] = forward_dirs[:, 1]   # forward.y
            lateral_dirs[:, 1] = -forward_dirs[:, 0]  # -forward.x
            # Normalize (forward_dirs are already unit-length in XY, but be safe)
            lat_norm = np.linalg.norm(lateral_dirs, axis=1, keepdims=True)
            lat_norm = np.where(lat_norm > 1e-8, lat_norm, 1.0)
            lateral_dirs /= lat_norm

            # Project world-frame velocities onto wheel-local directions
            forward_speed = np.einsum("ij,ij->i", linear_velocities, forward_dirs)
            lateral_speed = np.einsum("ij,ij->i", linear_velocities, lateral_dirs)
            wheel_spin = np.einsum("ij,ij->i", angular_velocities, axle_dirs)

            # Apply terramechanics when wheel is active:
            #   - Driven: wheel is spinning (rolling resistance + traction)
            #   - Sliding: wheel is locked but rover body is moving (braking drag)
            # Only skip when both wheel and body are essentially stationary.
            _MIN_SPEED = 0.02   # m/s — body speed noise floor
            _MIN_OMEGA = 0.05   # rad/s — wheel spin noise floor
            body_speed = np.linalg.norm(linear_velocities[:, :2], axis=1)  # XY plane
            active = (np.abs(wheel_spin) > _MIN_OMEGA) | (body_speed > _MIN_SPEED)

            if not np.any(in_contact & active):
                continue

            # Compute sinkage from PhysX contact normal force via inverted Bekker eq.
            # PhysX treats terrain as rigid, so geometric penetration ≈ 0.
            # Instead, we compute the equilibrium sinkage that the contact force
            # would produce on Bekker soil:
            #   F_z ≈ 2*b*(k_c/b + k_phi)*sqrt(2*r) * z^(n+0.5)
            #   z = (F_z / (2*b*(k_c/b+k_phi)*sqrt(2*r)))^(1/(n+0.5))
            t = self._terrain_mech_param
            rp = self._robot_param
            bekker_mod = t.k_c / rp.wheel_width + t.k_phi
            denom = 2.0 * rp.wheel_width * bekker_mod * np.sqrt(2.0 * rp.wheel_radius)
            inv_exp = 1.0 / (t.n + 0.5)

            # Use vertical component of contact force (positive = into ground).
            # Clamp to 3x static weight to allow slope-induced weight transfer
            # while still rejecting PhysX impact/bounce spikes.
            static_weight = rp.mass * t.gravity / rp.num_wheels
            contact_fz = np.clip(np.abs(contact_forces[:, 2]), 0.0, static_weight * 3.0)
            sinkages = np.where(
                (contact_fz > 1.0) & in_contact,
                (contact_fz / denom) ** inv_exp,
                0.0,
            )
            # Clamp to physical maximum (wheel can't sink past its radius)
            np.clip(sinkages, 0.0, rp.wheel_radius * 0.5, out=sinkages)

            # Compute per-wheel soil variation if heterogeneity is enabled
            soil_mult = self._sample_soil_variation(positions, n_wheels)

            # Solver returns Fx (along forward), Fy (lateral), My (around axle)
            local_forces, local_torques = self._terramechanics_solver.compute_force_and_torque(
                forward_speed, wheel_spin, sinkages,
                soil_multipliers=soil_mult,
                lateral_speed=lateral_speed,
                contact_fz=contact_fz,
                physx_mu=self._terrain_mech_param.physx_mu or None,
            )

            # Zero out forces for non-contact or fully stationary wheels
            inactive = ~(in_contact & active)
            local_forces[inactive] = 0.0
            local_torques[inactive] = 0.0

            # Periodic debug logging (every ~2 seconds at 60Hz)
            if not hasattr(self, '_tm_log_counter'):
                self._tm_log_counter = 0
            self._tm_log_counter += 1
            if self._tm_log_counter % 120 == 1:
                fx_total = np.sum(local_forces[:, 0])
                fy_mag = np.sum(np.abs(local_forces[:, 1]))
                my_mag = np.sum(np.abs(local_torques[:, 1]))
                logger.info(
                    "[TM %s] sink=%.1fmm, Fx=%.1fN, Fy=%.1fN, My=%.2fNm, "
                    "slip=%.2f, v=%.2fm/s, w=%.1frad/s",
                    robot_name,
                    (np.mean(sinkages[in_contact]) if np.any(in_contact) else 0.0) * 1000,
                    fx_total, fy_mag, my_mag,
                    np.mean(np.abs(self._terramechanics_solver._compute_slip_ratio_vec(
                        forward_speed, wheel_spin))),
                    np.mean(np.abs(forward_speed)),
                    np.mean(np.abs(wheel_spin)),
                )

            # Transform solver output to world frame:
            #   world_force  = Fx * forward_dir + Fy * lateral_dir
            #   world_torque = My * axle_dir
            world_forces = (local_forces[:, 0:1] * forward_dirs    # Fx
                          + local_forces[:, 1:2] * lateral_dirs)   # Fy
            world_torques = local_torques[:, 1:2] * axle_dirs      # My

            rrg.apply_force_torque(world_forces, world_torques, is_global=True)

    def _sample_soil_variation(self, positions: np.ndarray, n_wheels: int) -> np.ndarray:
        """Sample spatially-coherent soil variation at wheel positions.

        Uses a simple hash-based noise to create patches of loose/compact
        regolith without requiring a Perlin noise library.

        Args:
            positions: Wheel positions (N, 3).
            n_wheels: Number of wheels.

        Returns:
            Soil multipliers (N,). 1.0 = nominal, >1 = looser, <1 = more compact.
        """
        t = self._terrain_mech_param
        if t.heterogeneity <= 0.0:
            return np.ones(n_wheels)

        scale = t.heterogeneity_scale
        seed = t.heterogeneity_seed
        h = t.heterogeneity  # 0..1

        # Multi-octave value noise for natural-looking patches.
        # Uses deterministic hashing so the same position always gives the
        # same value (no temporal flickering).
        result = np.zeros(n_wheels)
        for octave in range(3):
            freq = (1 << octave) / scale  # 1/s, 2/s, 4/s
            amp = 1.0 / (1 << octave)     # 1, 0.5, 0.25
            # Grid-based value noise with smooth interpolation
            gx = positions[:, 0] * freq
            gy = positions[:, 1] * freq
            ix = np.floor(gx).astype(int)
            iy = np.floor(gy).astype(int)
            fx = gx - ix  # fractional part
            fy = gy - iy
            # Smoothstep for interpolation
            fx = fx * fx * (3 - 2 * fx)
            fy = fy * fy * (3 - 2 * fy)
            # Hash corners to pseudo-random values in [0, 1]
            def _hash(x, y):
                h = ((x * 374761393 + y * 668265263 + seed) ^ (seed * 1274126177)) & 0x7FFFFFFF
                return (h % 10007) / 10007.0
            v00 = np.array([_hash(int(ix[i]), int(iy[i])) for i in range(n_wheels)])
            v10 = np.array([_hash(int(ix[i]) + 1, int(iy[i])) for i in range(n_wheels)])
            v01 = np.array([_hash(int(ix[i]), int(iy[i]) + 1) for i in range(n_wheels)])
            v11 = np.array([_hash(int(ix[i]) + 1, int(iy[i]) + 1) for i in range(n_wheels)])
            # Bilinear interpolation
            v0 = v00 * (1 - fx) + v10 * fx
            v1 = v01 * (1 - fx) + v11 * fx
            val = v0 * (1 - fy) + v1 * fy
            result += amp * (val - 0.5) * 2  # map [0,1] -> [-1,1]

        # Normalize to [-1, 1] range, then scale by heterogeneity
        result = result / 1.75  # sum of amplitudes: 1 + 0.5 + 0.25 = 1.75
        # Map to multiplier: h=1 gives range [0.4, 2.5] (log-symmetric)
        return np.exp(result * h * 0.9)  # exp(±0.9) ≈ [0.41, 2.46]

    def deform_terrain(self) -> None:
        """Apply wheel-terrain deformation using contact forces from the physics engine."""
        if self._robot_manager is None or self._terrain_manager is None:
            return

        cfg = self._cfg.get("terrain_manager")
        if cfg is None:
            return
        moon_yard = getattr(cfg, "moon_yard", None) or cfg.get("moon_yard", {})
        deform_cfg = getattr(moon_yard, "deformation_engine", None)
        if deform_cfg is None or not getattr(deform_cfg, "enable", False):
            return

        for rrg in self._robot_manager.rigid_groups.values():
            positions, orientations = rrg.get_pose()
            contact_forces = rrg.get_net_contact_forces()
            self._terrain_manager.deform(positions, orientations, contact_forces)

    def _create_root_xform(self) -> None:
        """Create the root USD Xform prim for the environment."""
        try:
            from pxr import UsdGeom
            UsdGeom.Xform.Define(self._stage, self._root_path)
            logger.info("Created root Xform at %s", self._root_path)
        except ImportError:
            logger.warning("USD not available, skipping root Xform creation")

    def _load_materials(self) -> None:
        """Load only the MDL materials actually referenced by the config."""
        import os
        from assets import resolve_path
        try:
            import omni.kit.commands
        except ImportError:
            logger.warning("omni.kit not available, skipping material loading")
            return

        looks_path = os.path.join(self._root_path, "Looks")
        self._stage.DefinePrim(looks_path, "Scope")

        # Map material name -> MDL file
        all_materials = {
            "Basalt": "assets/Textures/GravelStones.mdl",
            "Sand": "assets/Textures/Sand.mdl",
            "LunarRegolith8k": "assets/Textures/LunarRegolith8k.mdl",
            "LunarRegolith8k_antiTile": "assets/Textures/LunarRegolith8k_antiTile.mdl",
            "LunarRegolith8k_stochastic": "assets/Textures/LunarRegolith8k_stochastic.mdl",
        }

        # Collect material names referenced in config texture_path fields
        needed = set()
        terrain_cfg = self._cfg.get("terrain_manager", {})
        tex = getattr(terrain_cfg, "texture_path", None) or (
            terrain_cfg.get("texture_path", "") if isinstance(terrain_cfg, dict) else ""
        )
        if tex:
            needed.add(tex.rsplit("/", 1)[-1])

        landscape_cfg = self._cfg.get("landscape_settings", {})
        ltex = getattr(landscape_cfg, "texture_path", None) or (
            landscape_cfg.get("texture_path", "") if isinstance(landscape_cfg, dict) else ""
        )
        if ltex:
            needed.add(ltex.rsplit("/", 1)[-1])

        # Filter to only known materials that are actually needed
        needed = needed & set(all_materials.keys())
        if not needed:
            # Fallback: load all if no recognized material found
            needed = set(all_materials.keys())

        for mat_name in sorted(needed):
            mdl_rel_path = all_materials[mat_name]
            mdl_path = resolve_path(mdl_rel_path)
            if os.path.exists(mdl_path):
                mtl_path = os.path.join(looks_path, mat_name)
                omni.kit.commands.execute(
                    "CreateMdlMaterialPrimCommand",
                    mtl_url=mdl_path,
                    mtl_name=mat_name,
                    mtl_path=mtl_path,
                )
                logger.info("Loaded material '%s' from %s", mat_name, mdl_path)
            else:
                logger.warning("Material file not found: %s", mdl_path)


# Auto-register with SimulationManager
register_environment("LunarYard", LunarYardEnvironment)
