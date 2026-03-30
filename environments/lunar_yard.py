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
            from celestial.stellar_engine import StellarEngine

            self._stellar_engine = StellarEngine(stellar_cfg)
            coords = self._lunaryard_conf.coordinates
            self._stellar_engine.set_coordinates(coords.latitude, coords.longitude)
            logger.info("StellarEngine initialized at lat=%.1f, lon=%.1f",
                        coords.latitude, coords.longitude)

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
            usd_path = os.path.join(assets_root, asset["usd_path"])

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

            logger.info("Static asset '%s' loaded from %s", name, usd_path)

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

    def deform_terrain(self) -> None:
        """Delegate terrain deformation. No-op until robots provide contact data."""
        pass

    def _create_root_xform(self) -> None:
        """Create the root USD Xform prim for the environment."""
        try:
            from pxr import UsdGeom
            UsdGeom.Xform.Define(self._stage, self._root_path)
            logger.info("Created root Xform at %s", self._root_path)
        except ImportError:
            logger.warning("USD not available, skipping root Xform creation")

    def _load_materials(self) -> None:
        """Load MDL materials into the USD stage for terrain texturing."""
        import os
        from assets import resolve_path
        try:
            import omni.kit.commands
        except ImportError:
            logger.warning("omni.kit not available, skipping material loading")
            return

        looks_path = os.path.join(self._root_path, "Looks")
        self._stage.DefinePrim(looks_path, "Scope")

        materials = [
            ("Basalt", "assets/Textures/GravelStones.mdl"),
            ("Sand", "assets/Textures/Sand.mdl"),
            ("LunarRegolith8k", "assets/Textures/LunarRegolith8k.mdl"),
            ("LunarRegolith8k_antiTile", "assets/Textures/LunarRegolith8k_antiTile.mdl"),
            ("LunarRegolith8k_stochastic", "assets/Textures/LunarRegolith8k_stochastic.mdl"),
        ]
        for mat_name, mdl_rel_path in materials:
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
