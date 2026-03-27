"""
LunarYard outdoor procedural environment.

Composes terrain, celestial, rendering, and objects modules into a
concrete BaseEnvironment that can be run by SimulationManager.
"""

import logging
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

    def build_scene(self) -> None:
        """Create USD scene structure and celestial controllers."""
        logger.info("Building LunarYard scene at %s", self._root_path)

        # Create root Xform
        if self._stage is not None:
            self._create_root_xform()

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

    def load(self) -> None:
        """Generate terrain, render mesh, and place rocks."""
        logger.info("Loading LunarYard assets")

        # Terrain
        terrain_cfg = self._cfg.get("terrain_manager")
        if terrain_cfg is not None:
            from terrain.terrain_manager import TerrainManager

            self._terrain_manager = TerrainManager(terrain_cfg)
            self._terrain_manager.generate_terrain(self._seed)
            self._terrain_manager.render_mesh()
            logger.info("Terrain generated and rendered")

        # Rocks
        rocks_cfg = self._cfg.get("rocks_settings")
        if rocks_cfg is not None:
            from objects.rock_manager import RockManager

            self._rock_manager = RockManager(rocks_cfg, seed=self._seed)
            self._rock_manager.build()
            # Pass crater data from terrain if available
            if self._terrain_manager is not None:
                craters = self._terrain_manager.get_craters_data()
                if craters:
                    self._rock_manager.set_craters_data(craters)
            self._rock_manager.randomize()
            logger.info("Rocks placed")

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


# Auto-register with SimulationManager
register_environment("LunarYard", LunarYardEnvironment)
