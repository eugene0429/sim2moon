"""End-to-end test: YAML config -> OmegaConf -> dict -> dataclass instances.

Loads YAML files directly with OmegaConf.load (bypasses Hydra compose
which has version-dependent internal config issues in test environments).
"""

import os

import pytest
from omegaconf import OmegaConf

from core.config_factory import create_config_factory, instantiate_configs, PhysicsConf
from terrain.config import TerrainManagerConf
from celestial.config import StellarEngineConf, SunConf, EarthConf
from objects.config import RockManagerConf
from rendering.config import RenderingConf
from environments.lunar_yard_config import LunarYardConf


CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))


@pytest.fixture(scope="module")
def hydrated_config():
    """Load YAML configs, merge, and hydrate through ConfigFactory."""
    env_cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "environment", "lunar_yard_40m.yaml"))
    physics_cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "physics", "default.yaml"))
    rendering_cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "rendering", "ray_tracing.yaml"))

    merged = OmegaConf.merge(env_cfg, physics_cfg, rendering_cfg)
    cfg_dict = OmegaConf.to_container(merged, resolve=True)

    factory = create_config_factory()
    return instantiate_configs(cfg_dict, factory)


class TestE2EConfigPipeline:
    def test_name_preserved(self, hydrated_config):
        assert hydrated_config.get("name") == "LunarYard"

    def test_seed_preserved(self, hydrated_config):
        assert hydrated_config.get("seed") == 42

    def test_physics_instantiated(self, hydrated_config):
        assert isinstance(hydrated_config.get("physics_scene"), PhysicsConf)
        assert hydrated_config["physics_scene"].gravity == [0.0, 0.0, -1.62]

    def test_lunaryard_settings_instantiated(self, hydrated_config):
        cfg = hydrated_config.get("lunaryard_settings")
        assert isinstance(cfg, LunarYardConf)
        assert cfg.lab_length == 40.0
        assert cfg.coordinates.latitude == 46.8

    def test_terrain_manager_instantiated(self, hydrated_config):
        tm = hydrated_config.get("terrain_manager")
        assert isinstance(tm, TerrainManagerConf)
        assert tm.sim_length == 40.0

    def test_stellar_engine_instantiated(self, hydrated_config):
        assert isinstance(hydrated_config.get("stellar_engine_settings"), StellarEngineConf)

    def test_sun_settings_instantiated(self, hydrated_config):
        assert isinstance(hydrated_config.get("sun_settings"), SunConf)
        assert hydrated_config["sun_settings"].intensity == 1750.0

    def test_earth_settings_instantiated(self, hydrated_config):
        assert isinstance(hydrated_config.get("earth_settings"), EarthConf)

    def test_rocks_settings_instantiated(self, hydrated_config):
        assert isinstance(hydrated_config.get("rocks_settings"), RockManagerConf)
        assert hydrated_config["rocks_settings"].enable is True

    def test_rendering_instantiated(self, hydrated_config):
        cfg = hydrated_config.get("rendering")
        assert isinstance(cfg, RenderingConf)
        assert cfg.renderer.renderer == "RayTracedLighting"
