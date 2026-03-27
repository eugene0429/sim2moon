import pytest
from core.config_factory import create_config_factory


EXPECTED_CONFIGS = [
    "physics_scene",
    "renderer",
    "terrain_manager",
    "moon_yard",
    "stellar_engine_settings",
    "sun_settings",
    "earth_settings",
    "rocks_settings",
    "rendering",
    "lunaryard_settings",
]


class TestConfigRegistration:
    def test_all_configs_registered(self):
        factory = create_config_factory()
        names = factory.registered_names()
        for name in EXPECTED_CONFIGS:
            assert name in names, f"Config '{name}' not registered"

    def test_terrain_manager_creates(self):
        factory = create_config_factory()
        conf = factory.create("terrain_manager")
        assert conf.sim_length == 40.0
        assert conf.resolution == 0.02

    def test_stellar_engine_creates(self):
        factory = create_config_factory()
        conf = factory.create("stellar_engine_settings")
        assert conf.time_scale == 36000.0

    def test_sun_settings_creates(self):
        factory = create_config_factory()
        conf = factory.create("sun_settings")
        assert conf.intensity == 1750.0

    def test_earth_settings_creates(self):
        factory = create_config_factory()
        conf = factory.create("earth_settings")
        assert conf.enable is True

    def test_rocks_settings_creates(self):
        factory = create_config_factory()
        conf = factory.create("rocks_settings")
        assert conf.enable is True

    def test_rendering_creates(self):
        factory = create_config_factory()
        conf = factory.create("rendering")
        assert conf.renderer.renderer == "RayTracedLighting"

    def test_lunaryard_settings_creates(self):
        factory = create_config_factory()
        conf = factory.create("lunaryard_settings")
        assert conf.lab_length == 40.0
