import pytest
from environments.lunar_yard import LunarYardEnvironment
from environments.lunar_yard_config import LunarYardConf
from core.enums import SimulatorMode
from core.simulation_manager import _environment_registry


class TestLunarYardRegistration:
    def test_registered_in_environment_registry(self):
        """Importing the module registers the environment."""
        assert "LunarYard" in _environment_registry
        assert _environment_registry["LunarYard"] is LunarYardEnvironment


class TestLunarYardInit:
    def test_instantiation(self):
        cfg = {
            "name": "LunarYard",
            "seed": 42,
            "lunaryard_settings": LunarYardConf(),
        }
        env = LunarYardEnvironment(stage=None, mode=SimulatorMode.ROS2, cfg=cfg)
        assert env._mode is SimulatorMode.ROS2
        assert env._seed == 42

    def test_extracts_lunaryard_settings(self):
        ly_conf = LunarYardConf(lab_length=20.0)
        cfg = {
            "name": "LunarYard",
            "seed": 1,
            "lunaryard_settings": ly_conf,
        }
        env = LunarYardEnvironment(stage=None, mode=SimulatorMode.SDG, cfg=cfg)
        assert env._lunaryard_conf.lab_length == 20.0

    def test_default_lunaryard_settings(self):
        cfg = {"name": "LunarYard", "seed": 0}
        env = LunarYardEnvironment(stage=None, mode=SimulatorMode.ROS2, cfg=cfg)
        assert env._lunaryard_conf.lab_length == 40.0


class TestLunarYardSubmanagers:
    def test_sub_managers_none_before_build(self):
        cfg = {"name": "LunarYard", "seed": 42}
        env = LunarYardEnvironment(stage=None, mode=SimulatorMode.ROS2, cfg=cfg)
        assert env._terrain_manager is None
        assert env._stellar_engine is None
        assert env._sun_controller is None
        assert env._earth_controller is None
        assert env._rock_manager is None
