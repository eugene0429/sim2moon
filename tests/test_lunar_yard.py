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


import json
import os
import tempfile
from unittest.mock import patch, MagicMock


class TestLoadStaticAssetsCropScale:
    def _make_env(self):
        from environments.lunar_yard import LunarYardEnvironment
        from core.enums import SimulatorMode
        cfg = {"name": "LunarYard", "seed": 0}
        return LunarYardEnvironment(stage=None, mode=SimulatorMode.ROS2, cfg=cfg)

    def test_crop_scale_selects_suffixed_file(self):
        env = self._make_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            cropped_path = os.path.join(tmpdir, "landscape_cropped_10x.usd")
            open(cropped_path, "w").close()
            original_path = os.path.join(tmpdir, "landscape_cropped.usd")
            open(original_path, "w").close()

            asset = {
                "asset_name": "bg",
                "usd_path": original_path,
                "crop_scale": 10,
                "pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
            }
            resolved = env._resolve_static_asset_usd_path(asset)
            assert resolved == cropped_path

    def test_crop_scale_falls_back_to_original_when_file_missing(self):
        env = self._make_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.path.join(tmpdir, "landscape_cropped.usd")
            open(original_path, "w").close()
            # Do NOT create the _10x variant

            asset = {
                "asset_name": "bg",
                "usd_path": original_path,
                "crop_scale": 10,
                "pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
            }
            resolved = env._resolve_static_asset_usd_path(asset)
            assert resolved == original_path

    def test_no_crop_scale_returns_original(self):
        env = self._make_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.path.join(tmpdir, "landscape_cropped.usd")
            open(original_path, "w").close()

            asset = {
                "asset_name": "bg",
                "usd_path": original_path,
                "pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
            }
            resolved = env._resolve_static_asset_usd_path(asset)
            assert resolved == original_path


class TestComputeCropAdjustedPos:
    def _make_env(self, sim_length=80.0, sim_width=80.0, mesh_position=None):
        from environments.lunar_yard import LunarYardEnvironment
        from core.enums import SimulatorMode
        cfg = {
            "name": "LunarYard",
            "seed": 0,
            "terrain_manager": {
                "mesh_position": mesh_position or [0.0, 0.0, 0.0],
                "sim_length": sim_length,
                "sim_width": sim_width,
            },
        }
        return LunarYardEnvironment(stage=None, mode=SimulatorMode.ROS2, cfg=cfg)

    def test_pos_adjusted_to_terrain_center(self):
        env = self._make_env(sim_length=80.0, sim_width=80.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = {"local_cx": 2667.0, "local_cy": -4191.0}
            meta_path = os.path.join(tmpdir, "landscape_cropped_10x_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            original_pos = [-20405.6, -9502.6, 561.8]
            result = env._compute_crop_adjusted_pos(
                "landscape_cropped.usd", 10, tmpdir, original_pos
            )

            # terrain center = mesh_position + sim/2 = (40, 40)
            # new_x = 40 - 2667 = -2627; new_y = 40 - (-4191) = 4231
            assert result is not None
            assert result[0] == pytest.approx(-2627.0)
            assert result[1] == pytest.approx(4231.0)
            assert result[2] == pytest.approx(561.8)  # Z unchanged

    def test_returns_none_when_meta_missing(self):
        env = self._make_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = env._compute_crop_adjusted_pos(
                "landscape_cropped.usd", 10, tmpdir, [0.0, 0.0, 0.0]
            )
            assert result is None

    def test_mesh_position_offset_applied(self):
        env = self._make_env(sim_length=40.0, sim_width=40.0, mesh_position=[10.0, 5.0, 0.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = {"local_cx": 100.0, "local_cy": 200.0}
            meta_path = os.path.join(tmpdir, "landscape_cropped_5x_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            result = env._compute_crop_adjusted_pos(
                "landscape_cropped.usd", 5, tmpdir, [0.0, 0.0, 0.0]
            )

            # terrain center = (10 + 40/2, 5 + 40/2) = (30, 25)
            assert result is not None
            assert result[0] == pytest.approx(30.0 - 100.0)   # -70
            assert result[1] == pytest.approx(25.0 - 200.0)   # -175
