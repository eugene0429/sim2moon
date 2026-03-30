"""Tests for SouthPole DEM landscape builder."""

import os
import tempfile

import numpy as np
import pytest
import yaml

from terrain.config import LandscapeConf
from terrain.landscape_builder import LandscapeBuilder


class TestLandscapeConf:
    def test_defaults(self):
        conf = LandscapeConf()
        assert conf.enable is False
        assert conf.dem_path == ""
        assert conf.crop_size == 500.0
        assert conf.target_resolution == 5.0
        assert conf.hole_margin == 2.0
        assert conf.texture_path == ""
        assert conf.mesh_prim_path == "/LunarYard/Landscape"

    def test_from_dict(self):
        conf = LandscapeConf(**{
            "enable": True,
            "dem_path": "Terrains/SouthPole/Site01_final_adj_5mpp_surf",
            "crop_size": 200.0,
            "target_resolution": 10.0,
        })
        assert conf.enable is True
        assert conf.crop_size == 200.0
        assert conf.target_resolution == 10.0

    def test_validation_crop_size_positive(self):
        with pytest.raises(AssertionError):
            LandscapeConf(crop_size=-1.0)

    def test_validation_target_resolution_positive(self):
        with pytest.raises(AssertionError):
            LandscapeConf(target_resolution=0.0)


class TestLandscapeBuilderLoadDEM:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dem_dir = os.path.join(self.tmpdir, "test_dem")
        os.makedirs(self.dem_dir)

        self.dem_data = np.random.RandomState(42).rand(100, 100).astype(np.float32) * 100
        np.save(os.path.join(self.dem_dir, "dem.npy"), self.dem_data)

        self.meta = {
            "center_coordinates": [0.0, -89.5],
            "pixel_size": [5.0, -5.0],
            "size": [100, 100],
        }
        with open(os.path.join(self.dem_dir, "dem.yaml"), "w") as f:
            yaml.dump(self.meta, f)

    def test_load_dem(self):
        conf = LandscapeConf(enable=True, dem_path=self.dem_dir)
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))
        dem, meta = builder.load_dem()
        assert dem.shape == (100, 100)
        assert meta["pixel_size"] == [5.0, -5.0]
        np.testing.assert_array_equal(dem, self.dem_data)

    def test_load_dem_missing_file(self):
        conf = LandscapeConf(enable=True, dem_path="/nonexistent/path")
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))
        with pytest.raises(FileNotFoundError):
            builder.load_dem()


class TestCropDEM:
    def test_crop_center(self):
        dem = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
        conf = LandscapeConf(enable=True, crop_size=500.0, target_resolution=5.0)
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))

        cropped = builder.crop_dem(dem, dem_resolution=5.0)
        assert cropped.shape == (100, 100)

    def test_crop_larger_than_dem(self):
        dem = np.ones((50, 50), dtype=np.float32)
        conf = LandscapeConf(enable=True, crop_size=9999.0, target_resolution=5.0)
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))

        cropped = builder.crop_dem(dem, dem_resolution=5.0)
        assert cropped.shape == (50, 50)

    def test_crop_preserves_center_values(self):
        dem = np.zeros((200, 200), dtype=np.float32)
        dem[100, 100] = 999.0
        conf = LandscapeConf(enable=True, crop_size=100.0, target_resolution=5.0)
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))

        cropped = builder.crop_dem(dem, dem_resolution=5.0)
        assert cropped[10, 10] == 999.0


class TestCutHole:
    def test_hole_is_nan(self):
        dem = np.ones((200, 200), dtype=np.float32)
        conf = LandscapeConf(enable=True, crop_size=1000.0, hole_margin=0.0)
        builder = LandscapeBuilder(
            conf,
            main_terrain_size=(40.0, 40.0),
            main_terrain_position=(0.0, 0.0, 0.0),
        )

        result = builder.cut_hole(dem, dem_resolution=5.0, dem_extent=1000.0)
        center = result[96:104, 96:104]
        assert np.all(np.isnan(center)), "Hole region should be NaN"

        assert result[0, 0] == 1.0
        assert result[199, 199] == 1.0

    def test_hole_with_margin(self):
        dem = np.ones((200, 200), dtype=np.float32)
        conf = LandscapeConf(enable=True, crop_size=1000.0, hole_margin=10.0)
        builder = LandscapeBuilder(
            conf,
            main_terrain_size=(40.0, 40.0),
            main_terrain_position=(0.0, 0.0, 0.0),
        )

        result = builder.cut_hole(dem, dem_resolution=5.0, dem_extent=1000.0)
        assert np.isnan(result[94, 94])
        assert np.isnan(result[105, 105])


class TestDownsample:
    def test_downsample_factor_2(self):
        dem = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
        conf = LandscapeConf(enable=True, target_resolution=10.0)
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))

        result = builder.downsample(dem, source_resolution=5.0)
        assert result.shape == (50, 50)

    def test_no_downsample_if_same_resolution(self):
        dem = np.ones((100, 100), dtype=np.float32)
        conf = LandscapeConf(enable=True, target_resolution=5.0)
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))

        result = builder.downsample(dem, source_resolution=5.0)
        assert result.shape == (100, 100)

    def test_no_downsample_if_target_finer(self):
        dem = np.ones((100, 100), dtype=np.float32)
        conf = LandscapeConf(enable=True, target_resolution=2.0)
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))

        result = builder.downsample(dem, source_resolution=5.0)
        assert result.shape == (100, 100)

    def test_nan_preserved_after_downsample(self):
        dem = np.ones((100, 100), dtype=np.float32)
        dem[40:60, 40:60] = np.nan
        conf = LandscapeConf(enable=True, target_resolution=10.0)
        builder = LandscapeBuilder(conf, main_terrain_size=(40.0, 40.0))

        result = builder.downsample(dem, source_resolution=5.0)
        assert np.all(np.isnan(result[20:30, 20:30]))
