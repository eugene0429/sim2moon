"""Tests for realistic crater generator."""

import numpy as np
import pytest

from terrain.config import CraterGeneratorConf, RealisticCraterConf, MoonYardConf


class TestRealisticCraterConf:
    def test_defaults(self):
        conf = RealisticCraterConf()
        assert conf.n_harmonics == 4
        assert conf.harmonic_amp == 0.12
        assert conf.contour_noise_amp == 0.03
        assert conf.rim_n_harmonics == 3
        assert conf.rim_noise_amp == 0.15
        assert conf.slump_intensity == 0.1
        assert conf.slump_wall_range == (0.3, 0.8)
        assert conf.floor_noise_amp == 0.03
        assert conf.floor_radius_ratio == 0.3

    def test_from_dict(self):
        conf = RealisticCraterConf(**{"n_harmonics": 6, "harmonic_amp": 0.2})
        assert conf.n_harmonics == 6
        assert conf.harmonic_amp == 0.2

    def test_validation_n_harmonics_positive(self):
        with pytest.raises(AssertionError):
            RealisticCraterConf(n_harmonics=0)

    def test_validation_harmonic_amp_range(self):
        with pytest.raises(AssertionError):
            RealisticCraterConf(harmonic_amp=-0.1)

    def test_validation_slump_wall_range_order(self):
        with pytest.raises(AssertionError):
            RealisticCraterConf(slump_wall_range=(0.8, 0.3))


class TestCraterGeneratorConfMode:
    def test_default_mode_is_classic(self):
        conf = CraterGeneratorConf()
        assert conf.crater_mode == "classic"

    def test_realistic_mode(self):
        conf = CraterGeneratorConf(crater_mode="realistic")
        assert conf.crater_mode == "realistic"

    def test_invalid_mode_raises(self):
        with pytest.raises(AssertionError):
            CraterGeneratorConf(crater_mode="invalid")


class TestMoonYardConfRealistic:
    def test_default_has_realistic_crater(self):
        conf = MoonYardConf()
        assert isinstance(conf.realistic_crater, RealisticCraterConf)

    def test_from_dict_with_realistic(self):
        conf = MoonYardConf(realistic_crater={"n_harmonics": 5})
        assert conf.realistic_crater.n_harmonics == 5


from terrain.procedural.realistic_crater_generator import (
    RealisticCraterData,
    RealisticCraterGenerator,
)
from terrain.procedural.crater_generator import CraterData, CraterGenerator


class TestRealisticCraterData:
    def test_inherits_crater_data(self):
        rcd = RealisticCraterData()
        assert isinstance(rcd, CraterData)

    def test_has_harmonic_fields(self):
        rcd = RealisticCraterData()
        assert hasattr(rcd, "harmonic_amplitudes")
        assert hasattr(rcd, "harmonic_phases")
        assert hasattr(rcd, "contour_noise_seed")

    def test_has_rim_fields(self):
        rcd = RealisticCraterData()
        assert hasattr(rcd, "rim_amplitudes")
        assert hasattr(rcd, "rim_phases")

    def test_has_slump_and_floor_fields(self):
        rcd = RealisticCraterData()
        assert hasattr(rcd, "slump_noise_seed")
        assert hasattr(rcd, "floor_noise_seed")


class TestRealisticRandomizeParameters:
    @pytest.fixture
    def generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf(n_harmonics=4, rim_n_harmonics=3)
        return RealisticCraterGenerator(cfg, rcfg)

    def test_returns_realistic_crater_data(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert isinstance(cd, RealisticCraterData)

    def test_harmonic_amplitudes_shape(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert cd.harmonic_amplitudes.shape == (4,)
        assert cd.harmonic_phases.shape == (4,)

    def test_rim_amplitudes_shape(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert cd.rim_amplitudes.shape == (3,)
        assert cd.rim_phases.shape == (3,)

    def test_noise_seeds_are_integers(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert isinstance(cd.contour_noise_seed, (int, np.integer))
        assert isinstance(cd.slump_noise_seed, (int, np.integer))
        assert isinstance(cd.floor_noise_seed, (int, np.integer))

    def test_preserves_base_fields(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert cd.size == 101
        assert cd.deformation_spline is not None
        assert cd.marks_spline is not None

    def test_reproducible_with_same_seed(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf()
        gen1 = RealisticCraterGenerator(cfg, rcfg)
        gen2 = RealisticCraterGenerator(cfg, rcfg)
        cd1 = gen1.randomize_parameters(-1, 101)
        cd2 = gen2.randomize_parameters(-1, 101)
        np.testing.assert_array_equal(cd1.harmonic_amplitudes, cd2.harmonic_amplitudes)
        np.testing.assert_array_equal(cd1.rim_amplitudes, cd2.rim_amplitudes)


class TestRealisticDistanceMatrix:
    @pytest.fixture
    def generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf(n_harmonics=4, harmonic_amp=0.12, contour_noise_amp=0.03)
        return RealisticCraterGenerator(cfg, rcfg)

    def test_output_shape(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        dist = generator._centered_distance_matrix(cd)
        assert dist.shape == (101, 101)

    def test_contour_is_irregular(self, generator):
        """Distance values along a circle should vary (not uniform)."""
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        center = 100
        radius = 70
        n_samples = 360
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        ring_vals = []
        for a in angles:
            r, c = int(center + radius * np.sin(a)), int(center + radius * np.cos(a))
            if 0 <= r < 201 and 0 <= c < 201:
                ring_vals.append(dist[r, c])
        ring_vals = np.array(ring_vals)
        assert np.std(ring_vals) > 1.0, f"Ring std too low: {np.std(ring_vals)}"

    def test_center_is_zero(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        dist = generator._centered_distance_matrix(cd)
        center = cd.size // 2
        assert dist[center, center] < 5.0

    def test_clamped_to_half_size(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        dist = generator._centered_distance_matrix(cd)
        assert np.all(dist <= cd.size / 2 + 0.5)


class TestRealisticApplyProfile:
    @pytest.fixture
    def generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf(
            rim_noise_amp=0.15,
            slump_intensity=0.1,
            floor_noise_amp=0.03,
        )
        return RealisticCraterGenerator(cfg, rcfg)

    def test_rim_height_varies_azimuthally(self, generator):
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        center = 100
        rim_r = 90
        n_samples = 360
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        rim_heights = []
        for a in angles:
            r = int(center + rim_r * np.sin(a))
            c = int(center + rim_r * np.cos(a))
            if 0 <= r < 201 and 0 <= c < 201:
                rim_heights.append(crater[r, c])
        rim_heights = np.array(rim_heights)
        assert np.std(rim_heights) > 0.0001, f"Rim std too low: {np.std(rim_heights)}"

    def test_wall_has_slump_noise(self, generator):
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        center = 100
        wall_vals = crater[center, 50:90]
        diffs = np.diff(wall_vals)
        assert np.std(diffs) > 0.0, "Wall region has no variation"

    def test_floor_has_noise(self, generator):
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        center = 100
        floor_patch = crater[center - 15:center + 15, center - 15:center + 15]
        assert np.std(floor_patch) > 0.0, "Floor is perfectly flat"

    def test_output_shape(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        assert crater.shape == (101, 101)

    def test_no_nans(self, generator):
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        assert not np.any(np.isnan(crater))


class TestRealisticEndToEnd:
    @pytest.fixture
    def generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic", seed=42,
        )
        rcfg = RealisticCraterConf()
        return RealisticCraterGenerator(cfg, rcfg)

    def test_generate_single_returns_tuple(self, generator):
        crater, cd = generator.generate_single(size=101)
        assert isinstance(crater, np.ndarray)
        assert isinstance(cd, RealisticCraterData)
        assert crater.shape == (101, 101)

    def test_generate_single_no_nans(self, generator):
        crater, cd = generator.generate_single(size=201)
        assert not np.any(np.isnan(crater))

    def test_generate_single_has_depression(self, generator):
        crater, cd = generator.generate_single(size=201)
        assert np.min(crater) < 0

    def test_generate_craters_on_dem(self, generator):
        # DEM is 500x500 px at default 0.02 m/px = 10m x 10m physical extent.
        # Coords in meters must be within the DEM's extent; radii must be small enough.
        dem = np.zeros((500, 500), dtype=np.float32)
        coords = np.array([[3.0, 3.0], [7.0, 7.0]], dtype=np.float64)
        radii = np.array([0.5, 0.4], dtype=np.float64)
        dem_out, mask, craters_data = generator.generate_craters(dem, coords, radii)
        assert dem_out.shape == (500, 500)
        assert mask.shape == (500, 500)
        assert len(craters_data) == 2
        assert all(isinstance(cd, RealisticCraterData) for cd in craters_data)

    def test_generate_craters_modifies_dem(self, generator):
        # DEM is 500x500 px at default 0.02 m/px = 10m x 10m physical extent.
        dem = np.zeros((500, 500), dtype=np.float32)
        coords = np.array([[3.0, 3.0]], dtype=np.float64)
        radii = np.array([0.5], dtype=np.float64)
        dem_out, _, _ = generator.generate_craters(dem, coords, radii)
        assert not np.allclose(dem_out, 0.0)

    def test_reproducible_with_crater_data(self, generator):
        crater1, cd = generator.generate_single(size=101)
        crater2, _ = generator.generate_single(crater_data=cd)
        np.testing.assert_array_almost_equal(crater1, crater2)

    def test_interface_compatible_with_parent(self, generator):
        # DEM is 200x200 px at default 0.02 m/px = 4m x 4m physical extent.
        dem = np.zeros((200, 200), dtype=np.float32)
        coords = np.array([[2.0, 2.0]], dtype=np.float64)
        radii = np.array([0.3], dtype=np.float64)
        result = generator.generate_craters(dem, coords, radii)
        assert len(result) == 3
        dem_out, mask_out, cd_list = result
        assert isinstance(dem_out, np.ndarray)
        assert isinstance(mask_out, np.ndarray)
        assert isinstance(cd_list, list)


class TestMoonyardIntegration:
    def test_realistic_mode_uses_realistic_generator(self):
        from terrain.procedural.moonyard_generator import MoonyardGenerator
        cfg = MoonYardConf(
            crater_generator=CraterGeneratorConf(
                profiles_path="assets/Terrains/crater_spline_profiles.pkl",
                crater_mode="realistic", seed=42,
            ),
            realistic_crater=RealisticCraterConf(),
            crater_distribution={"x_size": 10.0, "y_size": 10.0, "densities": [0.1], "radius": [[0.5, 1.0]], "seed": 42},
            base_terrain_generator={"x_size": 10.0, "y_size": 10.0, "resolution": 0.05, "seed": 42},
        )
        gen = MoonyardGenerator(cfg)
        assert isinstance(gen._crater_gen, RealisticCraterGenerator)

    def test_classic_mode_uses_classic_generator(self):
        from terrain.procedural.moonyard_generator import MoonyardGenerator
        cfg = MoonYardConf(
            crater_generator=CraterGeneratorConf(
                profiles_path="assets/Terrains/crater_spline_profiles.pkl",
                crater_mode="classic", seed=42,
            ),
        )
        gen = MoonyardGenerator(cfg)
        from terrain.procedural.crater_generator import CraterGenerator
        assert isinstance(gen._crater_gen, CraterGenerator)
        assert not isinstance(gen._crater_gen, RealisticCraterGenerator)

    def test_randomize_with_realistic_mode(self):
        from terrain.procedural.moonyard_generator import MoonyardGenerator
        cfg = MoonYardConf(
            crater_generator=CraterGeneratorConf(
                profiles_path="assets/Terrains/crater_spline_profiles.pkl",
                crater_mode="realistic", seed=42,
            ),
            realistic_crater=RealisticCraterConf(),
            crater_distribution={"x_size": 10.0, "y_size": 10.0, "densities": [0.1], "radius": [[0.5, 1.0]], "seed": 42},
            base_terrain_generator={"x_size": 10.0, "y_size": 10.0, "resolution": 0.05, "seed": 42},
        )
        gen = MoonyardGenerator(cfg)
        dem, mask, craters_data = gen.randomize()
        assert dem.shape[0] > 0
        assert not np.any(np.isnan(dem))
        if len(craters_data) > 0:
            assert isinstance(craters_data[0], RealisticCraterData)


class TestYAMLConfig:
    def test_realistic_rocks_yaml_loads(self):
        import yaml
        with open("config/environment/lunar_yard_40m_realistic_rocks.yaml") as f:
            raw = yaml.safe_load(f)
        tm = raw["terrain_manager"]
        assert tm["moon_yard"]["crater_generator"]["crater_mode"] == "realistic"
        rc = tm["moon_yard"]["realistic_crater"]
        assert rc["n_harmonics"] == 4

    def test_realistic_rocks_yaml_creates_valid_conf(self):
        import yaml
        with open("config/environment/lunar_yard_40m_realistic_rocks.yaml") as f:
            raw = yaml.safe_load(f)
        my = raw["terrain_manager"]["moon_yard"]
        conf = MoonYardConf(**my)
        assert conf.crater_generator.crater_mode == "realistic"
        assert conf.realistic_crater.n_harmonics == 4
