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
