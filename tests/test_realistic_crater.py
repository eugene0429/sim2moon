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
