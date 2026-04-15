"""Tests for starfield and earthshine effects."""

import math

import numpy as np
import pytest

from effects.config import StarfieldConf, EarthshineConf
from effects.starfield import Starfield, _temperature_to_rgb
from effects.earthshine import Earthshine


# ── StarfieldConf ───────────────────────────────────────────────────────────

class TestStarfieldConf:
    def test_defaults(self):
        cfg = StarfieldConf()
        assert cfg.enable is True
        assert cfg.num_stars == 9000
        assert cfg.sun_fade_start > cfg.sun_fade_end

    def test_invalid_num_stars(self):
        with pytest.raises(AssertionError):
            StarfieldConf(num_stars=0)

    def test_invalid_fade_order(self):
        with pytest.raises(AssertionError):
            StarfieldConf(sun_fade_start=-5.0, sun_fade_end=5.0)


# ── Temperature to RGB ──────────────────────────────────────────────────────

class TestTemperatureToRGB:
    def test_hot_star_is_bluish(self):
        r, g, b = _temperature_to_rgb(25000)
        assert b > r  # Blue should dominate

    def test_cool_star_is_reddish(self):
        r, g, b = _temperature_to_rgb(3000)
        assert r > b  # Red should dominate

    def test_sun_temperature(self):
        r, g, b = _temperature_to_rgb(5778)
        # Should be roughly white-yellowish
        assert r > 0.8
        assert g > 0.7
        assert b > 0.5

    def test_clamped_extremes(self):
        r, g, b = _temperature_to_rgb(100)  # Below min
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1
        r, g, b = _temperature_to_rgb(100000)  # Above max
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1


# ── Starfield generation ───────────────────────────────────────────────────

class TestStarfieldGeneration:
    def test_generate_creates_catalog(self):
        sf = Starfield(StarfieldConf(num_stars=500, seed=42))
        sf.generate()
        assert sf.positions.shape == (500, 3)
        assert sf.magnitudes.shape == (500,)
        assert sf.colors.shape == (500, 3)
        assert sf.star_count == 500

    def test_positions_on_sphere(self):
        cfg = StarfieldConf(num_stars=1000, sphere_radius=5000.0, seed=42)
        sf = Starfield(cfg)
        sf.generate()
        distances = np.linalg.norm(sf.positions, axis=1)
        np.testing.assert_allclose(distances, 5000.0, atol=1.0)

    def test_magnitudes_within_range(self):
        sf = Starfield(StarfieldConf(num_stars=2000, magnitude_limit=6.5, seed=42))
        sf.generate()
        assert sf.magnitudes.min() >= -2.0  # Some tolerance
        assert sf.magnitudes.max() <= 7.0

    def test_more_faint_than_bright(self):
        sf = Starfield(StarfieldConf(num_stars=5000, seed=42))
        sf.generate()
        median_mag = np.median(sf.magnitudes)
        # Median should be closer to faint end (higher magnitude)
        assert median_mag > 3.0

    def test_colors_valid(self):
        sf = Starfield(StarfieldConf(num_stars=500, seed=42))
        sf.generate()
        assert np.all(sf.colors >= 0)
        assert np.all(sf.colors <= 1)

    def test_deterministic_seed(self):
        sf1 = Starfield(StarfieldConf(num_stars=100, seed=42))
        sf1.generate()
        sf2 = Starfield(StarfieldConf(num_stars=100, seed=42))
        sf2.generate()
        np.testing.assert_array_equal(sf1.positions, sf2.positions)


# ── Starfield brightness ───────────────────────────────────────────────────

class TestStarfieldBrightness:
    def _make_starfield(self):
        sf = Starfield(StarfieldConf(
            num_stars=100,
            sun_fade_start=5.0,
            sun_fade_end=0.0,
            base_brightness=1.0,
            seed=42,
        ))
        sf.generate()
        return sf

    def test_day_invisible(self):
        sf = self._make_starfield()
        b = sf.update(sun_altitude_deg=45.0)
        assert b == 0.0

    def test_night_full_brightness(self):
        sf = self._make_starfield()
        b = sf.update(sun_altitude_deg=-10.0)
        assert b == pytest.approx(1.0)

    def test_twilight_partial(self):
        sf = self._make_starfield()
        b = sf.update(sun_altitude_deg=2.5)  # Middle of fade zone
        assert 0 < b < 1.0

    def test_fade_monotonic(self):
        sf = self._make_starfield()
        altitudes = [10.0, 5.0, 3.0, 1.0, 0.0, -5.0]
        brightnesses = [sf.compute_brightness(a) for a in altitudes]
        # Should be non-decreasing as sun goes lower
        for i in range(len(brightnesses) - 1):
            assert brightnesses[i + 1] >= brightnesses[i]

    def test_disabled_returns_zero(self):
        sf = Starfield(StarfieldConf(enable=False))
        sf.generate()
        assert sf.update(-10.0) == 0.0

    def test_setup_without_usd(self):
        sf = self._make_starfield()
        sf.setup(stage=None)  # Should not raise


# ── EarthshineConf ──────────────────────────────────────────────────────────

class TestEarthshineConf:
    def test_defaults(self):
        cfg = EarthshineConf()
        assert cfg.enable is True
        assert cfg.base_intensity > 0
        assert len(cfg.color) == 3

    def test_invalid_intensity(self):
        with pytest.raises(AssertionError):
            EarthshineConf(base_intensity=-1.0)


# ── Earthshine computation ──────────────────────────────────────────────────

class TestEarthshineComputation:
    def _make_earthshine(self, **kwargs):
        cfg = EarthshineConf(**kwargs)
        return Earthshine(cfg)

    def test_daytime_no_earthshine(self):
        es = self._make_earthshine()
        intensity = es.compute_intensity(
            sun_alt=30.0,    # Sun high
            earth_alt=45.0,
            earth_az=180.0,
            sun_az=0.0,
        )
        assert intensity == 0.0

    def test_night_with_earth_visible(self):
        es = self._make_earthshine()
        intensity = es.compute_intensity(
            sun_alt=-20.0,   # Sun well below horizon
            earth_alt=45.0,  # Earth high
            earth_az=180.0,
            sun_az=0.0,      # Opposite side -> full Earth
        )
        assert intensity > 0.0

    def test_night_earth_below_horizon(self):
        es = self._make_earthshine()
        intensity = es.compute_intensity(
            sun_alt=-20.0,
            earth_alt=-10.0,  # Earth below horizon
            earth_az=180.0,
            sun_az=0.0,
        )
        assert intensity == 0.0

    def test_phase_full_earth_brightest(self):
        es = self._make_earthshine()
        # Full Earth: sun and earth on opposite sides (phase ~ 180°)
        # Sun behind observer illuminates Earth's Moon-facing side
        full_earth = es.compute_intensity(
            sun_alt=-20.0, earth_alt=45.0,
            earth_az=180.0, sun_az=0.0,
        )
        # New Earth: sun and earth in same direction (phase ~ 0°)
        # Sun behind Earth, dark side faces Moon
        new_earth = es.compute_intensity(
            sun_alt=-20.0, earth_alt=45.0,
            earth_az=0.0, sun_az=0.0,
        )
        assert full_earth > new_earth

    def test_higher_earth_brighter(self):
        es = self._make_earthshine()
        low = es.compute_intensity(
            sun_alt=-20.0, earth_alt=10.0,
            earth_az=180.0, sun_az=0.0,
        )
        high = es.compute_intensity(
            sun_alt=-20.0, earth_alt=60.0,
            earth_az=180.0, sun_az=0.0,
        )
        assert high > low

    def test_twilight_gradual(self):
        es = self._make_earthshine(sun_threshold=5.0, sun_fade_range=10.0)
        intensities = []
        for sun_alt in [5.0, 2.0, 0.0, -2.0, -5.0]:
            i = es.compute_intensity(sun_alt, 45.0, 180.0, 0.0)
            intensities.append(i)
        # Should be non-decreasing as sun goes lower
        for i in range(len(intensities) - 1):
            assert intensities[i + 1] >= intensities[i]

    def test_disabled(self):
        es = self._make_earthshine(enable=False)
        assert es.compute_intensity(-20.0, 45.0, 180.0, 0.0) == 0.0

    def test_update_returns_intensity(self):
        es = self._make_earthshine()
        intensity = es.update(-20.0, 45.0, 180.0, 0.0)
        assert intensity > 0.0
        assert es.current_intensity == intensity
        assert es.is_active is True

    def test_setup_without_usd(self):
        es = self._make_earthshine()
        es.setup(stage=None)  # Should not raise


# ── Phase angle computation ─────────────────────────────────────────────────

class TestPhaseAngle:
    def test_same_direction_is_zero(self):
        angle = Earthshine._compute_phase_angle(45.0, 90.0, 45.0, 90.0)
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_opposite_direction_is_180(self):
        angle = Earthshine._compute_phase_angle(0.0, 0.0, 0.0, 180.0)
        assert angle == pytest.approx(180.0, abs=0.1)

    def test_perpendicular_is_90(self):
        angle = Earthshine._compute_phase_angle(0.0, 0.0, 0.0, 90.0)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_phase_to_illumination_full(self):
        # phase 180° = sun opposite Earth = full Earth (max illumination)
        illum = Earthshine._phase_to_illumination(180.0)
        assert illum == pytest.approx(1.0, abs=0.01)

    def test_phase_to_illumination_new(self):
        # phase 0° = sun behind Earth = new Earth (no illumination)
        illum = Earthshine._phase_to_illumination(0.0)
        assert illum == pytest.approx(0.0)

    def test_phase_to_illumination_half(self):
        illum = Earthshine._phase_to_illumination(90.0)
        assert illum == pytest.approx(0.5, abs=0.01)
