"""Tests for NumPy-based Perlin noise utilities."""

import numpy as np
import pytest

from terrain.procedural.noise import perlin_1d, perlin_2d


class TestPerlin1D:
    """Tests for 1D Perlin noise with periodic boundary conditions."""

    def test_output_shape_matches_input(self):
        t = np.linspace(0, 1, 100)
        result = perlin_1d(t, seed=42)
        assert result.shape == (100,)

    def test_values_in_range(self):
        t = np.linspace(0, 10, 1000)
        result = perlin_1d(t, seed=42)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_periodicity(self):
        """Noise at t and t+period should match for periodic mode."""
        period = 2 * np.pi
        t = np.linspace(0, period, 200, endpoint=False)
        result = perlin_1d(t, period=period, seed=42)
        assert abs(result[0] - result[-1]) < 0.1

    def test_different_seeds_different_output(self):
        t = np.linspace(0, 5, 100)
        r1 = perlin_1d(t, seed=42)
        r2 = perlin_1d(t, seed=99)
        assert not np.allclose(r1, r2)

    def test_same_seed_reproducible(self):
        t = np.linspace(0, 5, 100)
        r1 = perlin_1d(t, seed=42)
        r2 = perlin_1d(t, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_not_all_zeros(self):
        t = np.linspace(0, 5, 100)
        result = perlin_1d(t, seed=42)
        assert np.std(result) > 0.01


class TestPerlin2D:
    """Tests for 2D Perlin noise."""

    def test_output_shape_matches_input(self):
        x = np.linspace(0, 5, 50)
        y = np.linspace(0, 5, 50)
        xx, yy = np.meshgrid(x, y)
        result = perlin_2d(xx, yy, seed=42)
        assert result.shape == (50, 50)

    def test_values_in_range(self):
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        result = perlin_2d(xx, yy, seed=42)
        assert np.all(result >= -1.5)
        assert np.all(result <= 1.5)

    def test_different_seeds_different_output(self):
        x = np.linspace(0, 5, 30)
        y = np.linspace(0, 5, 30)
        xx, yy = np.meshgrid(x, y)
        r1 = perlin_2d(xx, yy, seed=42)
        r2 = perlin_2d(xx, yy, seed=99)
        assert not np.allclose(r1, r2)

    def test_same_seed_reproducible(self):
        x = np.linspace(0, 5, 30)
        y = np.linspace(0, 5, 30)
        xx, yy = np.meshgrid(x, y)
        r1 = perlin_2d(xx, yy, seed=42)
        r2 = perlin_2d(xx, yy, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_spatial_variation(self):
        """Noise should vary spatially, not be constant."""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        result = perlin_2d(xx, yy, seed=42)
        assert np.std(result) > 0.01
