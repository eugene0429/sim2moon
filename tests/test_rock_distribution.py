"""Tests for rock distribution algorithms."""

import numpy as np
import pytest

from objects.rock_distribution import RockDistributor, RockPlacement, CraterData


class TestPowerLawSampling:
    def test_basic_shape(self):
        rng = np.random.default_rng(42)
        samples = RockDistributor.sample_power_law(rng, 1.0, 10.0, 2.0, 1000)
        assert samples.shape == (1000,)

    def test_within_bounds(self):
        rng = np.random.default_rng(42)
        samples = RockDistributor.sample_power_law(rng, 2.0, 8.0, 2.5, 5000)
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 8.0)

    def test_more_small_than_large(self):
        rng = np.random.default_rng(42)
        samples = RockDistributor.sample_power_law(rng, 1.0, 10.0, 2.5, 10000)
        median = np.median(samples)
        # Power-law: median should be closer to d_min
        assert median < 5.0

    def test_alpha_effect(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        steep = RockDistributor.sample_power_law(rng1, 1.0, 10.0, 3.0, 10000)
        shallow = RockDistributor.sample_power_law(rng2, 1.0, 10.0, 1.5, 10000)
        # Steeper alpha -> smaller mean
        assert np.mean(steep) < np.mean(shallow)


class TestBackgroundScatter:
    def test_returns_rock_placement(self):
        rng = np.random.default_rng(42)
        result = RockDistributor.background_scatter(
            rng, terrain_width=40.0, terrain_height=40.0,
            resolution=0.02, density=0.1,
        )
        assert isinstance(result, RockPlacement)
        assert result.positions.ndim == 2
        assert result.positions.shape[1] == 2
        assert len(result.diameters) == len(result.positions)

    def test_positions_within_bounds(self):
        rng = np.random.default_rng(42)
        result = RockDistributor.background_scatter(
            rng, terrain_width=20.0, terrain_height=30.0,
            resolution=0.02, density=0.5,
        )
        if len(result.positions) > 0:
            assert np.all(result.positions[:, 0] >= 0)
            assert np.all(result.positions[:, 0] < 20.0)
            assert np.all(result.positions[:, 1] >= 0)
            assert np.all(result.positions[:, 1] < 30.0)

    def test_density_affects_count(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        sparse = RockDistributor.background_scatter(
            rng1, 40.0, 40.0, 0.02, density=0.01,
        )
        dense = RockDistributor.background_scatter(
            rng2, 40.0, 40.0, 0.02, density=1.0,
        )
        assert len(dense.positions) > len(sparse.positions)

    def test_mask_filtering(self):
        rng = np.random.default_rng(42)
        # Mask: only top-left quadrant valid
        mask = np.zeros((2000, 2000), dtype=np.uint8)
        mask[:1000, :1000] = 1
        result = RockDistributor.background_scatter(
            rng, 40.0, 40.0, 0.02, mask=mask, density=0.5,
        )
        if len(result.positions) > 0:
            # Positions should be in lower half of terrain width (mask cols 0-999)
            assert np.all(result.positions[:, 0] < 20.0)


class TestClusteredScatter:
    def test_returns_valid_placement(self):
        rng = np.random.default_rng(42)
        result = RockDistributor.clustered_scatter(
            rng, 40.0, 40.0, 0.02,
            lambda_parent=0.01, lambda_daughter=3, sigma=5.0,
        )
        assert isinstance(result, RockPlacement)
        if len(result.positions) > 0:
            assert result.positions.shape[1] == 2


class TestCraterEjecta:
    def test_with_craters(self):
        rng = np.random.default_rng(42)
        # size is in pixels, radius_m = size * resolution / 2
        # size=500 at resolution=0.02 -> radius_m = 5.0m (> min_crater_radius)
        craters = [
            CraterData(coord=(20.0, 20.0), size=500.0),
            CraterData(coord=(15.0, 30.0), size=400.0),
        ]
        result = RockDistributor.crater_ejecta(
            rng, 40.0, 40.0, 0.02, craters,
            rocks_per_crater=20, min_crater_radius=0.5,
        )
        assert isinstance(result, RockPlacement)
        assert len(result.positions) > 0

    def test_no_craters(self):
        rng = np.random.default_rng(42)
        result = RockDistributor.crater_ejecta(
            rng, 40.0, 40.0, 0.02, [],
        )
        assert len(result.positions) == 0

    def test_small_craters_filtered(self):
        rng = np.random.default_rng(42)
        tiny_craters = [CraterData(coord=(10.0, 10.0), size=1.0)]
        result = RockDistributor.crater_ejecta(
            rng, 40.0, 40.0, 0.02, tiny_craters,
            min_crater_radius=5.0,
        )
        assert len(result.positions) == 0


class TestCraterWallDebris:
    def test_with_craters(self):
        rng = np.random.default_rng(42)
        # size=500 at resolution=0.02 -> radius_m = 5.0m
        craters = [CraterData(coord=(20.0, 20.0), size=500.0)]
        result = RockDistributor.crater_wall_debris(
            rng, 40.0, 40.0, 0.02, craters,
            rocks_per_crater=30, min_crater_radius=0.5,
        )
        assert isinstance(result, RockPlacement)
        assert len(result.positions) > 0


class TestOverlapRemoval:
    def test_removes_overlaps(self):
        # Two rocks at same position
        positions = np.array([[5.0, 5.0], [5.01, 5.01]])
        diameters = np.array([2.0, 1.0])
        placement = RockPlacement(positions, diameters)
        result = RockDistributor.remove_overlaps(placement, min_gap=0.0, native_rock_size=1.0)
        assert len(result.positions) == 1

    def test_keeps_separated_rocks(self):
        positions = np.array([[0.0, 0.0], [100.0, 100.0]])
        diameters = np.array([1.0, 1.0])
        placement = RockPlacement(positions, diameters)
        result = RockDistributor.remove_overlaps(placement, min_gap=0.0, native_rock_size=0.3)
        assert len(result.positions) == 2

    def test_preserves_larger_rocks(self):
        # Two overlapping rocks: should keep the larger one
        positions = np.array([[5.0, 5.0], [5.05, 5.05]])
        diameters = np.array([1.0, 3.0])
        placement = RockPlacement(positions, diameters)
        result = RockDistributor.remove_overlaps(placement, min_gap=0.0, native_rock_size=1.0)
        assert len(result.positions) == 1
        assert result.diameters[0] == 3.0

    def test_single_rock(self):
        placement = RockPlacement(np.array([[1.0, 2.0]]), np.array([1.0]))
        result = RockDistributor.remove_overlaps(placement)
        assert len(result.positions) == 1

    def test_empty_placement(self):
        placement = RockPlacement(np.zeros((0, 2)), np.zeros(0))
        result = RockDistributor.remove_overlaps(placement)
        assert len(result.positions) == 0
