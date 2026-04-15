"""Tests for the dust particle system."""

import numpy as np
import pytest

from effects.config import DustConf
from effects.dust_manager import DustManager, ParticlePool, DustEmitter


# ── DustConf ────────────────────────────────────────────────────────────────

class TestDustConf:
    def test_defaults(self):
        cfg = DustConf()
        assert cfg.enable is False
        assert cfg.gravity == (0.0, 0.0, -1.62)
        assert cfg.particle_lifetime == 3.0

    def test_invalid_threshold(self):
        with pytest.raises(AssertionError):
            DustConf(force_threshold=-1.0)

    def test_invalid_kick_speed(self):
        with pytest.raises(AssertionError):
            DustConf(kick_speed_min=1.0, kick_speed_max=0.5)

    def test_invalid_opacity(self):
        with pytest.raises(AssertionError):
            DustConf(opacity_initial=1.5)


# ── ParticlePool ────────────────────────────────────────────────────────────

class TestParticlePool:
    def test_initial_empty(self):
        pool = ParticlePool(100)
        assert pool.active_count == 0

    def test_emit_adds_particles(self):
        pool = ParticlePool(100)
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        vel = np.zeros((2, 3), dtype=np.float32)
        n = pool.emit(pos, vel, initial_opacity=0.8)
        assert n == 2
        assert pool.active_count == 2

    def test_emit_respects_capacity(self):
        pool = ParticlePool(3)
        pos = np.zeros((5, 3), dtype=np.float32)
        vel = np.zeros((5, 3), dtype=np.float32)
        n = pool.emit(pos, vel, 1.0)
        assert n == 3  # Capped at pool max
        assert pool.active_count == 3

    def test_emit_empty(self):
        pool = ParticlePool(100)
        n = pool.emit(np.zeros((0, 3)), np.zeros((0, 3)), 1.0)
        assert n == 0

    def test_step_applies_gravity(self):
        pool = ParticlePool(10)
        pos = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        vel = np.zeros((1, 3), dtype=np.float32)
        pool.emit(pos, vel, 1.0)

        gravity = np.array([0.0, 0.0, -1.62], dtype=np.float32)
        pool.step(dt=1.0, gravity=gravity, lifetime=5.0, opacity_decay=0.1)

        active_pos, _, count = pool.get_active()
        assert count == 1
        # Should have fallen: z = 1.0 + 0 * 1.0 + (-1.62) * 1.0 = -0.62
        # But velocity updated first: v_new = -1.62, pos = 1.0 + (-1.62)*1 = -0.62
        assert active_pos[0, 2] < 1.0

    def test_step_kills_expired(self):
        pool = ParticlePool(10)
        pos = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
        vel = np.zeros((1, 3), dtype=np.float32)
        pool.emit(pos, vel, 1.0)

        gravity = np.array([0.0, 0.0, -1.62], dtype=np.float32)
        # Step past lifetime
        for _ in range(50):
            pool.step(dt=0.1, gravity=gravity, lifetime=1.0, opacity_decay=0.5)

        assert pool.active_count == 0

    def test_step_ground_collision(self):
        pool = ParticlePool(10)
        pos = np.array([[0.0, 0.0, 0.01]], dtype=np.float32)
        vel = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        pool.emit(pos, vel, 1.0)

        gravity = np.array([0.0, 0.0, -1.62], dtype=np.float32)
        pool.step(dt=0.1, gravity=gravity, lifetime=5.0, opacity_decay=0.1,
                  ground_z=0.0, restitution=0.3)

        active_pos, _, count = pool.get_active()
        if count > 0:
            assert active_pos[0, 2] >= 0.0  # Should not be below ground

    def test_clear(self):
        pool = ParticlePool(50)
        pos = np.zeros((10, 3), dtype=np.float32)
        vel = np.zeros((10, 3), dtype=np.float32)
        pool.emit(pos, vel, 1.0)
        assert pool.active_count == 10
        pool.clear()
        assert pool.active_count == 0

    def test_recycling(self):
        """Dead particles should be reused for new emissions."""
        pool = ParticlePool(5)
        pos = np.zeros((5, 3), dtype=np.float32)
        vel = np.zeros((5, 3), dtype=np.float32)
        pool.emit(pos, vel, 1.0)
        assert pool.active_count == 5

        # Kill all by aging past lifetime
        gravity = np.zeros(3, dtype=np.float32)
        for _ in range(20):
            pool.step(dt=1.0, gravity=gravity, lifetime=0.5, opacity_decay=1.0)
        assert pool.active_count == 0

        # Should be able to emit again
        n = pool.emit(pos[:3], vel[:3], 1.0)
        assert n == 3
        assert pool.active_count == 3

    def test_opacity_decay(self):
        pool = ParticlePool(10)
        pos = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
        vel = np.zeros((1, 3), dtype=np.float32)
        pool.emit(pos, vel, 0.8)

        gravity = np.zeros(3, dtype=np.float32)
        pool.step(dt=1.0, gravity=gravity, lifetime=10.0, opacity_decay=0.2)

        _, opacities, count = pool.get_active()
        assert count == 1
        assert opacities[0] == pytest.approx(0.6, abs=0.01)


# ── DustEmitter ─────────────────────────────────────────────────────────────

class TestDustEmitter:
    def _make_emitter(self, **cfg_overrides):
        cfg = DustConf(enable=True, **cfg_overrides)
        pool = ParticlePool(cfg.max_particles_per_emitter)
        rng = np.random.default_rng(42)
        return DustEmitter("test_wheel", pool, cfg, rng), pool

    def test_no_emission_below_threshold(self):
        emitter, pool = self._make_emitter(force_threshold=10.0)
        n = emitter.emit(
            dt=0.033,
            contact_position=np.array([0.0, 0.0, 0.0]),
            contact_force=np.array([0.0, 0.0, -5.0]),  # Below threshold
            wheel_velocity=np.array([1.0, 0.0, 0.0]),
        )
        assert n == 0
        assert pool.active_count == 0

    def test_emission_above_threshold(self):
        emitter, pool = self._make_emitter(
            force_threshold=0.1,
            emission_rate_scale=1000.0,
        )
        n = emitter.emit(
            dt=0.1,
            contact_position=np.array([5.0, 5.0, 0.0]),
            contact_force=np.array([0.0, 0.0, -10.0]),
            wheel_velocity=np.array([1.0, 0.0, 0.0]),
        )
        assert n > 0
        assert pool.active_count == n

    def test_higher_force_more_particles(self):
        emitter_lo, pool_lo = self._make_emitter(
            force_threshold=0.0, emission_rate_scale=100.0,
        )
        emitter_hi, pool_hi = self._make_emitter(
            force_threshold=0.0, emission_rate_scale=100.0,
        )

        emitter_lo.emit(0.1, np.zeros(3), np.array([0, 0, -2.0]), np.zeros(3))
        emitter_hi.emit(0.1, np.zeros(3), np.array([0, 0, -20.0]), np.zeros(3))

        assert pool_hi.active_count > pool_lo.active_count


# ── DustManager ─────────────────────────────────────────────────────────────

class TestDustManager:
    def test_disabled_is_noop(self):
        cfg = DustConf(enable=False)
        mgr = DustManager(cfg)
        mgr.update(
            dt=0.033,
            wheel_positions=np.zeros((4, 3)),
            wheel_velocities=np.zeros((4, 3)),
            contact_forces=np.full((4, 3), -10.0),
        )
        assert mgr.active_count == 0

    def test_enabled_emits(self):
        cfg = DustConf(
            enable=True,
            force_threshold=0.1,
            emission_rate_scale=500.0,
        )
        mgr = DustManager(cfg)
        positions = np.array([
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ])
        velocities = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        forces = np.array([
            [0.0, 0.0, -5.0],
            [0.0, 0.0, -8.0],
        ])
        mgr.update(0.033, positions, velocities, forces)
        assert mgr.active_count > 0

    def test_particles_decay_over_time(self):
        cfg = DustConf(
            enable=True,
            force_threshold=0.0,
            emission_rate_scale=1000.0,
            particle_lifetime=0.5,
            opacity_decay_rate=2.0,
        )
        mgr = DustManager(cfg)

        # Emit once
        mgr.update(
            0.1,
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, -10.0]]),
        )
        count_after_emit = mgr.active_count
        assert count_after_emit > 0

        # Let particles die
        for _ in range(20):
            mgr.update(
                0.1,
                np.array([[0.0, 0.0, 1.0]]),
                np.zeros((1, 3)),
                np.zeros((1, 3)),  # No new contact force
            )
        assert mgr.active_count < count_after_emit

    def test_clear(self):
        cfg = DustConf(enable=True, force_threshold=0.0, emission_rate_scale=500.0)
        mgr = DustManager(cfg)
        mgr.update(
            0.033,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, -5.0]]),
        )
        assert mgr.active_count > 0
        mgr.clear()
        assert mgr.active_count == 0

    def test_get_particles(self):
        cfg = DustConf(enable=True, force_threshold=0.0, emission_rate_scale=500.0)
        mgr = DustManager(cfg)
        mgr.update(
            0.033,
            np.array([[0.0, 0.0, 0.5]]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, -5.0]]),
        )
        positions, opacities, count = mgr.get_particles()
        assert count > 0
        assert positions.shape == (count, 3)
        assert opacities.shape == (count,)
        assert np.all(opacities > 0)

    def test_setup_without_usd(self):
        cfg = DustConf(enable=True)
        mgr = DustManager(cfg)
        mgr.setup(stage=None)  # Should not raise

    def test_multiple_wheels(self):
        cfg = DustConf(
            enable=True,
            force_threshold=0.0,
            emission_rate_scale=200.0,
            max_emitters=4,
        )
        mgr = DustManager(cfg)
        mgr.update(
            0.033,
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
            np.ones((4, 3)),
            np.array([[0, 0, -5]] * 4, dtype=float),
        )
        assert mgr.active_count > 0
