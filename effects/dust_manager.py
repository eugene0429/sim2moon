"""Wheel-dust particle system for lunar simulation.

Emits dust particles at wheel-terrain contact points using Isaac Sim's
PhysX particle system. Particle count scales with contact force magnitude,
and particles follow lunar ballistic trajectories (no air drag).

Integration:
    dust = DustManager(cfg)
    dust.setup(stage)                  # Create particle system on USD stage
    dust.update(dt, contact_data)      # Called each physics step

Contact data format (from RobotRigidGroup):
    positions:    np.ndarray [N, 3]   wheel world positions
    velocities:   np.ndarray [N, 3]   wheel linear velocities
    forces:       np.ndarray [N, 3]   contact force vectors (z < 0 = ground contact)
"""

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from effects.config import DustConf

logger = logging.getLogger(__name__)

try:
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


@dataclass
class Particle:
    """A single dust particle with position, velocity, age, and opacity."""
    position: np.ndarray     # [3]
    velocity: np.ndarray     # [3]
    age: float = 0.0
    opacity: float = 1.0
    alive: bool = True


class ParticlePool:
    """Fixed-size particle pool with recycling to avoid allocations.

    Pre-allocates arrays for position, velocity, age, opacity.
    Dead particles are recycled by overwriting with new emissions.
    """

    def __init__(self, max_particles: int) -> None:
        self._max = max_particles
        self.positions = np.zeros((max_particles, 3), dtype=np.float32)
        self.velocities = np.zeros((max_particles, 3), dtype=np.float32)
        self.ages = np.full(max_particles, -1.0, dtype=np.float32)  # -1 = dead
        self.opacities = np.zeros(max_particles, dtype=np.float32)
        self._count = 0  # Number of active particles

    @property
    def active_count(self) -> int:
        return int(np.sum(self.ages >= 0))

    def emit(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        initial_opacity: float,
    ) -> int:
        """Add new particles, recycling dead slots.

        Args:
            positions: [N, 3] emission positions.
            velocities: [N, 3] initial velocities.
            initial_opacity: Starting opacity.

        Returns:
            Number of particles actually emitted.
        """
        n = len(positions)
        if n == 0:
            return 0

        # Find dead slots
        dead_mask = self.ages < 0
        dead_indices = np.where(dead_mask)[0]

        # Limit to available slots
        n_emit = min(n, len(dead_indices))
        if n_emit == 0:
            return 0

        slots = dead_indices[:n_emit]
        self.positions[slots] = positions[:n_emit]
        self.velocities[slots] = velocities[:n_emit]
        self.ages[slots] = 0.0
        self.opacities[slots] = initial_opacity

        return n_emit

    def step(
        self,
        dt: float,
        gravity: np.ndarray,
        lifetime: float,
        opacity_decay: float,
        ground_z: float = 0.0,
        restitution: float = 0.1,
    ) -> None:
        """Advance all active particles by dt seconds.

        Applies gravity, updates positions, fades opacity, kills expired particles.
        Particles that hit the ground bounce with restitution or die.
        """
        alive = self.ages >= 0

        if not np.any(alive):
            return

        # Physics integration (semi-implicit Euler)
        self.velocities[alive] += gravity * dt
        self.positions[alive] += self.velocities[alive] * dt

        # Age and opacity
        self.ages[alive] += dt
        self.opacities[alive] -= opacity_decay * dt
        self.opacities[alive] = np.clip(self.opacities[alive], 0.0, 1.0)

        # Ground collision
        below_ground = alive & (self.positions[:, 2] < ground_z)
        if np.any(below_ground):
            self.positions[below_ground, 2] = ground_z
            self.velocities[below_ground, 2] *= -restitution
            # Kill particles with very low bounce velocity
            slow_bounce = below_ground & (np.abs(self.velocities[:, 2]) < 0.01)
            self.ages[slow_bounce] = -1.0

        # Kill expired particles
        expired = alive & ((self.ages > lifetime) | (self.opacities <= 0.0))
        self.ages[expired] = -1.0

    def get_active(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions, opacities, count) of active particles."""
        alive = self.ages >= 0
        return self.positions[alive], self.opacities[alive], int(np.sum(alive))

    def clear(self) -> None:
        """Kill all particles."""
        self.ages[:] = -1.0


class DustEmitter:
    """Manages dust emission for a single wheel contact point."""

    def __init__(
        self,
        name: str,
        pool: ParticlePool,
        cfg: DustConf,
        rng: np.random.Generator,
    ) -> None:
        self._name = name
        self._pool = pool
        self._cfg = cfg
        self._rng = rng
        self._residual = 0.0  # Sub-frame emission accumulator

    def emit(
        self,
        dt: float,
        contact_position: np.ndarray,
        contact_force: np.ndarray,
        wheel_velocity: np.ndarray,
    ) -> int:
        """Emit dust particles based on contact force magnitude.

        Args:
            dt: Time step (seconds).
            contact_position: [3] world position of wheel contact.
            contact_force: [3] contact force vector.
            wheel_velocity: [3] wheel linear velocity.

        Returns:
            Number of particles emitted.
        """
        force_mag = np.linalg.norm(contact_force)
        if force_mag < self._cfg.force_threshold:
            self._residual = 0.0
            return 0

        # Compute emission count (force-proportional)
        rate = (force_mag - self._cfg.force_threshold) * self._cfg.emission_rate_scale
        self._residual += rate * dt
        n = int(self._residual)
        self._residual -= n

        if n <= 0:
            return 0

        # Clamp to pool capacity
        n = min(n, self._cfg.max_particles_per_emitter - self._pool.active_count)
        if n <= 0:
            return 0

        # Generate initial velocities
        cfg = self._cfg

        # Upward kick (lunar ballistic)
        kick_z = self._rng.uniform(cfg.kick_speed_min, cfg.kick_speed_max, n)

        # Lateral spread from wheel velocity direction
        wheel_speed = np.linalg.norm(wheel_velocity[:2])
        lateral_x = self._rng.normal(0, cfg.lateral_spread, n)
        lateral_y = self._rng.normal(0, cfg.lateral_spread, n)

        # Add fraction of wheel velocity (thrown backward)
        vx = -wheel_velocity[0] * cfg.velocity_scale + lateral_x
        vy = -wheel_velocity[1] * cfg.velocity_scale + lateral_y
        vz = kick_z

        velocities = np.column_stack([vx, vy, vz]).astype(np.float32)

        # Slight position jitter around contact point
        jitter = self._rng.normal(0, cfg.particle_radius * 5, (n, 3)).astype(np.float32)
        jitter[:, 2] = np.abs(jitter[:, 2])  # Keep above ground
        positions = (contact_position + jitter).astype(np.float32)

        return self._pool.emit(positions, velocities, cfg.opacity_initial)


class DustManager:
    """Top-level dust effect manager.

    Creates a particle pool and per-wheel emitters. Each physics step,
    feed contact data from the robot rigid body group to emit and
    simulate dust particles.

    Usage:
        manager = DustManager(cfg)
        manager.setup(stage)    # Optional: create USD visualization
        manager.update(dt, wheel_positions, wheel_velocities, contact_forces)
        positions, opacities, count = manager.get_particles()
    """

    def __init__(self, cfg: DustConf, seed: int = 42) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(seed)
        total_particles = cfg.max_particles_per_emitter * cfg.max_emitters
        self._pool = ParticlePool(total_particles)
        self._emitters: Dict[int, DustEmitter] = {}
        self._gravity = np.array(cfg.gravity, dtype=np.float32)
        self._usd_prim = None

    def setup(self, stage=None, root_path: str = "/Effects/Dust") -> None:
        """Initialize USD particle visualization (optional).

        Creates a UsdGeom.Points prim to render particles.
        If USD/Isaac Sim is unavailable, the manager still works
        as a pure-numpy particle simulation.
        """
        if not _HAS_USD or stage is None:
            logger.info("DustManager: running in CPU-only mode (no USD visualization)")
            return

        # Create Points prim for particle rendering
        points_path = f"{root_path}/particles"
        stage.DefinePrim(root_path, "Xform")
        self._usd_prim = UsdGeom.Points.Define(stage, points_path)
        self._usd_prim.CreateWidthsAttr()
        self._usd_prim.CreateDisplayColorAttr()
        self._usd_prim.CreateDisplayOpacityAttr()
        logger.info("DustManager: USD Points prim created at %s", points_path)

    def update(
        self,
        dt: float,
        wheel_positions: np.ndarray,
        wheel_velocities: np.ndarray,
        contact_forces: np.ndarray,
    ) -> None:
        """Update dust simulation for one physics step.

        Args:
            dt: Physics time step (seconds).
            wheel_positions: [N, 3] world positions of wheel contact points.
            wheel_velocities: [N, 3] linear velocities of wheels.
            contact_forces: [N, 3] contact force vectors per wheel.
        """
        if not self._cfg.enable:
            return

        n_wheels = len(wheel_positions)

        # Emit from each wheel
        for i in range(min(n_wheels, self._cfg.max_emitters)):
            if i not in self._emitters:
                self._emitters[i] = DustEmitter(
                    name=f"wheel_{i}",
                    pool=self._pool,
                    cfg=self._cfg,
                    rng=self._rng,
                )
            self._emitters[i].emit(
                dt,
                wheel_positions[i],
                contact_forces[i],
                wheel_velocities[i],
            )

        # Simulate particles
        ground_z = 0.0
        if n_wheels > 0:
            ground_z = float(np.min(wheel_positions[:, 2])) - 0.01

        self._pool.step(
            dt,
            self._gravity,
            self._cfg.particle_lifetime,
            self._cfg.opacity_decay_rate,
            ground_z=ground_z,
            restitution=self._cfg.restitution,
        )

        # Update USD visualization
        self._update_usd()

    def get_particles(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return active particle data.

        Returns:
            (positions [M, 3], opacities [M], active_count)
        """
        return self._pool.get_active()

    def clear(self) -> None:
        """Kill all particles and clear emitters."""
        self._pool.clear()
        self._emitters.clear()

    @property
    def active_count(self) -> int:
        return self._pool.active_count

    def _update_usd(self) -> None:
        """Push particle positions/opacities to USD Points prim."""
        if self._usd_prim is None:
            return

        positions, opacities, count = self._pool.get_active()
        if count == 0:
            self._usd_prim.GetPointsAttr().Set([])
            return

        # Convert to USD types
        points = [Gf.Vec3f(*p) for p in positions]
        widths = [self._cfg.particle_radius * 2] * count
        color = self._cfg.color
        colors = [Gf.Vec3f(*color)] * count

        self._usd_prim.GetPointsAttr().Set(points)
        self._usd_prim.GetWidthsAttr().Set(widths)
        self._usd_prim.GetDisplayColorAttr().Set(colors)
        self._usd_prim.GetDisplayOpacityAttr().Set(opacities.tolist())
