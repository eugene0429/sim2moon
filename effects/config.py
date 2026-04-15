"""Configuration dataclasses for visual effects."""

import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class DustConf:
    """Configuration for wheel-dust particle effects.

    Controls how dust particles are emitted when rover wheels
    contact the terrain surface.
    """

    enable: bool = False

    # Emission
    force_threshold: float = 0.5         # Min contact force (N) to trigger emission
    emission_rate_scale: float = 50.0    # Particles/s per Newton of contact force
    max_particles_per_emitter: int = 500 # Max active particles per wheel
    max_emitters: int = 8                # Max simultaneous emitters (wheels)

    # Particle properties
    particle_lifetime: float = 3.0       # Seconds before particle expires
    particle_radius: float = 0.005       # Particle radius (m), small dust grains
    particle_mass: float = 1e-6          # Particle mass (kg)

    # Velocity
    kick_speed_min: float = 0.1          # Min upward ejection speed (m/s)
    kick_speed_max: float = 0.5          # Max upward ejection speed (m/s)
    lateral_spread: float = 0.3          # Lateral velocity spread factor
    velocity_scale: float = 0.5          # Scale factor for wheel velocity contribution

    # Visual
    color: Tuple[float, float, float] = (0.55, 0.52, 0.48)  # Regolith grey-brown
    opacity_initial: float = 0.6
    opacity_decay_rate: float = 0.3      # Opacity per second decay

    # Physics
    gravity: Tuple[float, float, float] = (0.0, 0.0, -1.62)  # Moon gravity
    drag_coefficient: float = 0.0        # No atmosphere on the Moon
    restitution: float = 0.1             # Bounce factor when hitting ground

    def __post_init__(self):
        assert self.force_threshold >= 0, "force_threshold must be non-negative"
        assert self.emission_rate_scale > 0, "emission_rate_scale must be positive"
        assert self.max_particles_per_emitter > 0, "max_particles_per_emitter must be positive"
        assert self.particle_lifetime > 0, "particle_lifetime must be positive"
        assert self.particle_radius > 0, "particle_radius must be positive"
        assert self.kick_speed_min >= 0, "kick_speed_min must be non-negative"
        assert self.kick_speed_max >= self.kick_speed_min, (
            "kick_speed_max must be >= kick_speed_min"
        )
        assert 0 <= self.opacity_initial <= 1, "opacity_initial must be in [0, 1]"
        assert len(self.color) == 3, "color must be (r, g, b)"
        assert len(self.gravity) == 3, "gravity must be (x, y, z)"


@dataclasses.dataclass
class StarfieldConf:
    """Configuration for the starfield sky background.

    Stars are rendered as Points on a celestial sphere. Brightness
    is modulated by sun altitude to simulate camera exposure adaptation:
    - Sun above horizon: stars invisible (camera exposed for bright surface)
    - Sun below horizon: stars at full brightness
    - Twilight zone (0-5°): gradual fade
    """

    enable: bool = True
    num_stars: int = 4000000            # Realistic: ~9k naked eye + fainter on Moon
    magnitude_limit: float = 7.5     # No atmosphere limit
    magnitude_slope: float = 0.8     # Bright star ratio (log N∝slope*m). Lower=more bright stars
    sphere_radius: float = 5000.0    # Radius of celestial sphere (m)
    base_brightness: float = 1.0     # Peak star brightness multiplier
    texture_resolution: int = 4096   # Width of lat-lon HDR texture

    # Sun-altitude thresholds for dynamic brightness
    sun_fade_start: float = 5.0      # Sun altitude (deg) where stars begin to fade
    sun_fade_end: float = 0.0        # Sun altitude (deg) where stars reach full brightness

    # Star visual
    point_size_min: float = 1.0      # Min star point size (pixels)
    point_size_max: float = 4.0      # Max star point size (brightest stars)
    color_temperature_min: float = 3000.0   # Coolest star (K) - reddish
    color_temperature_max: float = 30000.0  # Hottest star (K) - bluish

    seed: int = 42

    def __post_init__(self):
        assert self.num_stars > 0, "num_stars must be positive"
        assert self.magnitude_limit > 0, "magnitude_limit must be positive"
        assert self.magnitude_slope > 0, "magnitude_slope must be positive"
        assert self.sphere_radius > 0, "sphere_radius must be positive"
        assert self.base_brightness > 0, "base_brightness must be positive"
        assert self.sun_fade_start > self.sun_fade_end, (
            "sun_fade_start must be > sun_fade_end"
        )
        assert self.point_size_max >= self.point_size_min, (
            "point_size_max must be >= point_size_min"
        )


@dataclasses.dataclass
class EarthshineConf:
    """Configuration for earthshine (reflected Earth light).

    During lunar night, when Earth is above the horizon, it provides
    diffuse illumination. Intensity depends on:
    - Sun altitude (earthshine only matters when sun is below horizon)
    - Earth phase angle (full Earth = max earthshine)
    - Earth altitude (below horizon = no earthshine)
    """

    enable: bool = True
    base_intensity: float = 3.0        # DistantLight intensity at full-earth, zenith
    color: Tuple[float, float, float] = (0.85, 0.90, 1.0)  # Slightly blue-white
    temperature: float = 5500.0        # Color temperature (K)
    angle: float = 2.0                 # Angular diameter (deg) — Earth seen from Moon ~2°

    # Sun-altitude gating (earthshine is negligible during daytime)
    sun_threshold: float = 5.0         # Sun alt (deg) above which earthshine = 0
    sun_fade_range: float = 10.0       # Degrees of gradual fade (5 to -5)

    # Earth altitude gating
    earth_min_altitude: float = -5.0   # Earth alt (deg) below which earthshine = 0

    def __post_init__(self):
        assert self.base_intensity > 0, "base_intensity must be positive"
        assert len(self.color) == 3, "color must be (r, g, b)"
        assert self.temperature > 0, "temperature must be positive"
        assert self.sun_fade_range > 0, "sun_fade_range must be positive"
        assert self.angle > 0, "angle must be positive"
