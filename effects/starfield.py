"""Procedural starfield for lunar simulation.

Generates a celestial sphere of stars with realistic:
- Spatial distribution (uniform on sphere)
- Magnitude-based brightness (power-law, more faint stars)
- Color temperature (spectral class approximation)
- Dynamic brightness modulated by sun altitude (exposure adaptation)

Stars are rendered as a DomeLight with a procedurally generated HDR texture.
UsdGeom.Points do not render in RTX mode, so DomeLight is the reliable approach.

Usage:
    starfield = Starfield(cfg)
    starfield.generate()                          # Create star catalog + texture
    starfield.setup(stage)                        # Create USD DomeLight
    starfield.update(sun_altitude_deg)            # Adjust brightness
"""

import logging
import math
import os
import tempfile
from typing import Optional, Tuple

import numpy as np

from effects.config import StarfieldConf

logger = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


def _temperature_to_rgb(temp_k: float) -> Tuple[float, float, float]:
    """Convert star color temperature to approximate RGB (0-1 range).

    Uses Tanner Helland's algorithm for blackbody color approximation.
    """
    temp = max(1000.0, min(40000.0, temp_k)) / 100.0

    if temp <= 66:
        r = 1.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592) / 255.0

    if temp <= 66:
        g = (99.4708025861 * math.log(temp) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492) / 255.0

    if temp >= 66:
        b = 1.0
    elif temp <= 19:
        b = 0.0
    else:
        b = (138.5177312231 * math.log(temp - 10) - 305.0447927307) / 255.0

    return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))


class Starfield:
    """Procedural starfield rendered as a DomeLight with HDR texture.

    The star catalog is generated once and baked into a lat-lon HDR texture.
    The DomeLight intensity is modulated by sun altitude each frame.
    """

    def __init__(self, cfg: StarfieldConf) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        # Star catalog
        self._positions: Optional[np.ndarray] = None
        self._magnitudes: Optional[np.ndarray] = None
        self._colors: Optional[np.ndarray] = None

        # USD
        self._dome_light = None
        self._texture_path: Optional[str] = None
        self._current_brightness = 0.0

    def generate(self) -> None:
        """Generate a procedural star catalog."""
        n = self._cfg.num_stars
        rng = self._rng

        # Uniform distribution on sphere
        u = rng.uniform(-1, 1, n)
        theta = rng.uniform(0, 2 * np.pi, n)

        x = np.sqrt(1 - u**2) * np.cos(theta)
        y = np.sqrt(1 - u**2) * np.sin(theta)
        z = u
        self._positions = np.column_stack([x, y, z]).astype(np.float32)

        # Magnitude distribution (slope controls bright/faint ratio)
        mag_min, mag_max = -1.5, self._cfg.magnitude_limit
        slope = self._cfg.magnitude_slope
        u_mag = rng.uniform(0, 1, n)
        a = 10 ** (slope * mag_min)
        b = 10 ** (slope * mag_max)
        self._magnitudes = (np.log10(a + u_mag * (b - a)) / slope).astype(np.float32)

        # Flux from magnitude
        flux = 10 ** (-0.4 * self._magnitudes)
        flux_max = 10 ** (-0.4 * mag_min)
        brightnesses = (flux / flux_max).astype(np.float32)

        # Color temperature
        mag_norm = (self._magnitudes - mag_min) / (mag_max - mag_min)
        t_min, t_max = self._cfg.color_temperature_min, self._cfg.color_temperature_max
        temperatures = t_max - mag_norm * (t_max - t_min)
        temperatures += rng.normal(0, 1000, n)
        temperatures = np.clip(temperatures, t_min, t_max)

        colors = np.array([_temperature_to_rgb(t) for t in temperatures], dtype=np.float32)
        # Scale colors by brightness
        self._colors = colors * brightnesses[:, np.newaxis]

        logger.info("Starfield: generated %d stars (mag %.1f to %.1f)",
                     n, self._magnitudes.min(), self._magnitudes.max())

    def _generate_texture(self) -> str:
        """Bake stars into a lat-lon HDR texture (EXR or HDR format).

        Returns:
            Path to the generated texture file.
        """
        width = getattr(self._cfg, 'texture_resolution', 4096)
        height = width // 2
        # Create black HDR image
        img = np.zeros((height, width, 3), dtype=np.float32)

        # Physical flux: 10^(-0.4 * mag), normalized so mag=-1.5 → 1.0
        mag_min = -1.5
        flux = 10 ** (-0.4 * self._magnitudes)
        flux_max = 10 ** (-0.4 * mag_min)
        brightnesses = flux / flux_max  # [0..1]

        # Recover base color (undo brightness bake-in from generate())
        base_colors = self._colors / (brightnesses[:, np.newaxis] + 1e-10)
        base_colors = np.clip(base_colors, 0, 1)

        for i in range(len(self._positions)):
            x, y, z = self._positions[i]
            br = float(brightnesses[i])
            cr, cg, cb = base_colors[i]
            mag = float(self._magnitudes[i])

            # Convert unit sphere → lat-lon pixel coords
            lat = math.asin(np.clip(z, -1, 1))
            lon = math.atan2(y, x)
            px = int((lon / (2 * math.pi) + 0.5) * width) % width
            py = int((0.5 - lat / math.pi) * height)
            py = max(0, min(height - 1, py))

            # Physical HDR value: proportional to flux
            # Scale so brightest stars (mag ~ -1) are very bright in HDR,
            # faintest (mag ~ 7.5) are barely visible single pixels
            hdr_value = 30.0 * br  # linear in flux, bright star ~30, faint ~0.003

            # Only the ~brightest stars (mag < 2) get multi-pixel glow
            if mag < 0:
                radius = 2
            elif mag < 2:
                radius = 1
            else:
                radius = 0  # single pixel — most stars

            if radius == 0:
                # Single pixel stamp — fast path
                img[py, px, 0] += cr * hdr_value
                img[py, px, 1] += cg * hdr_value
                img[py, px, 2] += cb * hdr_value
            else:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ppx = (px + dx) % width
                        ppy = max(0, min(height - 1, py + dy))
                        dist_sq = dx * dx + dy * dy
                        falloff = math.exp(-dist_sq * 1.2)
                        img[ppy, ppx, 0] += cr * hdr_value * falloff
                        img[ppy, ppx, 1] += cg * hdr_value * falloff
                        img[ppy, ppx, 2] += cb * hdr_value * falloff

        # Save as HDR file
        texture_dir = os.path.join(tempfile.gettempdir(), "lunar_sim_starfield")
        os.makedirs(texture_dir, exist_ok=True)
        texture_path = os.path.join(texture_dir, "starfield.hdr")

        # Write Radiance HDR format
        self._write_hdr(texture_path, img)
        logger.info("Starfield texture saved to %s (%dx%d)", texture_path, width, height)
        return texture_path

    @staticmethod
    def _write_hdr(path: str, img: np.ndarray) -> None:
        """Write an HDR (Radiance RGBE) image file."""
        try:
            import cv2
            # OpenCV can write HDR
            cv2.imwrite(path, img)
        except ImportError:
            # Fallback: write as raw numpy, use .npy extension
            npy_path = path.replace(".hdr", ".npy")
            np.save(npy_path, img)
            logger.warning("cv2 not available, saved raw numpy to %s", npy_path)

    def setup(self, stage=None, root_path: str = "/Effects/Starfield") -> None:
        """Create USD DomeLight with star texture.

        Args:
            stage: USD stage. If None, runs in CPU-only mode.
            root_path: USD prim path for the starfield.
        """
        if self._positions is None:
            self.generate()

        if not _HAS_USD or stage is None:
            logger.info("Starfield: CPU-only mode (no USD visualization)")
            return

        if not self._cfg.enable:
            logger.info("Starfield: disabled by config")
            return

        # Generate texture
        self._texture_path = self._generate_texture()

        # Create DomeLight
        stage.DefinePrim("/Effects", "Xform")
        self._dome_light = UsdLux.DomeLight.Define(stage, root_path)
        self._dome_light.CreateIntensityAttr().Set(0.0)
        self._dome_light.CreateTextureFileAttr().Set(self._texture_path)
        self._dome_light.CreateTextureFormatAttr().Set("latlong")

        # Set initial brightness
        initial_brightness = self._cfg.base_brightness
        self._dome_light.GetIntensityAttr().Set(initial_brightness * 500.0)
        self._current_brightness = initial_brightness

        logger.info("Starfield: DomeLight created at %s", root_path)

    def update(self, sun_altitude_deg: float) -> float:
        """Update star brightness based on sun altitude.

        Args:
            sun_altitude_deg: Sun altitude in degrees (negative = below horizon).

        Returns:
            Current brightness multiplier [0, 1].
        """
        if not self._cfg.enable:
            return 0.0

        brightness = self.compute_brightness(sun_altitude_deg)
        self._current_brightness = brightness

        if self._dome_light is not None:
            # Scale DomeLight intensity (500 lux at full brightness is subtle fill)
            self._dome_light.GetIntensityAttr().Set(brightness * 500.0)

        return brightness

    def compute_brightness(self, sun_altitude_deg: float) -> float:
        """Compute star brightness multiplier from sun altitude."""
        fade_start = self._cfg.sun_fade_start
        fade_end = self._cfg.sun_fade_end

        if sun_altitude_deg >= fade_start:
            return 0.0
        elif sun_altitude_deg <= fade_end:
            return self._cfg.base_brightness
        else:
            t = (fade_start - sun_altitude_deg) / (fade_start - fade_end)
            return t * self._cfg.base_brightness

    # ── Accessors ───────────────────────────────────────────────────────

    @property
    def positions(self) -> Optional[np.ndarray]:
        return self._positions

    @property
    def magnitudes(self) -> Optional[np.ndarray]:
        return self._magnitudes

    @property
    def colors(self) -> Optional[np.ndarray]:
        return self._colors

    @property
    def current_brightness(self) -> float:
        return self._current_brightness

    @property
    def star_count(self) -> int:
        return len(self._positions) if self._positions is not None else 0
