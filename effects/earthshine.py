"""Earthshine illumination for lunar night scenes.

During lunar night, the Earth reflects sunlight onto the Moon's surface.
This module computes the earthshine intensity based on:
- Sun altitude (earthshine only matters at night)
- Earth altitude (must be above horizon)
- Earth phase angle (full Earth = max illumination)

The earthshine is rendered as a DistantLight oriented toward the Earth's
position, using the same two-level Xform hierarchy as the SunController:
  - Parent Xform: receives alt/az orientation quaternion
  - Child DistantLight: has a fixed pre-rotation to align the light direction

This avoids the inverse-square falloff problem of SphereLight, which made
earthshine invisible at the distances involved.

Usage:
    earthshine = Earthshine(cfg)
    earthshine.setup(stage)
    earthshine.update(sun_alt, earth_alt, earth_az, sun_az)
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

from effects.config import EarthshineConf

logger = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


# Fixed pre-rotation for DistantLight (same as SunController)
_LIGHT_PRE_ROTATION = (0.5, 0.5, -0.5, -0.5)  # (w, x, y, z)


class Earthshine:
    """Manages earthshine illumination during lunar night.

    Creates a DistantLight oriented toward the Earth's position that
    activates when the sun is below the horizon and the Earth is visible.
    Intensity is modulated by Earth phase angle.

    Uses the same two-level Xform hierarchy as SunController:
      - Parent Xform: receives alt/az orientation quaternion
      - Child DistantLight: fixed pre-rotation to align light direction
    """

    def __init__(self, cfg: EarthshineConf) -> None:
        self._cfg = cfg
        self._earth_xform = None
        self._light_prim = None
        self._current_intensity = 0.0

    def setup(self, stage=None, root_path: str = "/Effects/Earthshine") -> None:
        """Create USD DistantLight for earthshine with two-level Xform.

        Args:
            stage: USD stage. If None, runs in compute-only mode.
            root_path: USD prim path.
        """
        if not _HAS_USD or stage is None:
            logger.info("Earthshine: compute-only mode (no USD)")
            return

        if not self._cfg.enable:
            logger.info("Earthshine: disabled by config")
            return

        stage.DefinePrim("/Effects", "Xform")

        # Parent Xform — receives alt/az rotation
        self._earth_xform = stage.DefinePrim(root_path, "Xform")
        xform = UsdGeom.Xformable(self._earth_xform)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(1, 0, 0, 0)
        )
        xform.AddScaleOp().Set(Gf.Vec3d(1, 1, 1))

        # Child DistantLight — fixed pre-rotation
        light_path = f"{root_path}/light"
        self._light_prim = UsdLux.DistantLight.Define(stage, light_path)
        self._light_prim.CreateIntensityAttr().Set(0.0)
        self._light_prim.CreateAngleAttr().Set(self._cfg.angle)
        self._light_prim.CreateColorAttr().Set(Gf.Vec3f(*self._cfg.color))
        self._light_prim.CreateColorTemperatureAttr().Set(self._cfg.temperature)
        self._light_prim.CreateEnableColorTemperatureAttr().Set(True)
        self._light_prim.CreateDiffuseAttr().Set(1.0)
        self._light_prim.CreateSpecularAttr().Set(0.2)

        # Apply fixed pre-rotation to the light prim
        light_prim = self._light_prim.GetPrim()
        light_xform = UsdGeom.Xformable(light_prim)
        light_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        light_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(
                _LIGHT_PRE_ROTATION[0],
                Gf.Vec3d(_LIGHT_PRE_ROTATION[1], _LIGHT_PRE_ROTATION[2], _LIGHT_PRE_ROTATION[3]),
            )
        )
        light_xform.AddScaleOp().Set(Gf.Vec3d(1, 1, 1))

        logger.info("Earthshine: DistantLight created at %s", root_path)

    def update(
        self,
        sun_altitude_deg: float,
        earth_altitude_deg: float,
        earth_azimuth_deg: float,
        sun_azimuth_deg: float = 0.0,
    ) -> float:
        """Update earthshine intensity and position.

        Args:
            sun_altitude_deg: Sun altitude (degrees). Negative = below horizon.
            earth_altitude_deg: Earth altitude (degrees).
            earth_azimuth_deg: Earth azimuth (degrees).
            sun_azimuth_deg: Sun azimuth (degrees), for phase angle computation.

        Returns:
            Current earthshine intensity (0 = off).
        """
        if not self._cfg.enable:
            self._current_intensity = 0.0
            return 0.0

        intensity = self.compute_intensity(
            sun_altitude_deg, earth_altitude_deg,
            earth_azimuth_deg, sun_azimuth_deg,
        )
        self._current_intensity = intensity

        if self._light_prim is not None:
            self._update_usd(intensity, earth_altitude_deg, earth_azimuth_deg)

        return intensity

    def compute_intensity(
        self,
        sun_alt: float,
        earth_alt: float,
        earth_az: float,
        sun_az: float,
    ) -> float:
        """Compute earthshine intensity from celestial geometry.

        Factors:
        1. Sun gate: earthshine fades as sun rises
        2. Earth visibility: must be above min altitude
        3. Earth phase: derived from sun-earth angular separation

        Args:
            sun_alt: Sun altitude (degrees).
            earth_alt: Earth altitude (degrees).
            earth_az: Earth azimuth (degrees).
            sun_az: Sun azimuth (degrees).

        Returns:
            Earthshine intensity (lux-like units, 0 = off).
        """
        if not self._cfg.enable:
            return 0.0

        cfg = self._cfg

        # 1. Sun gating: full earthshine when sun is well below horizon
        sun_threshold = cfg.sun_threshold
        sun_fade = cfg.sun_fade_range

        if sun_alt >= sun_threshold:
            return 0.0
        elif sun_alt <= sun_threshold - sun_fade:
            sun_factor = 1.0
        else:
            sun_factor = (sun_threshold - sun_alt) / sun_fade

        # 2. Earth visibility
        if earth_alt < cfg.earth_min_altitude:
            return 0.0

        # Earth altitude factor: higher = brighter (cosine law)
        earth_factor = max(0.0, math.sin(math.radians(max(0.0, earth_alt))))

        # 3. Earth phase angle
        # Phase angle = angular separation between sun and earth as seen from Moon
        # When sun and earth are on opposite sides: full Earth (phase ~ 180°)
        # When same side: new Earth (phase ~ 0°)
        phase_angle = self._compute_phase_angle(sun_alt, sun_az, earth_alt, earth_az)
        phase_factor = self._phase_to_illumination(phase_angle)

        intensity = cfg.base_intensity * sun_factor * earth_factor * phase_factor
        return max(0.0, intensity)

    @staticmethod
    def _compute_phase_angle(
        sun_alt: float, sun_az: float,
        earth_alt: float, earth_az: float,
    ) -> float:
        """Compute angular separation between sun and earth positions.

        Uses the spherical law of cosines for angular distance.

        Returns:
            Phase angle in degrees (0-180).
        """
        # Convert to radians
        sa = math.radians(sun_alt)
        saz = math.radians(sun_az)
        ea = math.radians(earth_alt)
        eaz = math.radians(earth_az)

        # Spherical angular distance
        cos_angle = (
            math.sin(sa) * math.sin(ea)
            + math.cos(sa) * math.cos(ea) * math.cos(saz - eaz)
        )
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def _phase_to_illumination(phase_angle_deg: float) -> float:
        """Convert phase angle to illuminated fraction of Earth as seen from Moon.

        Phase angle = angular separation between sun and Earth as seen from Moon.
        - phase ~0°: sun and Earth in same direction -> sun behind Earth -> new Earth (dark)
        - phase ~180°: sun and Earth opposite -> sun illuminates Earth's Moon-facing side -> full Earth

        Uses Lambertian approximation: illumination = (1 - cos(phase)) / 2

        Args:
            phase_angle_deg: Phase angle in degrees.

        Returns:
            Illumination fraction [0, 1].
        """
        return (1.0 - math.cos(math.radians(phase_angle_deg))) / 2.0

    def _update_usd(
        self,
        intensity: float,
        earth_alt: float,
        earth_az: float,
    ) -> None:
        """Update USD light intensity and orientation toward Earth."""
        self._light_prim.GetIntensityAttr().Set(intensity)

        # Orient parent Xform toward the Earth (same method as SunController)
        if self._earth_xform is not None:
            self._set_orientation_from_alt_az(earth_alt, earth_az)

    def _set_orientation_from_alt_az(self, alt: float, az: float) -> None:
        """Set parent Xform orientation from altitude/azimuth.

        Uses the same StellarEngine.convert_alt_az_to_quat as SunController
        to ensure consistent light direction encoding.
        """
        from celestial.stellar_engine import StellarEngine
        w, x, y, z = StellarEngine.convert_alt_az_to_quat(alt, az)

        ops = UsdGeom.Xformable(self._earth_xform).GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
                break

    # ── Accessors ───────────────────────────────────────────────────────

    @property
    def current_intensity(self) -> float:
        return self._current_intensity

    @property
    def is_active(self) -> bool:
        return self._current_intensity > 0.0
