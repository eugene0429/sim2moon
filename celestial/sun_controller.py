"""
Sun light controller for lunar simulation.

Manages a USD DistantLight prim representing the sun, updating its
orientation and intensity based on StellarEngine ephemeris output.

Uses the same two-level Xform hierarchy as OmniLRS:
  - Parent Xform: receives alt/az orientation quaternion
  - Child DistantLight: has a fixed pre-rotation to align the light direction

Reference: OmniLRS src/environments/lunaryard.py (sun management sections)
"""

import logging
import math
from typing import Optional, Tuple

from celestial.config import SunConf

logger = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    import omni.usd
    _HAS_USD = True
except ImportError:
    _HAS_USD = False

# Fixed pre-rotation for DistantLight to align its -Z axis correctly.
# This is the same hardcoded quaternion OmniLRS uses on the inner light prim.
_LIGHT_PRE_ROTATION = (0.5, 0.5, -0.5, -0.5)  # (w, x, y, z)


class SunController:
    """
    Manages a USD DistantLight representing the sun.

    Creates a parent Xform + child DistantLight during build().
    The parent Xform receives alt/az orientation updates.
    The child DistantLight has a fixed pre-rotation.

    Interface contract:
        build(parent_path) -> None
        update_from_stellar(stellar_engine) -> None
        set_intensity(value) -> None
        set_color(r, g, b) -> None
    """

    def __init__(self, cfg: SunConf) -> None:
        self._cfg = cfg
        self._sun_xform = None   # parent Xform prim (receives alt/az)
        self._sun_light = None   # child DistantLight prim (fixed pre-rotation)

    def build(self, parent_path: str = "/LunarYard") -> None:
        """
        Create the sun Xform + DistantLight prims in the USD stage.

        Args:
            parent_path: Parent prim path for the sun.
        """
        if not _HAS_USD:
            logger.warning("SunController: USD not available, skipping build")
            return

        stage = omni.usd.get_context().get_stage()
        sun_path = f"{parent_path}/Sun"

        # Parent Xform — receives alt/az rotation
        self._sun_xform = stage.DefinePrim(sun_path, "Xform")
        xform = UsdGeom.Xformable(self._sun_xform)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(1, 0, 0, 0)
        )
        xform.AddScaleOp().Set(Gf.Vec3d(1, 1, 1))

        # Child DistantLight — fixed pre-rotation
        light_path = f"{sun_path}/sun"
        self._sun_light = UsdLux.DistantLight.Define(stage, light_path)
        self._sun_light.CreateIntensityAttr(self._cfg.intensity)
        self._sun_light.CreateAngleAttr(self._cfg.angle)
        self._sun_light.CreateColorAttr(Gf.Vec3f(*self._cfg.color))
        self._sun_light.CreateColorTemperatureAttr(self._cfg.temperature)
        self._sun_light.CreateDiffuseAttr(self._cfg.diffuse_multiplier)
        self._sun_light.CreateSpecularAttr(self._cfg.specular_multiplier)

        # Apply fixed pre-rotation to the light prim
        light_prim = self._sun_light.GetPrim()
        light_xform = UsdGeom.Xformable(light_prim)
        light_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        light_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(
                _LIGHT_PRE_ROTATION[0],
                Gf.Vec3d(_LIGHT_PRE_ROTATION[1], _LIGHT_PRE_ROTATION[2], _LIGHT_PRE_ROTATION[3]),
            )
        )
        light_xform.AddScaleOp().Set(Gf.Vec3d(1, 1, 1))

        # Set initial orientation from config azimuth/elevation
        self._set_orientation_from_alt_az(self._cfg.elevation, self._cfg.azimuth)
        logger.info("SunController: built at %s", sun_path)

    def update_from_stellar(self, stellar_engine) -> None:
        """
        Update sun position from a StellarEngine instance.

        Args:
            stellar_engine: A StellarEngine that has been updated this frame.
        """
        if self._sun_xform is None:
            return

        alt, az, distance = stellar_engine.get_sun_alt_az()
        self._set_orientation_from_alt_az(alt, az)

        # Intensity stays constant — the DistantLight orientation naturally
        # handles sunrise/sunset. When the sun is below the horizon, the light
        # direction points upward from below, so it doesn't illuminate the
        # terrain surface. No artificial intensity cutoff needed.

    def set_intensity(self, value: float) -> None:
        """Set the sun light intensity directly."""
        if self._sun_light is not None:
            self._sun_light.GetIntensityAttr().Set(value)

    def set_color(self, r: float, g: float, b: float) -> None:
        """Set the sun light color."""
        if self._sun_light is not None:
            self._sun_light.GetColorAttr().Set(Gf.Vec3f(r, g, b))

    def set_orientation(self, quat_wxyz: Tuple[float, float, float, float]) -> None:
        """
        Set the sun orientation directly from a quaternion (w, x, y, z).
        Applied to the parent Xform.

        Args:
            quat_wxyz: Quaternion in (w, x, y, z) order.
        """
        if self._sun_xform is None:
            return
        w, x, y, z = quat_wxyz
        ops = UsdGeom.Xformable(self._sun_xform).GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
                break

    def _set_orientation_from_alt_az(self, alt: float, az: float) -> None:
        """Set parent Xform orientation from altitude/azimuth."""
        if self._sun_xform is None:
            return

        from celestial.stellar_engine import StellarEngine
        w, x, y, z = StellarEngine.convert_alt_az_to_quat(alt, az)

        ops = UsdGeom.Xformable(self._sun_xform).GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
                break

    @property
    def is_built(self) -> bool:
        return self._sun_xform is not None
