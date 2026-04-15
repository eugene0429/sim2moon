"""
Earth visual controller for lunar simulation.

Loads the pre-built Earth.usd asset and positions it dynamically
based on StellarEngine ephemeris output.

Reference: OmniLRS src/environments/lunaryard.py (Earth rendering sections)
"""

import logging
import math
import os
from typing import Optional

from celestial.config import EarthConf
from assets import resolve_path

logger = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    import omni.usd
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


class EarthController:
    """
    Manages the Earth visual using the pre-built Earth.usd asset.

    Loads the USD asset (which has proper geometry, UVs, and OmniPBR material),
    patches the diffuse texture, and positions it based on stellar engine data.

    Interface contract:
        build(parent_path) -> None
        update_from_stellar(stellar_engine) -> None
        set_visible(flag) -> None
    """

    # OmniLRS defaults
    EARTH_USD_PATH = "assets/USD_Assets/common/Earth.usd"
    EARTH_DISTANCE = 384_400_000.0  # meters (real Earth-Moon distance)
    EARTH_SCALE = 0.001  # visual scale factor
    DEFAULT_AZIMUTH = 90.0
    DEFAULT_ELEVATION = 22.0

    def __init__(self, cfg: EarthConf) -> None:
        self._cfg = cfg
        self._earth_prim = None
        self._earth_path: Optional[str] = None
        self._scale = self.EARTH_SCALE

    def build(self, parent_path: str = "/LunarYard") -> None:
        """
        Create the Earth by loading Earth.usd and patching its texture.

        Args:
            parent_path: Parent prim path for the Earth.
        """
        if not _HAS_USD:
            logger.warning("EarthController: USD not available, skipping build")
            return

        if not self._cfg.enable:
            logger.info("EarthController: disabled by config")
            return

        stage = omni.usd.get_context().get_stage()
        self._earth_path = f"{parent_path}/Earth"

        # Load the pre-built Earth.usd asset
        earth_usd = resolve_path(self.EARTH_USD_PATH)
        self._earth_prim = stage.DefinePrim(self._earth_path, "Xform")
        self._earth_prim.GetReferences().AddReference(earth_usd)

        # Patch the diffuse texture on the existing OmniPBR material
        texture_path = resolve_path(self._cfg.texture_path)
        material_path = f"{self._earth_path}/Looks/OmniPBR"
        self._set_texture_path(stage, material_path, "Shader", texture_path)

        # Make Earth self-illuminating so it's visible during lunar night
        # (the sun still illuminates Earth in space even when below the Moon's horizon)
        self._set_emissive(stage, material_path, "Shader", texture_path)

        # Set initial position
        dist = self.EARTH_DISTANCE * self._scale
        px = math.cos(math.radians(self.DEFAULT_AZIMUTH)) * dist
        py = math.sin(math.radians(self.DEFAULT_AZIMUTH)) * dist
        pz = math.sin(math.radians(self.DEFAULT_ELEVATION)) * dist

        # Use pxr_utils to set xform ops (handles existing attribute types)
        from core.pxr_utils import addDefaultOps, setDefaultOps
        xformable = UsdGeom.Xformable(self._earth_prim)
        xformable.ClearXformOpOrder()
        # Remove pre-existing xform attributes from the referenced USD
        for attr_name in ["xformOp:translate", "xformOp:orient", "xformOp:scale"]:
            attr = self._earth_prim.GetAttribute(attr_name)
            if attr:
                self._earth_prim.RemoveProperty(attr_name)
        addDefaultOps(xformable)
        setDefaultOps(
            xformable,
            (px, py, pz),
            (0, 0, 0, 1),  # identity quaternion (x, y, z, w)
            (1.0, 1.0, 1.0),  # OmniLRS uses native USD size, only position is scaled
        )

        logger.info("EarthController: built at %s", self._earth_path)

    @staticmethod
    def _set_texture_path(stage, material_path: str, shader_name: str, texture_path: str):
        """Patch the diffuse texture on an existing material shader."""
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid():
            logger.warning("Material not found at %s", material_path)
            return
        shader = UsdShade.Shader(material_prim.GetPrimAtPath(shader_name))
        if not shader:
            logger.warning("Shader '%s' not found in material '%s'", shader_name, material_path)
            return
        file_attr = shader.GetInput("diffuse_texture")
        if file_attr:
            file_attr.Set(texture_path)
        else:
            logger.warning("Shader at '%s' has no 'diffuse_texture' input", shader.GetPath())

    @staticmethod
    def _set_emissive(stage, material_path: str, shader_name: str, texture_path: str):
        """Enable emissive on OmniPBR so Earth glows without external light.

        Uses omni.kit.commands to properly set OmniPBR emissive properties
        at runtime, since OmniPBR MDL requires runtime property registration.
        Falls back to direct USD input creation if omni.kit is unavailable.
        """
        shader_path = f"{material_path}/{shader_name}"

        try:
            import omni.kit.commands
            # Use ChangeProperty commands which properly register with OmniPBR MDL
            props = {
                "enable_emission": True,
                "emissive_intensity": 1.0,
                "emissive_color": Gf.Vec3f(1.0, 1.0, 1.0),
                "emissive_color_texture": texture_path,
            }
            for prop_name, value in props.items():
                try:
                    omni.kit.commands.execute(
                        "ChangeProperty",
                        prop_path=f"{shader_path}.inputs:{prop_name}",
                        value=value,
                        prev=None,
                    )
                except Exception:
                    pass
            logger.info("EarthController: emissive enabled via omni.kit on %s", shader_path)
        except ImportError:
            # Fallback: direct USD input creation (may not work with OmniPBR MDL)
            material_prim = stage.GetPrimAtPath(material_path)
            if not material_prim.IsValid():
                return
            shader = UsdShade.Shader(material_prim.GetPrimAtPath(shader_name))
            if not shader:
                return
            shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(1.0, 1.0, 1.0)
            )
            shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
                texture_path
            )
            logger.info("EarthController: emissive set via USD fallback on %s", shader_path)

    def update_from_stellar(self, stellar_engine) -> None:
        """
        Update Earth position from StellarEngine ephemeris.

        Args:
            stellar_engine: A StellarEngine that has been updated this frame.
        """
        if self._earth_prim is None:
            return

        alt, az, distance = stellar_engine.get_earth_alt_az()

        dist = self.EARTH_DISTANCE * self._scale
        px = math.cos(math.radians(alt)) * math.cos(math.radians(az)) * dist
        py = math.cos(math.radians(alt)) * math.sin(math.radians(az)) * dist
        pz = math.sin(math.radians(alt)) * dist

        xform_ops = UsdGeom.Xformable(self._earth_prim).GetOrderedXformOps()
        if xform_ops:
            xform_ops[0].Set(Gf.Vec3d(px, py, pz))

        # Hide Earth if below horizon
        if alt < -5.0:
            self.set_visible(False)
        else:
            self.set_visible(True)

    def set_visible(self, flag: bool) -> None:
        """Set Earth visibility."""
        if self._earth_prim is None:
            return
        imageable = UsdGeom.Imageable(self._earth_prim)
        if flag:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    @property
    def is_built(self) -> bool:
        return self._earth_prim is not None
