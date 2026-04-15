"""
Static asset manager for lunar simulation.

Manages placement of non-rock static objects (props, structures,
equipment) in the simulation environment.

Reference: OmniLRS environments use direct USD prim creation for
static assets like projectors, curtains, etc.
"""

import logging
from typing import Dict, List, Optional, Tuple

from objects.config import StaticAssetConf

logger = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, UsdGeom
    import omni.usd
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


class StaticAssetManager:
    """
    Manages static (non-rock) assets in the scene.

    Each asset is loaded from a USD file and placed at a configured
    position/orientation/scale. Assets can be toggled visible/invisible.

    Interface contract:
        add_asset(name, cfg) -> None
        build() -> None
        set_visible(name, flag) -> None
        set_all_visible(flag) -> None
    """

    def __init__(self, parent_path: str = "/LunarYard/Props") -> None:
        self._parent_path = parent_path
        self._assets: Dict[str, StaticAssetConf] = {}
        self._prims: Dict[str, object] = {}

    def add_asset(self, name: str, cfg: StaticAssetConf) -> None:
        """
        Register a static asset for placement.

        Args:
            name: Unique name for the asset.
            cfg: Asset configuration (USD path, position, orientation, scale).
        """
        self._assets[name] = cfg

    def build(self) -> None:
        """Place all registered assets in the USD stage."""
        if not _HAS_USD:
            logger.warning("StaticAssetManager: USD not available, skipping build")
            return

        stage = omni.usd.get_context().get_stage()
        stage.DefinePrim(self._parent_path, "Xform")

        for name, cfg in self._assets.items():
            prim_path = cfg.prim_path or f"{self._parent_path}/{name}"

            # Add reference to the USD asset file
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(cfg.usd_path)

            # Set transform
            xformable = UsdGeom.Xformable(prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(*cfg.position))
            w, x, y, z = cfg.orientation
            xformable.AddOrientOp().Set(Gf.Quatd(w, x, y, z))
            xformable.AddScaleOp().Set(Gf.Vec3d(*cfg.scale))

            self._prims[name] = prim
            logger.info("StaticAssetManager: placed '%s' at %s", name, prim_path)

    def set_visible(self, name: str, flag: bool) -> None:
        """Set visibility of a specific asset."""
        if name in self._prims:
            self._prims[name].GetAttribute("visibility").Set(
                "inherited" if flag else "invisible"
            )

    def set_all_visible(self, flag: bool) -> None:
        """Set visibility of all assets."""
        for name in self._prims:
            self.set_visible(name, flag)

    def get_asset_names(self) -> List[str]:
        """Return list of registered asset names."""
        return list(self._assets.keys())
