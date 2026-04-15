"""
Terrain material management.

Handles assignment of PBR materials (regolith, sand, etc.) to terrain meshes.
Supports multiple material presets and dynamic material switching.
"""

from typing import Dict, Optional


# Deferred USD imports for testability
try:
    import omni
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


# Built-in material presets (relative paths under assets/Textures/)
MATERIAL_PRESETS: Dict[str, str] = {
    "regolith": "LunarRegolith8k.mdl",
    "sand": "Sand.mdl",
    "gravel": "GravelStones.mdl",
}


class TerrainMaterialManager:
    """Manages terrain material assignment and switching."""

    def __init__(self, pxr_utils=None) -> None:
        """
        Args:
            pxr_utils: USD utility module (injected for testability).
        """
        self._pxr_utils = pxr_utils
        self._stage = None
        self._current_material: Optional[str] = None

    def initialize(self, stage=None) -> None:
        """Set the USD stage reference."""
        if _HAS_USD:
            self._stage = stage or omni.usd.get_context().get_stage()

    def apply_material(self, mesh_path: str, material_path: str) -> None:
        """
        Apply a material to the terrain mesh.

        Args:
            mesh_path: USD prim path of the terrain mesh.
            material_path: USD prim path of the material (e.g., "/LunarYard/Looks/Basalt").
        """
        if not _HAS_USD or self._pxr_utils is None:
            return
        self._pxr_utils.applyMaterialFromPath(self._stage, mesh_path, material_path)
        self._current_material = material_path

    @property
    def current_material(self) -> Optional[str]:
        return self._current_material
