"""
Terrain mesh builder for USD scenes.

Converts a DEM (2D numpy height array) into a USD mesh with vertices,
triangle indices, UVs, collision, and semantic labeling.

This module encapsulates all USD mesh operations, keeping them cleanly
separated from DEM generation logic.
"""

from typing import Optional, Tuple

import numpy as np

# Isaac Sim / USD imports are deferred to allow unit testing without Omniverse
_HAS_USD = False
UsdGeom = None
Sdf = None
omni = None


class TerrainMeshBuilder:
    """
    Builds and updates a USD mesh from a DEM.

    Owns the vertex grid, triangle topology, and UVs. Provides methods
    to render/update the mesh and manage collision.
    """

    def __init__(
        self,
        sim_width_px: int,
        sim_length_px: int,
        grid_size: float,
        root_path: str,
        mesh_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        mesh_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pxr_utils=None,
    ) -> None:
        """
        Args:
            sim_width_px: Number of grid points in X.
            sim_length_px: Number of grid points in Y.
            grid_size: Spacing between grid points in meters.
            root_path: USD path root for terrain prims.
            mesh_position: World position of the mesh.
            mesh_orientation: World orientation (quaternion xyzw).
            mesh_scale: World scale.
            pxr_utils: USD utility module (injected for testability).
        """
        self._sim_width = sim_width_px
        self._sim_length = sim_length_px
        self._grid_size = grid_size
        self._root_path = root_path
        self._mesh_pos = mesh_position
        self._mesh_rot = mesh_orientation
        self._mesh_scale = mesh_scale
        self._pxr_utils = pxr_utils

        self._og_mesh_path = f"{root_path}/Terrain/terrain_mesh"
        self._mesh_path = self._og_mesh_path
        self._id = 0

        # Build grid topology once
        self._vertices, self._indices, self._uvs = self._build_grid()
        self._stage = None

    def _build_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build vertex positions, triangle indices, and UV coordinates.

        Returns:
            Tuple of (vertices [N*M, 3], indices [F*3], uvs [F*3, 2]).
        """
        w, h = self._sim_width, self._sim_length

        # Vertex positions: regular grid with Z=0
        xs = np.arange(w, dtype=np.float32) * self._grid_size
        ys = np.arange(h, dtype=np.float32) * self._grid_size
        gx, gy = np.meshgrid(xs, ys)  # both (h, w)
        verts = np.zeros((h * w, 3), dtype=np.float32)
        verts[:, 0] = gx.ravel()
        verts[:, 1] = gy.ravel()

        # Triangle indices: two triangles per interior quad
        # For each cell (x, y) where x in [1, w) and y in [1, h):
        #   v00 = (y-1)*w + (x-1),  v10 = (y-1)*w + x
        #   v01 = y*w + (x-1),      v11 = y*w + x
        #   Tri1: v10, v11, v00
        #   Tri2: v11, v01, v00
        cx = np.arange(1, w, dtype=np.int32)
        cy = np.arange(1, h, dtype=np.int32)
        gix, giy = np.meshgrid(cx, cy)
        gix = gix.ravel()
        giy = giy.ravel()

        v00 = (giy - 1) * w + (gix - 1)
        v10 = (giy - 1) * w + gix
        v01 = giy * w + (gix - 1)
        v11 = giy * w + gix

        n_quads = len(gix)
        indices = np.empty(n_quads * 6, dtype=np.int32)
        indices[0::6] = v10
        indices[1::6] = v11
        indices[2::6] = v00
        indices[3::6] = v11
        indices[4::6] = v01
        indices[5::6] = v00

        # UV coordinates (faceVarying — one per face-vertex)
        uvs = np.empty((n_quads * 6, 2), dtype=np.float32)
        uvs[0::6, 0] = gix;   uvs[0::6, 1] = giy - 1
        uvs[1::6, 0] = gix;   uvs[1::6, 1] = giy
        uvs[2::6, 0] = gix - 1; uvs[2::6, 1] = giy - 1
        uvs[3::6, 0] = gix;   uvs[3::6, 1] = giy
        uvs[4::6, 0] = gix - 1; uvs[4::6, 1] = giy
        uvs[5::6, 0] = gix - 1; uvs[5::6, 1] = giy - 1
        uvs *= self._grid_size

        return verts, indices, uvs

    def initialize_stage(self, stage=None) -> None:
        """
        Set up USD stage and create Xform hierarchy.

        Args:
            stage: USD stage (if None, gets from omni context).
        """
        global _HAS_USD, UsdGeom, Sdf, omni
        if not _HAS_USD:
            from pxr import UsdGeom as _UsdGeom, Sdf as _Sdf
            import omni as _omni
            UsdGeom = _UsdGeom
            Sdf = _Sdf
            omni = _omni
            _HAS_USD = True

        self._stage = stage or omni.usd.get_context().get_stage()
        self._pxr_utils.createXform(self._stage, self._root_path, add_default_op=True)
        self._pxr_utils.createXform(
            self._stage, f"{self._root_path}/Terrain", add_default_op=True
        )
        self._pxr_utils.createXform(
            self._stage, self._og_mesh_path, add_default_op=True
        )

    def update_heights(self, dem: np.ndarray) -> None:
        """
        Update vertex Z values from a DEM.

        Args:
            dem: 2D height array (flipped Y to match USD convention).
        """
        self._vertices[:, 2] = np.flip(dem, 0).flatten()

    def render(
        self,
        update_topology: bool = False,
        new_prim: bool = False,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        """
        Create or update the USD mesh prim.

        Args:
            update_topology: Whether to re-set indices/UVs.
            new_prim: If True, create a new mesh prim (for collider rebuild).
            colors: Optional per-vertex colors.
        """
        if not _HAS_USD:
            return

        if new_prim:
            self._mesh_path = f"{self._og_mesh_path}_{self._id}"

        mesh = UsdGeom.Mesh.Get(self._stage, self._mesh_path)
        if not mesh:
            mesh = UsdGeom.Mesh.Define(self._stage, self._mesh_path)
            UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation("vertex")
            self._pxr_utils.addDefaultOps(mesh)
            update_topology = True

        mesh.GetPointsAttr().Set(self._vertices)

        if update_topology:
            idxs = np.array(self._indices).reshape(-1, 3)
            mesh.GetFaceVertexIndicesAttr().Set(idxs)
            mesh.GetFaceVertexCountsAttr().Set([3] * len(idxs))
            UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation("vertex")
            pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st", Sdf.ValueTypeNames.Float2Array
            )
            pv.Set(self._uvs)
            pv.SetInterpolation("faceVarying")

        if colors is not None:
            mesh.GetDisplayColorAttr().Set(colors)

        if new_prim:
            self._id += 1
            self._pxr_utils.setDefaultOps(
                mesh, self._mesh_pos, self._mesh_rot, self._mesh_scale
            )

    def update_collider(self) -> None:
        """Rebuild the collision mesh for the current terrain."""
        if not _HAS_USD:
            return
        self._pxr_utils.removeCollision(self._stage, self._mesh_path)
        self._pxr_utils.addCollision(self._stage, self._mesh_path, mode="meshSimplification")

    def apply_material(self, texture_path: str) -> None:
        """Apply a material from the given USD path."""
        if not _HAS_USD:
            return
        self._pxr_utils.applyMaterialFromPath(self._stage, self._mesh_path, texture_path)

    def apply_semantic_label(self, label: str = "ground") -> None:
        """Add semantic labeling for synthetic data generation."""
        if not _HAS_USD:
            return
        try:
            from semantics.schema.editor import PrimSemanticData

            prim_sd = PrimSemanticData(self._stage.GetPrimAtPath(self._mesh_path))
            prim_sd.add_entry("class", label)
        except ImportError:
            pass

    @property
    def mesh_path(self) -> str:
        return self._mesh_path

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices
