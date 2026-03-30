"""
SouthPole DEM-based landscape builder.

Loads a real lunar DEM, crops it around the main terrain area,
cuts a rectangular hole for the main procedural terrain,
downsamples for rendering performance, and builds a USD mesh.
"""

import logging
import os
from typing import Tuple

import numpy as np
import yaml

from terrain.config import LandscapeConf

logger = logging.getLogger(__name__)


class LandscapeBuilder:
    """Builds an outer landscape mesh from SouthPole DEM data."""

    def __init__(
        self,
        conf: LandscapeConf,
        main_terrain_size: Tuple[float, float] = (40.0, 40.0),
        main_terrain_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._conf = conf
        self._main_size = main_terrain_size
        self._main_pos = main_terrain_position

    def load_dem(self) -> Tuple[np.ndarray, dict]:
        """Load DEM array and metadata from the configured path.

        Returns:
            Tuple of (dem_array, metadata_dict).

        Raises:
            FileNotFoundError: If dem.npy does not exist at the path.
        """
        dem_file = os.path.join(self._conf.dem_path, "dem.npy")
        meta_file = os.path.join(self._conf.dem_path, "dem.yaml")

        if not os.path.isfile(dem_file):
            raise FileNotFoundError(f"DEM file not found: {dem_file}")

        dem = np.load(dem_file)
        meta = {}
        if os.path.isfile(meta_file):
            with open(meta_file) as f:
                meta = yaml.safe_load(f)

        logger.info("Loaded DEM %s: shape=%s", self._conf.dem_path, dem.shape)
        return dem, meta

    def crop_dem(self, dem: np.ndarray, dem_resolution: float) -> np.ndarray:
        """Crop DEM to crop_size around center.

        Args:
            dem: Full DEM array.
            dem_resolution: Meters per pixel of the DEM.

        Returns:
            Cropped DEM array.
        """
        crop_px = int(self._conf.crop_size / dem_resolution)
        h, w = dem.shape
        half = crop_px // 2

        cy, cx = h // 2, w // 2
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)

        return dem[y0:y1, x0:x1].copy()

    def cut_hole(
        self, dem: np.ndarray, dem_resolution: float, dem_extent: float
    ) -> np.ndarray:
        """Cut a rectangular hole in the DEM where the main terrain sits.

        Args:
            dem: Cropped DEM array (centered at main terrain position).
            dem_resolution: Meters per pixel.
            dem_extent: Total extent of the DEM in meters (assumes square).

        Returns:
            DEM with NaN values in the hole region.
        """
        result = dem.copy()
        h, w = result.shape
        cy, cx = h // 2, w // 2

        half_x = (self._main_size[0] / 2 + self._conf.hole_margin) / dem_resolution
        half_y = (self._main_size[1] / 2 + self._conf.hole_margin) / dem_resolution

        y0 = max(0, int(cy - half_y))
        y1 = min(h, int(cy + half_y))
        x0 = max(0, int(cx - half_x))
        x1 = min(w, int(cx + half_x))

        result[y0:y1, x0:x1] = np.nan
        return result

    def downsample(self, dem: np.ndarray, source_resolution: float) -> np.ndarray:
        """Downsample DEM to target_resolution for rendering performance.

        Uses block averaging, treating NaN as missing (NaN blocks stay NaN).

        Args:
            dem: DEM array (may contain NaN for hole).
            source_resolution: Current meters per pixel.

        Returns:
            Downsampled DEM array.
        """
        factor = int(self._conf.target_resolution / source_resolution)
        if factor <= 1:
            return dem.copy()

        h, w = dem.shape
        new_h = h // factor
        new_w = w // factor
        trimmed = dem[:new_h * factor, :new_w * factor]

        # Reshape into blocks and average, preserving NaN
        blocks = trimmed.reshape(new_h, factor, new_w, factor)
        with np.errstate(invalid="ignore"):
            result = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)

        return result

    def build_mesh_arrays(self) -> Tuple[np.ndarray, list, np.ndarray, float]:
        """Run the full pipeline and return mesh data arrays.

        Returns:
            Tuple of (vertices [N,3], triangle_indices list, uvs [F*3,2], resolution).
        """
        dem, meta = self.load_dem()
        dem_resolution = abs(meta.get("pixel_size", [5.0, -5.0])[0])

        # Crop
        dem = self.crop_dem(dem, dem_resolution)
        crop_extent = min(dem.shape[0], dem.shape[1]) * dem_resolution

        # Cut hole
        dem = self.cut_hole(dem, dem_resolution, crop_extent)

        # Downsample
        dem = self.downsample(dem, dem_resolution)
        final_resolution = max(self._conf.target_resolution, dem_resolution)

        # Build mesh arrays from DEM, skipping NaN pixels
        h, w = dem.shape
        valid_mask = ~np.isnan(dem)

        # Vertex index map: -1 for invalid, sequential index for valid
        index_map = np.full((h, w), -1, dtype=np.int32)
        valid_coords = np.argwhere(valid_mask)  # [N, 2] — (row, col)
        index_map[valid_mask] = np.arange(valid_coords.shape[0])

        # Vertices: x = col * res, y = row * res, z = height
        # Center the mesh so (0,0) is at the DEM center
        cx = w / 2.0
        cy = h / 2.0
        vertices = np.zeros((valid_coords.shape[0], 3), dtype=np.float32)
        vertices[:, 0] = (valid_coords[:, 1] - cx) * final_resolution
        vertices[:, 1] = (valid_coords[:, 0] - cy) * final_resolution
        vertices[:, 2] = dem[valid_coords[:, 0], valid_coords[:, 1]]

        # Triangles: for each cell (r, c) where all 4 corners are valid, emit 2 tris
        indices = []
        uvs = []
        for r in range(h - 1):
            for c in range(w - 1):
                i00 = index_map[r, c]
                i10 = index_map[r, c + 1]
                i01 = index_map[r + 1, c]
                i11 = index_map[r + 1, c + 1]

                if i00 < 0 or i10 < 0 or i01 < 0 or i11 < 0:
                    continue

                # Triangle 1: (r,c), (r+1,c), (r,c+1)
                indices.extend([i00, i01, i10])
                uvs.extend([
                    (c / w, r / h),
                    (c / w, (r + 1) / h),
                    ((c + 1) / w, r / h),
                ])

                # Triangle 2: (r,c+1), (r+1,c), (r+1,c+1)
                indices.extend([i10, i01, i11])
                uvs.extend([
                    ((c + 1) / w, r / h),
                    (c / w, (r + 1) / h),
                    ((c + 1) / w, (r + 1) / h),
                ])

        uv_arr = np.array(uvs, dtype=np.float32) if uvs else np.zeros((0, 2), dtype=np.float32)

        logger.info(
            "Landscape mesh: %d vertices, %d triangles (from %dx%d DEM at %.1fm/px)",
            vertices.shape[0], len(indices) // 3, h, w, final_resolution,
        )

        return vertices, indices, uv_arr, final_resolution

    def render(self, stage, pxr_utils, texture_path: str = "") -> None:
        """Build mesh arrays and render as a USD mesh prim.

        Args:
            stage: USD stage.
            pxr_utils: USD utility module.
            texture_path: USD path to the material to apply.
        """
        vertices, indices, uvs, resolution = self.build_mesh_arrays()

        if vertices.shape[0] == 0:
            logger.warning("No landscape vertices to render")
            return

        try:
            from pxr import UsdGeom, Sdf
        except ImportError:
            logger.warning("USD not available, skipping landscape render")
            return

        prim_path = self._conf.mesh_prim_path
        pxr_utils.createXform(stage, prim_path, add_default_op=True)

        mesh_path = f"{prim_path}/landscape_mesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        mesh.GetPointsAttr().Set(vertices)
        tri_indices = np.array(indices).reshape(-1, 3)
        mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        mesh.GetFaceVertexCountsAttr().Set([3] * len(tri_indices))

        # UVs
        if uvs.shape[0] > 0:
            pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st", Sdf.ValueTypeNames.Float2Array
            )
            pv.Set(uvs)
            pv.SetInterpolation("faceVarying")

        # Smooth shading
        pxr_utils.enableSmoothShade(mesh)

        # Apply material
        material_path = texture_path or self._conf.texture_path
        if material_path:
            pxr_utils.applyMaterialFromPath(stage, mesh_path, material_path)

        logger.info("Landscape mesh rendered at %s (%d verts)", mesh_path, vertices.shape[0])
