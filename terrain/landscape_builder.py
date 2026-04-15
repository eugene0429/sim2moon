"""
SouthPole DEM-based landscape builder.

Loads a real lunar DEM, crops it around the main terrain area,
cuts a rectangular hole for the main procedural terrain,
downsamples for rendering performance, and builds a USD mesh.
"""

import logging
import os
from typing import Optional, Tuple

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
        self._landscape_dem = None
        self._final_resolution = None
        self._hole_edge_max_z = 0.0

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

    def find_flat_region(
        self, dem: np.ndarray, dem_resolution: float
    ) -> Tuple[int, int]:
        """Find the flattest region in the DEM for the main terrain hole.

        Computes a slope map, then uses a sliding window the size of the main
        terrain to find the region with the lowest average slope.  Among
        candidates below ``max_slope``, the one closest to the DEM center is
        chosen so the surrounding landscape stays roughly centred.

        Args:
            dem: Full DEM array.
            dem_resolution: Meters per pixel.

        Returns:
            (center_row, center_col) in pixel coordinates of the best region.
        """
        from scipy.ndimage import uniform_filter

        # Slope map (degrees)
        gy, gx = np.gradient(dem, dem_resolution)
        slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))

        # Window size = main terrain in pixels
        hole_h = int(self._main_size[1] / dem_resolution)
        hole_w = int(self._main_size[0] / dem_resolution)

        # Average slope inside a sliding window of hole size
        avg_slope = uniform_filter(slope, size=(hole_h, hole_w), mode="constant", cval=90.0)

        dem_cy, dem_cx = dem.shape[0] // 2, dem.shape[1] // 2

        # Ensure candidate centres leave room for the full crop_size box
        crop_half = int(self._conf.crop_size / dem_resolution) // 2
        margin_r = max(crop_half, hole_h // 2)
        margin_c = max(crop_half, hole_w // 2)
        valid_region = np.full_like(avg_slope, fill_value=np.inf)
        valid_region[
            margin_r : dem.shape[0] - margin_r,
            margin_c : dem.shape[1] - margin_c,
        ] = avg_slope[
            margin_r : dem.shape[0] - margin_r,
            margin_c : dem.shape[1] - margin_c,
        ]

        # Candidates below threshold
        candidates = np.argwhere(valid_region <= self._conf.max_slope)

        if candidates.shape[0] > 0:
            # Pick closest to DEM centre
            dists = np.linalg.norm(candidates - np.array([dem_cy, dem_cx]), axis=1)
            best = candidates[np.argmin(dists)]
            logger.info(
                "Flat region found at (%d, %d), avg slope %.1f° (threshold %.1f°)",
                best[0], best[1],
                float(avg_slope[best[0], best[1]]),
                self._conf.max_slope,
            )
            return int(best[0]), int(best[1])

        # Fallback: pick the globally flattest spot
        best_idx = np.unravel_index(np.argmin(valid_region), valid_region.shape)
        logger.warning(
            "No region below %.1f° slope; using flattest spot at (%d, %d) with %.1f°",
            self._conf.max_slope,
            best_idx[0], best_idx[1],
            float(avg_slope[best_idx[0], best_idx[1]]),
        )
        return int(best_idx[0]), int(best_idx[1])

    def crop_dem(
        self, dem: np.ndarray, dem_resolution: float,
        center: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Crop DEM to crop_size around a given centre pixel.

        Args:
            dem: Full DEM array.
            dem_resolution: Meters per pixel of the DEM.
            center: (row, col) to crop around.  Defaults to DEM centre.

        Returns:
            Cropped DEM array.
        """
        crop_px = int(self._conf.crop_size / dem_resolution)
        h, w = dem.shape
        half = crop_px // 2

        cy, cx = center if center is not None else (h // 2, w // 2)
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)

        return dem[y0:y1, x0:x1].copy()

    def cut_hole(
        self, dem: np.ndarray, dem_resolution: float, dem_extent: float
    ) -> np.ndarray:
        """Cut a rectangular hole in the DEM where the main terrain sits.

        Uses world-space coordinates to compute pixel bounds, ensuring the
        hole is symmetric around the main terrain center.

        Args:
            dem: Cropped DEM array (centered at main terrain position).
            dem_resolution: Meters per pixel.
            dem_extent: Total extent of the DEM in meters (assumes square).

        Returns:
            DEM with NaN values in the hole region.
        """
        result = dem.copy()
        h, w = result.shape
        cx = w / 2.0
        cy = h / 2.0

        # Main terrain world-space bounds (with margin)
        margin = self._conf.hole_margin
        main_center_x = self._main_pos[0] + self._main_size[0] / 2.0
        main_center_y = self._main_pos[1] + self._main_size[1] / 2.0
        hole_x_min = self._main_pos[0] - margin
        hole_x_max = self._main_pos[0] + self._main_size[0] + margin
        hole_y_min = self._main_pos[1] - margin
        hole_y_max = self._main_pos[1] + self._main_size[1] + margin

        # Invert vertex mapping: world_x = (col - cx) * res + main_center_x
        # => col = (world_x - main_center_x) / res + cx
        x0 = max(0, int(np.ceil((hole_x_min - main_center_x) / dem_resolution + cx)))
        x1 = min(w, int(np.floor((hole_x_max - main_center_x) / dem_resolution + cx)) + 1)
        y0 = max(0, int(np.ceil((hole_y_min - main_center_y) / dem_resolution + cy)))
        y1 = min(h, int(np.floor((hole_y_max - main_center_y) / dem_resolution + cy)) + 1)

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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)

        return result

    def _sample_landscape_height(self, wx: float, wy: float) -> float:
        """Bilinear interpolation of landscape DEM height at world position."""
        dem = self._landscape_dem
        h, w = dem.shape
        res = self._final_resolution
        cx, cy = w / 2.0, h / 2.0
        mcx = self._main_pos[0] + self._main_size[0] / 2.0
        mcy = self._main_pos[1] + self._main_size[1] / 2.0

        fc = (wx - mcx) / res + cx
        fr = (wy - mcy) / res + cy
        c0 = max(0, min(int(np.floor(fc)), w - 1))
        r0 = max(0, min(int(np.floor(fr)), h - 1))
        c1 = min(c0 + 1, w - 1)
        r1 = min(r0 + 1, h - 1)
        dc = fc - int(np.floor(fc))
        dr = fr - int(np.floor(fr))

        vals = [dem[r0, c0], dem[r0, c1], dem[r1, c0], dem[r1, c1]]
        vals = [float(v) if np.isfinite(v) else 0.0 for v in vals]
        return (vals[0] * (1 - dc) * (1 - dr) + vals[1] * dc * (1 - dr) +
                vals[2] * (1 - dc) * dr + vals[3] * dc * dr)

    @staticmethod
    def _sample_main_height(
        main_dem: np.ndarray, main_res: float,
        wx: float, wy: float,
        mx0: float, my1: float,
    ) -> float:
        """Bilinear interpolation of main terrain DEM height at world position."""
        mh, mw = main_dem.shape
        fc = (wx - mx0) / main_res
        fr = (my1 - wy) / main_res  # Y-flipped
        c0 = max(0, min(int(np.floor(fc)), mw - 1))
        r0 = max(0, min(int(np.floor(fr)), mh - 1))
        c1 = min(c0 + 1, mw - 1)
        r1 = min(r0 + 1, mh - 1)
        dc = fc - int(np.floor(fc))
        dr = fr - int(np.floor(fr))
        h00, h10 = float(main_dem[r0, c0]), float(main_dem[r0, c1])
        h01, h11 = float(main_dem[r1, c0]), float(main_dem[r1, c1])
        return (h00 * (1 - dc) * (1 - dr) + h10 * dc * (1 - dr) +
                h01 * (1 - dc) * dr + h11 * dc * dr)

    def _compute_hole_edge_max_z(self, dem: np.ndarray) -> float:
        """Return the max height among valid pixels adjacent to the NaN hole."""
        nan_mask = np.isnan(dem)
        if not np.any(nan_mask):
            return 0.0
        # Dilate NaN mask by 1 pixel to find boundary
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(nan_mask)
        edge_mask = dilated & ~nan_mask
        if not np.any(edge_mask):
            return 0.0
        return float(np.nanmax(dem[edge_mask]))

    def get_hole_edge_max_z(self) -> float:
        """Max landscape height at the hole boundary (for main terrain Z alignment)."""
        return self._hole_edge_max_z

    def build_mesh_arrays(self) -> Tuple[np.ndarray, list, np.ndarray, float]:
        """Run the full pipeline and return mesh data arrays.

        Returns:
            Tuple of (vertices [N,3], triangle_indices list, uvs [F*3,2], resolution).
        """
        dem, meta = self.load_dem()
        dem_resolution = abs(meta.get("pixel_size", [5.0, -5.0])[0])

        # Find flattest region for the hole, then crop around it
        flat_center = self.find_flat_region(dem, dem_resolution)
        dem = self.crop_dem(dem, dem_resolution, center=flat_center)
        crop_extent = min(dem.shape[0], dem.shape[1]) * dem_resolution

        # Normalize Z: subtract center height so landscape sits at Z≈0
        # (real lunar DEMs have absolute altitude in the thousands of meters)
        center_h = dem[dem.shape[0] // 2, dem.shape[1] // 2]
        if np.isfinite(center_h):
            dem = dem - center_h

        # Downsample BEFORE cutting hole to avoid quantization asymmetry.
        # Cutting the hole in the high-res grid then downsampling causes
        # NaN block boundaries to shift unevenly due to int() truncation.
        dem = self.downsample(dem, dem_resolution)
        final_resolution = max(self._conf.target_resolution, dem_resolution)

        # Cut hole in the final (downsampled) grid
        dem = self.cut_hole(dem, final_resolution, crop_extent)

        # Store for transition mesh building
        self._landscape_dem = dem
        self._final_resolution = final_resolution

        # Compute max height at hole boundary for Z-offset alignment
        self._hole_edge_max_z = self._compute_hole_edge_max_z(dem)

        # Build mesh arrays from DEM, skipping NaN pixels
        h, w = dem.shape
        valid_mask = ~np.isnan(dem)

        # Vertex index map: -1 for invalid, sequential index for valid
        index_map = np.full((h, w), -1, dtype=np.int32)
        valid_coords = np.argwhere(valid_mask)  # [N, 2] — (row, col)
        index_map[valid_mask] = np.arange(valid_coords.shape[0])

        # Vertices: x = col * res, y = row * res, z = height
        # Align landscape center with main terrain center.
        # Main terrain vertices go from (0,0) to (size_x, size_y),
        # so its center is at (size_x/2, size_y/2).
        cx = w / 2.0
        cy = h / 2.0
        main_center_x = self._main_size[0] / 2.0
        main_center_y = self._main_size[1] / 2.0
        vertices = np.zeros((valid_coords.shape[0], 3), dtype=np.float32)
        vertices[:, 0] = (valid_coords[:, 1] - cx) * final_resolution + main_center_x
        vertices[:, 1] = (valid_coords[:, 0] - cy) * final_resolution + main_center_y
        vertices[:, 2] = dem[valid_coords[:, 0], valid_coords[:, 1]]

        # Triangles: for each cell (r, c) where all 4 corners are valid, emit 2 tris
        i00_all = index_map[:-1, :-1].ravel()
        i10_all = index_map[:-1, 1:].ravel()
        i01_all = index_map[1:, :-1].ravel()
        i11_all = index_map[1:, 1:].ravel()

        valid_quads = (i00_all >= 0) & (i10_all >= 0) & (i01_all >= 0) & (i11_all >= 0)
        i00_v = i00_all[valid_quads]
        i10_v = i10_all[valid_quads]
        i01_v = i01_all[valid_quads]
        i11_v = i11_all[valid_quads]

        n_quads = i00_v.shape[0]
        indices = np.empty(n_quads * 6, dtype=np.int32)
        indices[0::6] = i00_v;  indices[1::6] = i01_v;  indices[2::6] = i10_v
        indices[3::6] = i10_v;  indices[4::6] = i01_v;  indices[5::6] = i11_v

        # UV coordinates
        cr, cc = np.divmod(np.where(valid_quads)[0], w - 1)
        uv_arr = np.empty((n_quads * 6, 2), dtype=np.float32)
        inv_w, inv_h = 1.0 / w, 1.0 / h
        uv_arr[0::6, 0] = cc * inv_w;        uv_arr[0::6, 1] = cr * inv_h
        uv_arr[1::6, 0] = cc * inv_w;        uv_arr[1::6, 1] = (cr + 1) * inv_h
        uv_arr[2::6, 0] = (cc + 1) * inv_w;  uv_arr[2::6, 1] = cr * inv_h
        uv_arr[3::6, 0] = (cc + 1) * inv_w;  uv_arr[3::6, 1] = cr * inv_h
        uv_arr[4::6, 0] = cc * inv_w;        uv_arr[4::6, 1] = (cr + 1) * inv_h
        uv_arr[5::6, 0] = (cc + 1) * inv_w;  uv_arr[5::6, 1] = (cr + 1) * inv_h

        logger.info(
            "Landscape mesh: %d vertices, %d triangles (from %dx%d DEM at %.1fm/px)",
            vertices.shape[0], indices.shape[0] // 3, h, w, final_resolution,
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
        pxr_utils.enableSmoothShade(mesh.GetPrim())

        # Apply material
        material_path = texture_path or self._conf.texture_path
        if material_path:
            pxr_utils.applyMaterialFromPath(stage, mesh_path, material_path)

        logger.info("Landscape mesh rendered at %s (%d verts)", mesh_path, vertices.shape[0])

    def build_transition_arrays(
        self, main_dem: np.ndarray, main_dem_resolution: float,
        z_offset: float = 0.0,
        n_subdivisions: int = 6,
    ) -> Tuple[np.ndarray, list, np.ndarray]:
        """Build a curved transition strip between main terrain and landscape hole.

        Creates multiple rows of vertices between the inner (main terrain edge)
        and outer (landscape hole edge) rings.  Heights are interpolated using
        a Hermite smoothstep so the surface curves naturally rather than forming
        a flat ramp.

        Args:
            main_dem: Main terrain heightmap (2D, Y-flipped for USD).
            main_dem_resolution: Meters per pixel of the main terrain DEM.
            z_offset: Z offset applied to the main terrain (added to inner heights).
            n_subdivisions: Number of intermediate rows between inner and outer.

        Returns:
            Tuple of (vertices [N,3], triangle_indices list, uvs [F*3,2]).
        """
        if self._landscape_dem is None:
            raise RuntimeError("Must call build_mesh_arrays() before transition")

        dem = self._landscape_dem
        res = self._final_resolution
        h, w = dem.shape
        cx, cy = w / 2.0, h / 2.0
        mcx = self._main_pos[0] + self._main_size[0] / 2.0
        mcy = self._main_pos[1] + self._main_size[1] / 2.0

        mx0 = self._main_pos[0]
        mx1 = self._main_pos[0] + self._main_size[0]
        my0 = self._main_pos[1]
        my1 = self._main_pos[1] + self._main_size[1]

        # Compute hole boundary in pixel space (same logic as cut_hole)
        margin = self._conf.hole_margin
        col_lo = int(np.ceil((mx0 - margin - mcx) / res + cx))
        col_hi = int(np.floor((mx1 + margin - mcx) / res + cx))
        row_lo = int(np.ceil((my0 - margin - mcy) / res + cy))
        row_hi = int(np.floor((my1 + margin - mcy) / res + cy))

        # Outer boundary = first valid pixel outside hole
        out_x_l = (col_lo - 1 - cx) * res + mcx
        out_x_r = (col_hi + 1 - cx) * res + mcx
        out_y_b = (row_lo - 1 - cy) * res + mcy
        out_y_t = (row_hi + 1 - cy) * res + mcy

        # Inner boundary = exact main terrain edge (no overlap, no inset).
        # C1 Hermite interpolation ensures smooth slope continuity at both
        # boundaries so no seam is visible even without overlap.
        t_res = 2.0
        nx = max(2, int(round((mx1 - mx0) / t_res)) + 1)
        ny = max(2, int(round((my1 - my0) / t_res)) + 1)
        x_edge = np.linspace(mx0, mx1, nx)
        y_edge = np.linspace(my0, my1, ny)

        def _mh(wx, wy):
            return self._sample_main_height(
                main_dem, main_dem_resolution, wx, wy, mx0, my1
            ) + z_offset

        def _lh(wx, wy):
            return self._sample_landscape_height(wx, wy)

        # Finite-difference gradient helpers (for C1 slope matching)
        _eps_m = main_dem_resolution * 2.0
        _eps_l = res * 2.0

        def _main_slope_radial(wx, wy, dx, dy, length):
            """Radial slope of main DEM at (wx, wy) along direction (dx, dy)."""
            dz_dx = (_mh(wx + _eps_m, wy) - _mh(wx - _eps_m, wy)) / (2 * _eps_m)
            dz_dy = (_mh(wx, wy + _eps_m) - _mh(wx, wy - _eps_m)) / (2 * _eps_m)
            return (dz_dx * dx + dz_dy * dy) / length if length > 0 else 0.0

        def _land_slope_radial(wx, wy, dx, dy, length):
            """Radial slope of landscape DEM at (wx, wy) along direction (dx, dy)."""
            dz_dx = (_lh(wx + _eps_l, wy) - _lh(wx - _eps_l, wy)) / (2 * _eps_l)
            dz_dy = (_lh(wx, wy + _eps_l) - _lh(wx, wy - _eps_l)) / (2 * _eps_l)
            return (dz_dx * dx + dz_dy * dy) / length if length > 0 else 0.0

        # Build ring: (inner_x, inner_y, inner_z, outer_x, outer_y, outer_z)
        ring = []

        # Bottom edge (left to right)
        for x in x_edge:
            ring.append((x, my0, _mh(x, my0), x, out_y_b, _lh(x, out_y_b)))
        # Bottom-right corner
        ring.append((mx1, my0, _mh(mx1, my0), out_x_r, out_y_b, _lh(out_x_r, out_y_b)))
        # Right edge (skip endpoints)
        for y in y_edge[1:-1]:
            ring.append((mx1, y, _mh(mx1, y), out_x_r, y, _lh(out_x_r, y)))
        # Top-right corner
        ring.append((mx1, my1, _mh(mx1, my1), out_x_r, out_y_t, _lh(out_x_r, out_y_t)))
        # Top edge (right to left)
        for x in x_edge[::-1]:
            ring.append((x, my1, _mh(x, my1), x, out_y_t, _lh(x, out_y_t)))
        # Top-left corner
        ring.append((mx0, my1, _mh(mx0, my1), out_x_l, out_y_t, _lh(out_x_l, out_y_t)))
        # Left edge (top to bottom, skip endpoints)
        for y in y_edge[::-1][1:-1]:
            ring.append((mx0, y, _mh(mx0, y), out_x_l, y, _lh(out_x_l, y)))
        # Bottom-left corner
        ring.append((mx0, my0, _mh(mx0, my0), out_x_l, out_y_b, _lh(out_x_l, out_y_b)))

        # Generate multi-row vertices with cubic Hermite interpolation.
        # Matches the slope (first derivative) of the main terrain at the
        # inner edge and of the landscape at the outer edge, producing a
        # C1-continuous surface across both boundaries.
        n_cols = n_subdivisions + 1
        n_ring = len(ring)
        ring_arr = np.array(ring, dtype=np.float64)  # (n_ring, 6)

        ix = ring_arr[:, 0]; iy = ring_arr[:, 1]; iz = ring_arr[:, 2]
        ox = ring_arr[:, 3]; oy = ring_arr[:, 4]; oz = ring_arr[:, 5]
        dx = ox - ix
        dy = oy - iy
        strip_len = np.sqrt(dx * dx + dy * dy)

        # Vectorised slope computation for all ring points at once
        m0 = np.zeros(n_ring, dtype=np.float64)
        m1 = np.zeros(n_ring, dtype=np.float64)
        nonzero = strip_len > 0
        if np.any(nonzero):
            # Main terrain slopes (finite differences)
            dz_dx_m = np.array([_mh(x + _eps_m, y) - _mh(x - _eps_m, y)
                                for x, y in zip(ix[nonzero], iy[nonzero])]) / (2 * _eps_m)
            dz_dy_m = np.array([_mh(x, y + _eps_m) - _mh(x, y - _eps_m)
                                for x, y in zip(ix[nonzero], iy[nonzero])]) / (2 * _eps_m)
            m0[nonzero] = ((dz_dx_m * dx[nonzero] + dz_dy_m * dy[nonzero])
                           / strip_len[nonzero]) * strip_len[nonzero]
            # Landscape slopes
            dz_dx_l = np.array([_lh(x + _eps_l, y) - _lh(x - _eps_l, y)
                                for x, y in zip(ox[nonzero], oy[nonzero])]) / (2 * _eps_l)
            dz_dy_l = np.array([_lh(x, y + _eps_l) - _lh(x, y - _eps_l)
                                for x, y in zip(ox[nonzero], oy[nonzero])]) / (2 * _eps_l)
            m1[nonzero] = ((dz_dx_l * dx[nonzero] + dz_dy_l * dy[nonzero])
                           / strip_len[nonzero]) * strip_len[nonzero]

        # Hermite interpolation for all ring points × columns at once
        t = np.linspace(0.0, 1.0, n_cols, dtype=np.float64)  # (n_cols,)
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        all_verts = np.zeros((n_ring * n_cols, 3), dtype=np.float32)
        # Outer product: (n_ring,) x (n_cols,) -> (n_ring, n_cols)
        all_verts[:, 0] = (ix[:, None] + dx[:, None] * t[None, :]).ravel()
        all_verts[:, 1] = (iy[:, None] + dy[:, None] * t[None, :]).ravel()
        all_verts[:, 2] = (h00[None, :] * iz[:, None] + h10[None, :] * m0[:, None]
                           + h01[None, :] * oz[:, None] + h11[None, :] * m1[:, None]).ravel()

        # Triangulate grid: ring_index × column_index (vectorised)
        ri = np.arange(n_ring, dtype=np.int32)
        rj = (ri + 1) % n_ring
        ci = np.arange(n_subdivisions, dtype=np.int32)
        gri, gci = np.meshgrid(ri, ci, indexing='ij')
        grj = (gri + 1) % n_ring
        gri = gri.ravel(); grj = grj.ravel(); gci = gci.ravel()

        v00 = gri * n_cols + gci
        v01 = gri * n_cols + gci + 1
        v10 = grj * n_cols + gci
        v11 = grj * n_cols + gci + 1

        n_quads = len(v00)
        indices = np.empty(n_quads * 6, dtype=np.int32)
        indices[0::6] = v00; indices[1::6] = v01; indices[2::6] = v10
        indices[3::6] = v10; indices[4::6] = v01; indices[5::6] = v11

        # Planar UV projection (vectorised)
        x_span = out_x_r - out_x_l
        y_span = out_y_t - out_y_b
        idx_verts = all_verts[indices]  # (n_indices, 3)
        uv_arr = np.zeros((indices.shape[0], 2), dtype=np.float32)
        uv_arr[:, 0] = (idx_verts[:, 0] - out_x_l) / x_span if x_span > 0 else 0.5
        uv_arr[:, 1] = (idx_verts[:, 1] - out_y_b) / y_span if y_span > 0 else 0.5

        logger.info(
            "Transition mesh: %d vertices, %d triangles (n_subdivisions=%d)",
            all_verts.shape[0], indices.shape[0] // 3, n_subdivisions,
        )
        return all_verts, indices, uv_arr

    def render_transition(
        self, stage, pxr_utils,
        main_dem: np.ndarray, main_dem_resolution: float,
        texture_path: str = "",
        z_offset: float = 0.0,
    ) -> None:
        """Build and render the transition strip mesh.

        Args:
            stage: USD stage.
            pxr_utils: USD utility module.
            main_dem: Main terrain heightmap (2D, Y-flipped).
            main_dem_resolution: Meters per pixel.
            texture_path: USD material path.
            z_offset: Z offset applied to the main terrain.
        """
        vertices, indices, uvs = self.build_transition_arrays(
            main_dem, main_dem_resolution, z_offset=z_offset,
        )
        if vertices.shape[0] == 0:
            logger.warning("No transition vertices to render")
            return

        try:
            from pxr import UsdGeom, Sdf
        except ImportError:
            logger.warning("USD not available, skipping transition render")
            return

        prim_path = self._conf.mesh_prim_path
        mesh_path = f"{prim_path}/transition_mesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        mesh.GetPointsAttr().Set(vertices)
        tri_indices = np.array(indices).reshape(-1, 3)
        mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        mesh.GetFaceVertexCountsAttr().Set([3] * len(tri_indices))

        if uvs.shape[0] > 0:
            pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st", Sdf.ValueTypeNames.Float2Array
            )
            pv.Set(uvs)
            pv.SetInterpolation("faceVarying")

        pxr_utils.enableSmoothShade(mesh.GetPrim())

        material_path = texture_path or self._conf.texture_path
        if material_path:
            pxr_utils.applyMaterialFromPath(stage, mesh_path, material_path)

        logger.info("Transition mesh rendered at %s (%d verts)", mesh_path, vertices.shape[0])
