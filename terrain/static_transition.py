# terrain/static_transition.py
"""
Transition mesh between elevated main terrain and flat static background landscape.

Used when background_landscape is loaded as a static USD asset and the main
terrain is raised above it.  Samples the background mesh vertex heights once
(median Z within main terrain XY bounds) and builds a Hermite-interpolated
ring mesh that bridges the main terrain edge down to the background surface.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _sample_main_height(
    main_dem: np.ndarray,
    main_res: float,
    wx: float,
    wy: float,
    mx0: float,
    my1: float,
) -> float:
    """Bilinear interpolation of main terrain DEM height at world position.

    Args:
        main_dem: 2D heightmap (row=Y-flipped, col=X).
        main_res: Meters per pixel.
        wx: World-space X coordinate.
        wy: World-space Y coordinate.
        mx0: World-space X origin of the main terrain (left edge).
        my1: World-space Y top edge of the main terrain.
    """
    mh, mw = main_dem.shape
    fc = (wx - mx0) / main_res
    fr = (my1 - wy) / main_res  # Y-flipped
    c0 = max(0, min(int(np.floor(fc)), mw - 1))
    r0 = max(0, min(int(np.floor(fr)), mh - 1))
    c1 = min(c0 + 1, mw - 1)
    r1 = min(r0 + 1, mh - 1)
    dc = fc - int(np.floor(fc))
    dr = fr - int(np.floor(fr))
    return (
        float(main_dem[r0, c0]) * (1 - dc) * (1 - dr)
        + float(main_dem[r0, c1]) * dc * (1 - dr)
        + float(main_dem[r1, c0]) * (1 - dc) * dr
        + float(main_dem[r1, c1]) * dc * dr
    )


def build_static_transition_arrays(
    main_dem: np.ndarray,
    main_dem_resolution: float,
    main_pos: Tuple[float, float, float],
    main_size: Tuple[float, float],
    outer_z: float,
    band_width: float = 10.0,
    n_subdivisions: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build transition mesh arrays from the main terrain edge to a flat background Z.

    Inner ring vertices follow the main terrain DEM edge heights plus z_offset.
    Outer ring vertices sit at the constant ``outer_z``.  Cubic Hermite
    interpolation ensures C1 slope continuity at the inner edge; the outer
    boundary slope is zero (flat background).

    Args:
        main_dem: Main terrain heightmap (2-D float32, Y-flipped for USD).
        main_dem_resolution: Meters per pixel of the main terrain DEM.
        main_pos: (x0, y0, z_offset) world origin of the main terrain.
        main_size: (width, length) of the main terrain in meters.
        outer_z: Target Z height at the outer boundary (background surface).
        band_width: Width of the transition strip in meters.
        n_subdivisions: Number of intermediate rows between inner and outer.

    Returns:
        Tuple of (vertices [N,3] float32, indices [M*6] int32, uvs [M*6,2] float32).
    """
    mx0, my0 = main_pos[0], main_pos[1]
    z_offset = main_pos[2]
    mx1 = mx0 + main_size[0]
    my1 = my0 + main_size[1]

    out_x_l = mx0 - band_width
    out_x_r = mx1 + band_width
    out_y_b = my0 - band_width
    out_y_t = my1 + band_width

    t_res = 2.0
    nx = max(2, int(round((mx1 - mx0) / t_res)) + 1)
    ny = max(2, int(round((my1 - my0) / t_res)) + 1)
    x_edge = np.linspace(mx0, mx1, nx)
    y_edge = np.linspace(my0, my1, ny)

    def _mh(wx: float, wy: float) -> float:
        return _sample_main_height(main_dem, main_dem_resolution, wx, wy, mx0, my1) + z_offset

    # Ring: each entry is (inner_x, inner_y, inner_z, outer_x, outer_y, outer_z)
    ring = []
    # Bottom edge (left -> right)
    for x in x_edge:
        ring.append((x, my0, _mh(x, my0), x, out_y_b, outer_z))
    # Bottom-right corner
    ring.append((mx1, my0, _mh(mx1, my0), out_x_r, out_y_b, outer_z))
    # Right edge (skip endpoints)
    for y in y_edge[1:-1]:
        ring.append((mx1, y, _mh(mx1, y), out_x_r, y, outer_z))
    # Top-right corner
    ring.append((mx1, my1, _mh(mx1, my1), out_x_r, out_y_t, outer_z))
    # Top edge (right -> left)
    for x in x_edge[::-1]:
        ring.append((x, my1, _mh(x, my1), x, out_y_t, outer_z))
    # Top-left corner
    ring.append((mx0, my1, _mh(mx0, my1), out_x_l, out_y_t, outer_z))
    # Left edge (top -> bottom, skip endpoints)
    for y in y_edge[::-1][1:-1]:
        ring.append((mx0, y, _mh(mx0, y), out_x_l, y, outer_z))
    # Bottom-left corner
    ring.append((mx0, my0, _mh(mx0, my0), out_x_l, out_y_b, outer_z))

    n_cols = n_subdivisions + 1
    n_ring = len(ring)
    ring_arr = np.array(ring, dtype=np.float64)

    ix = ring_arr[:, 0]; iy = ring_arr[:, 1]; iz = ring_arr[:, 2]
    ox = ring_arr[:, 3]; oy = ring_arr[:, 4]; oz_arr = ring_arr[:, 5]
    dx = ox - ix
    dy = oy - iy
    strip_len = np.sqrt(dx ** 2 + dy ** 2)

    # Inner slope: radial finite difference from main DEM
    _eps = main_dem_resolution * 0.5
    m0 = np.zeros(n_ring, dtype=np.float64)
    nonzero = strip_len > 0
    if np.any(nonzero):
        dz_dx_m = np.array([
            _mh(x + _eps, y) - _mh(x - _eps, y)
            for x, y in zip(ix[nonzero], iy[nonzero])
        ]) / (2 * _eps)
        dz_dy_m = np.array([
            _mh(x, y + _eps) - _mh(x, y - _eps)
            for x, y in zip(ix[nonzero], iy[nonzero])
        ]) / (2 * _eps)
        m0[nonzero] = dz_dx_m * dx[nonzero] + dz_dy_m * dy[nonzero]

    m1 = np.zeros(n_ring, dtype=np.float64)  # flat outer boundary -> zero slope

    # Cubic Hermite basis at n_cols points
    t = np.linspace(0.0, 1.0, n_cols, dtype=np.float64)
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    all_verts = np.zeros((n_ring * n_cols, 3), dtype=np.float32)
    all_verts[:, 0] = (ix[:, None] + dx[:, None] * t[None, :]).ravel()
    all_verts[:, 1] = (iy[:, None] + dy[:, None] * t[None, :]).ravel()
    all_verts[:, 2] = (
        h00[None, :] * iz[:, None]
        + h10[None, :] * m0[:, None]
        + h01[None, :] * oz_arr[:, None]
        + h11[None, :] * m1[:, None]
    ).ravel()

    # Triangulate grid
    ri = np.arange(n_ring, dtype=np.int32)
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

    # Planar UV projection
    x_span = out_x_r - out_x_l
    y_span = out_y_t - out_y_b
    idx_verts = all_verts[indices]
    uv_arr = np.zeros((indices.shape[0], 2), dtype=np.float32)
    uv_arr[:, 0] = (idx_verts[:, 0] - out_x_l) / x_span if x_span > 0 else 0.5
    uv_arr[:, 1] = (idx_verts[:, 1] - out_y_b) / y_span if y_span > 0 else 0.5

    logger.info(
        "Static transition mesh: %d vertices, %d triangles (band_width=%.1f m, subdivisions=%d)",
        all_verts.shape[0], indices.shape[0] // 3, band_width, n_subdivisions,
    )
    return all_verts, indices, uv_arr
