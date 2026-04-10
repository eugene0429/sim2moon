# Static Background Landscape Transition Strip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a procedural Hermite-interpolated transition mesh that bridges the elevated main terrain edge smoothly down to the static background landscape surface.

**Architecture:** A new `terrain/static_transition.py` module handles all mesh math (pure numpy) and USD operations. `environments/lunar_yard.py` gains a `_build_background_transition()` method called from `_load_static_assets()` when a `transition_strip` block is present on a static asset config. Background surface Z is determined by sampling USD mesh vertices within the main terrain's XY footprint.

**Tech Stack:** numpy, pxr (USD / Pixar USD), Python 3

---

## File Map

| Path | Action | Responsibility |
|------|--------|----------------|
| `terrain/static_transition.py` | **Create** | `_sample_main_height`, `build_static_transition_arrays`, `sample_background_z`, `render_static_transition` |
| `environments/lunar_yard.py` | **Modify** | Add `_build_background_transition()`, hook in `_load_static_assets()` |
| `config/environment/lunar_yard_40m_workshop_full.yaml` | **Modify** | Add `transition_strip` block to `background_landscape` asset |
| `tests/test_static_transition.py` | **Create** | Pure-numpy tests (no USD dependency) |

---

### Task 1: Core mesh builder — pure numpy

**Files:**
- Create: `terrain/static_transition.py`
- Create: `tests/test_static_transition.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_static_transition.py
import numpy as np
import pytest


def test_build_arrays_returns_correct_shape():
    """Transition arrays have valid shapes: verts (N,3), indices (M,), uvs (M,2)."""
    from terrain.static_transition import build_static_transition_arrays

    dem = np.zeros((50, 50), dtype=np.float32)
    verts, indices, uvs = build_static_transition_arrays(
        main_dem=dem,
        main_dem_resolution=0.2,
        main_pos=(0.0, 0.0, 1.0),
        main_size=(8.0, 8.0),
        outer_z=0.0,
        band_width=2.0,
        n_subdivisions=4,
    )
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert indices.ndim == 1 and len(indices) % 6 == 0
    assert uvs.shape == (len(indices), 2)


def test_outer_ring_has_exact_outer_z():
    """Last column of every ring segment (t=1) must equal outer_z exactly."""
    from terrain.static_transition import build_static_transition_arrays

    dem = np.zeros((50, 50), dtype=np.float32)
    outer_z = 0.75
    n_subdivisions = 6
    n_cols = n_subdivisions + 1

    verts, indices, uvs = build_static_transition_arrays(
        main_dem=dem,
        main_dem_resolution=0.2,
        main_pos=(0.0, 0.0, 2.0),
        main_size=(8.0, 8.0),
        outer_z=outer_z,
        band_width=2.0,
        n_subdivisions=n_subdivisions,
    )
    outer_mask = np.arange(len(verts)) % n_cols == (n_cols - 1)
    assert np.allclose(verts[outer_mask, 2], outer_z, atol=1e-5)


def test_inner_ring_matches_dem_plus_offset():
    """First column of every ring segment (t=0) equals flat DEM height + z_offset."""
    from terrain.static_transition import build_static_transition_arrays

    dem = np.full((50, 50), 0.3, dtype=np.float32)
    z_offset = 2.0
    n_subdivisions = 6
    n_cols = n_subdivisions + 1

    verts, indices, uvs = build_static_transition_arrays(
        main_dem=dem,
        main_dem_resolution=0.2,
        main_pos=(0.0, 0.0, z_offset),
        main_size=(8.0, 8.0),
        outer_z=0.0,
        band_width=2.0,
        n_subdivisions=n_subdivisions,
    )
    inner_mask = np.arange(len(verts)) % n_cols == 0
    # flat DEM 0.3 + z_offset 2.0 = 2.3
    assert np.allclose(verts[inner_mask, 2], 0.3 + z_offset, atol=0.05)


def test_no_indices_out_of_bounds():
    """All triangle indices reference valid vertex positions."""
    from terrain.static_transition import build_static_transition_arrays

    dem = np.zeros((50, 50), dtype=np.float32)
    verts, indices, uvs = build_static_transition_arrays(
        main_dem=dem,
        main_dem_resolution=0.2,
        main_pos=(0.0, 0.0, 1.0),
        main_size=(8.0, 8.0),
        outer_z=0.0,
        band_width=2.0,
        n_subdivisions=4,
    )
    assert int(indices.min()) >= 0
    assert int(indices.max()) < len(verts)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_static_transition.py -v
```
Expected: `ImportError: No module named 'terrain.static_transition'`

- [ ] **Step 3: Create `terrain/static_transition.py` with `_sample_main_height` and `build_static_transition_arrays`**

```python
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
    # Bottom edge (left → right)
    for x in x_edge:
        ring.append((x, my0, _mh(x, my0), x, out_y_b, outer_z))
    # Bottom-right corner
    ring.append((mx1, my0, _mh(mx1, my0), out_x_r, out_y_b, outer_z))
    # Right edge (skip endpoints)
    for y in y_edge[1:-1]:
        ring.append((mx1, y, _mh(mx1, y), out_x_r, y, outer_z))
    # Top-right corner
    ring.append((mx1, my1, _mh(mx1, my1), out_x_r, out_y_t, outer_z))
    # Top edge (right → left)
    for x in x_edge[::-1]:
        ring.append((x, my1, _mh(x, my1), x, out_y_t, outer_z))
    # Top-left corner
    ring.append((mx0, my1, _mh(mx0, my1), out_x_l, out_y_t, outer_z))
    # Left edge (top → bottom, skip endpoints)
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
    _eps = main_dem_resolution * 2.0
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
        m0[nonzero] = (
            (dz_dx_m * dx[nonzero] + dz_dy_m * dy[nonzero])
            / strip_len[nonzero]
        ) * strip_len[nonzero]

    m1 = np.zeros(n_ring, dtype=np.float64)  # flat outer boundary → zero slope

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_static_transition.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add terrain/static_transition.py tests/test_static_transition.py
git commit -m "feat: add static transition mesh builder (pure numpy)"
```

---

### Task 2: USD sampling and render functions

**Files:**
- Modify: `terrain/static_transition.py` (append two functions)

- [ ] **Step 1: Append `sample_background_z` and `render_static_transition` to `terrain/static_transition.py`**

Add the following at the end of `terrain/static_transition.py` (after `build_static_transition_arrays`):

```python
def sample_background_z(
    stage,
    prim_path: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> float:
    """Sample median Z of a background USD mesh within a world-space XY bounding box.

    Iterates all UsdGeom.Mesh descendants of ``prim_path``, transforms their
    vertices to world space, and returns the median Z of those inside the box.

    Args:
        stage: USD stage.
        prim_path: USD path to the background landscape root prim.
        x_min, x_max: World-space X bounds.
        y_min, y_max: World-space Y bounds.

    Returns:
        Median world-space Z of vertices inside the bounding box, or 0.0 if
        no vertices are found or pxr is unavailable.
    """
    try:
        from pxr import UsdGeom, Usd
    except ImportError:
        logger.warning("pxr not available; using outer_z=0.0")
        return 0.0

    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        logger.warning("Background prim not found: %s; using outer_z=0.0", prim_path)
        return 0.0

    z_values = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points is None or len(points) == 0:
            continue

        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        pts = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
        ones = np.ones((len(pts), 1), dtype=np.float64)
        # USD GfMatrix4d is row-major: world = local @ M
        m = np.array(xform).reshape(4, 4)
        world = np.hstack([pts, ones]) @ m  # (N, 4)
        wx, wy, wz = world[:, 0], world[:, 1], world[:, 2]

        mask = (wx >= x_min) & (wx <= x_max) & (wy >= y_min) & (wy <= y_max)
        z_values.extend(wz[mask].tolist())

    if not z_values:
        logger.warning(
            "No background vertices in [%.1f, %.1f] x [%.1f, %.1f]; using outer_z=0.0",
            x_min, x_max, y_min, y_max,
        )
        return 0.0

    result = float(np.median(z_values))
    logger.info(
        "Background Z sampled from %d vertices in bounds: median=%.3f m",
        len(z_values), result,
    )
    return result


def render_static_transition(
    stage,
    pxr_utils,
    root_path: str,
    main_dem: np.ndarray,
    main_dem_resolution: float,
    main_pos: Tuple[float, float, float],
    main_size: Tuple[float, float],
    outer_z: float,
    band_width: float = 10.0,
    n_subdivisions: int = 8,
    material_path: str = "",
) -> None:
    """Build and render the static transition strip as a USD Mesh prim.

    Creates the mesh at ``{root_path}/background_landscape_transition/mesh``.

    Args:
        stage: USD stage.
        pxr_utils: core.pxr_utils module (provides createXform, enableSmoothShade,
            applyMaterialFromPath).
        root_path: USD parent path for static assets (e.g. "/StaticAssets").
        main_dem: Main terrain heightmap (2-D float32, Y-flipped).
        main_dem_resolution: Meters per pixel of the main terrain DEM.
        main_pos: (x0, y0, z_offset) world origin of the main terrain.
        main_size: (width, length) of the main terrain in meters.
        outer_z: Background landscape Z at the outer boundary.
        band_width: Width of the transition strip in meters.
        n_subdivisions: Number of intermediate rows.
        material_path: USD stage path of the material to apply.
    """
    try:
        from pxr import UsdGeom, Sdf
    except ImportError:
        logger.warning("pxr not available; skipping transition render")
        return

    vertices, indices, uvs = build_static_transition_arrays(
        main_dem, main_dem_resolution, main_pos, main_size,
        outer_z, band_width, n_subdivisions,
    )
    if vertices.shape[0] == 0:
        logger.warning("No transition vertices; skipping render")
        return

    prim_path = f"{root_path}/background_landscape_transition"
    pxr_utils.createXform(stage, prim_path, add_default_op=True)

    mesh_path = f"{prim_path}/mesh"
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
    if material_path:
        pxr_utils.applyMaterialFromPath(stage, mesh_path, material_path)

    logger.info(
        "Static transition rendered at %s (%d verts, %d tris)",
        mesh_path, vertices.shape[0], indices.shape[0] // 3,
    )
```

- [ ] **Step 2: Run existing tests to confirm nothing broke**

```bash
python -m pytest tests/test_static_transition.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add terrain/static_transition.py
git commit -m "feat: add USD Z sampler and render function to static_transition"
```

---

### Task 3: Integration in `lunar_yard.py`

**Files:**
- Modify: `environments/lunar_yard.py`

- [ ] **Step 1: Add `_build_background_transition()` method**

Open `environments/lunar_yard.py`. Find the `_apply_material_override` method (around line 407). Insert the following new method **directly before** `_apply_material_override`:

```python
def _build_background_transition(
    self,
    root_path: str,
    bg_prim_path: str,
    transition_cfg: dict,
    material_path: str,
) -> None:
    """Build and render a Hermite transition strip between main terrain and background.

    Args:
        root_path: USD root path for static assets (e.g. "/StaticAssets").
        bg_prim_path: USD path of the loaded background landscape prim.
        transition_cfg: Dict with optional keys: band_width (float, default 10.0),
            n_subdivisions (int, default 8).
        material_path: USD material path to apply to the transition mesh.
    """
    import core.pxr_utils as pxr_utils
    from terrain.static_transition import sample_background_z, render_static_transition

    if self._terrain_manager is None:
        logger.warning("No terrain manager available; skipping background transition strip")
        return

    main_dem = self._terrain_manager.get_dem()
    if main_dem is None:
        logger.warning("Terrain DEM not available; skipping background transition strip")
        return

    tm_cfg = self._cfg.get("terrain_manager", {})
    main_res = (
        tm_cfg.get("resolution", 0.02)
        if isinstance(tm_cfg, dict)
        else getattr(tm_cfg, "resolution", 0.02)
    )
    mesh_position = (
        tm_cfg.get("mesh_position", [0.0, 0.0, 0.0])
        if isinstance(tm_cfg, dict)
        else list(getattr(tm_cfg, "mesh_position", [0.0, 0.0, 0.0]))
    )
    sim_length = (
        tm_cfg.get("sim_length", 40.0)
        if isinstance(tm_cfg, dict)
        else getattr(tm_cfg, "sim_length", 40.0)
    )
    sim_width = (
        tm_cfg.get("sim_width", 40.0)
        if isinstance(tm_cfg, dict)
        else getattr(tm_cfg, "sim_width", 40.0)
    )

    main_pos = (float(mesh_position[0]), float(mesh_position[1]), float(mesh_position[2]))
    main_size = (float(sim_width), float(sim_length))

    x_min, x_max = main_pos[0], main_pos[0] + main_size[0]
    y_min, y_max = main_pos[1], main_pos[1] + main_size[1]

    outer_z = sample_background_z(
        self._stage, bg_prim_path, x_min, x_max, y_min, y_max
    )

    render_static_transition(
        self._stage,
        pxr_utils,
        root_path,
        main_dem,
        float(main_res),
        main_pos,
        main_size,
        outer_z,
        band_width=float(transition_cfg.get("band_width", 10.0)),
        n_subdivisions=int(transition_cfg.get("n_subdivisions", 8)),
        material_path=material_path,
    )
    logger.info("Background transition strip rendered (outer_z=%.3f m)", outer_z)
```

- [ ] **Step 2: Hook into `_load_static_assets()`**

In `_load_static_assets()`, find the `logger.info("Static asset '%s' loaded ...")` line (around line 405). Insert the following block **immediately before** that line:

```python
            # Transition strip (for elevated main terrain → background landscape)
            transition_cfg = asset.get("transition_strip")
            if transition_cfg and transition_cfg.get("enabled", False):
                self._build_background_transition(
                    root_path,
                    prim_path,
                    transition_cfg,
                    material_override or "",
                )
```

- [ ] **Step 3: Run existing tests to confirm nothing broke**

```bash
python -m pytest tests/test_static_transition.py -v
```
Expected: 4 tests PASS (no regressions in the integration code path since it only runs when `enabled: true`).

- [ ] **Step 4: Commit**

```bash
git add environments/lunar_yard.py
git commit -m "feat: integrate static transition strip into _load_static_assets"
```

---

### Task 4: Enable in config

**Files:**
- Modify: `config/environment/lunar_yard_40m_workshop_full.yaml`

- [ ] **Step 1: Add `transition_strip` block to `background_landscape`**

In `config/environment/lunar_yard_40m_workshop_full.yaml`, find the `background_landscape` asset entry and add the `transition_strip` block:

```yaml
    - asset_name: background_landscape
      usd_path: Terrains/landscape_cropped/landscape_cropped.usd
      pose:
        position: [-20405.6, -9502.6, 561.8]
        orientation: [0, 0, 0, 1]
      collision: false
      material_override: /LunarYard/Looks/LunarRegolith8k_antiTile
      transition_strip:
        enabled: true
        band_width: 10.0
        n_subdivisions: 8
```

- [ ] **Step 2: Commit**

```bash
git add config/environment/lunar_yard_40m_workshop_full.yaml
git commit -m "config: enable background landscape transition strip"
```
