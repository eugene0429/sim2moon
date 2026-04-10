# Static Background Landscape Transition Strip Design

## Problem

When `background_landscape` is loaded as a static USD asset, the main procedural terrain
(e.g. 80×80 m) sits on top of it. To prevent z-fighting, the main terrain's Z position
(`mesh_position[2]`) is raised above the background. This creates an unnatural vertical
cliff at the main terrain boundary.

## Solution

Generate a procedural transition mesh at runtime that bridges the elevated main terrain
edge down to the background landscape surface. The outer boundary Z is determined by
sampling the background USD mesh vertices directly — no separate DEM file required.

---

## Architecture

```
_load_static_assets()
  ├─ load USD reference          (existing)
  ├─ pose / collision / material (existing)
  └─ transition_strip.enabled?
       ├─ sample_background_z()
       │    └─ iterate USD mesh vertices → world space → median Z within main terrain bounds
       ├─ terrain_manager.get_dem() → main terrain heightmap
       └─ build_and_render_static_transition()
            ├─ inner ring: main terrain edge, Z = DEM height + z_offset
            ├─ outer ring: band_width outside, Z = outer_z (constant)
            ├─ Hermite cubic interpolation (C1-continuous at inner edge)
            └─ render as USD Mesh prim
```

**New file:** `terrain/static_transition.py`
**Modified file:** `environments/lunar_yard.py` — `_load_static_assets()` + new `_build_background_transition()`
**Unchanged:** `terrain/landscape_builder.py`, `terrain/config.py`

---

## New Module: `terrain/static_transition.py`

### `sample_background_z`

```python
def sample_background_z(
    stage,
    prim_path: str,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> float:
```

1. Traverse all `UsdGeom.Mesh` prims under `prim_path`.
2. For each mesh, apply `ComputeLocalToWorldTransform()` to transform vertices to world space.
3. Filter vertices where `x_min ≤ x ≤ x_max` AND `y_min ≤ y ≤ y_max`.
4. Return the **median Z** of filtered vertices (robust to outliers).
5. If no vertices found in range, log a warning and return `0.0`.

### `build_static_transition_arrays`

```python
def build_static_transition_arrays(
    main_dem: np.ndarray,
    main_dem_resolution: float,
    main_pos: tuple,        # (x0, y0, z_offset) — world origin of main terrain
    main_size: tuple,       # (width, length)
    outer_z: float,         # background landscape Z
    band_width: float = 10.0,
    n_subdivisions: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

**Ring geometry:**

- `inner ring`: boundary of the main terrain
  - x ∈ [mx0, mx1], y ∈ [my0, my1]
  - Z sampled from `main_dem` (bilinear) + `z_offset`
- `outer ring`: boundary expanded outward by `band_width`
  - x ∈ [mx0 − band_width, mx1 + band_width], y ∈ [my0 − band_width, my1 + band_width]
  - Z = `outer_z` (constant)
- Ring traversal order (same as `LandscapeBuilder.build_transition_arrays`):
  bottom edge → bottom-right corner → right edge → top-right corner →
  top edge → top-left corner → left edge → bottom-left corner
- Ring point spacing: `t_res = 2.0 m`

**Hermite interpolation:**

```
t = 0 (inner):  z = DEM_edge_height,  slope m0 = radial DEM gradient
t = 1 (outer):  z = outer_z,          slope m1 = 0  (flat background)
```

Basis functions: standard cubic Hermite (`h00, h10, h01, h11`).

**Output:** `(vertices [N,3], indices [M*6], uvs [M*6, 2])`

### `render_static_transition`

```python
def render_static_transition(
    stage,
    pxr_utils,
    root_path: str,        # e.g. "/StaticAssets"
    main_dem: np.ndarray,
    main_dem_resolution: float,
    main_pos: tuple,
    main_size: tuple,
    outer_z: float,
    band_width: float,
    n_subdivisions: int,
    material_path: str = "",
) -> None:
```

- Creates prim at `{root_path}/background_landscape_transition/mesh`.
- Applies smooth shading and material (same as `background_landscape`).

---

## Integration: `environments/lunar_yard.py`

### `_load_static_assets()` — end of per-asset loop

```python
transition_cfg = asset.get("transition_strip")
if transition_cfg and transition_cfg.get("enabled", False):
    self._build_background_transition(
        root_path, prim_path, transition_cfg,
        asset.get("material_override", ""),
    )
```

### New method `_build_background_transition()`

```python
def _build_background_transition(
    self, root_path, bg_prim_path, transition_cfg, material_path
) -> None:
```

1. Check `self._terrain_manager` is not `None`; log warning and return if missing.
2. Get `main_dem = self._terrain_manager.get_dem()`.
3. Get `main_res` and `main_pos` / `main_size` from terrain manager config.
4. `z_offset = main_pos[2]` (the Z component of `mesh_position`).
5. Determine sampling bounds = main terrain XY bounds (world space).
6. Call `sample_background_z(stage, bg_prim_path, x_min, x_max, y_min, y_max)`.
7. Call `render_static_transition(...)`.

---

## Config Format

```yaml
static_assets_settings:
  root_path: "/StaticAssets"
  parameters:
    - asset_name: background_landscape
      usd_path: Terrains/landscape_cropped/landscape_cropped.usd
      pose:
        position: [-20405.6, -9502.6, 561.8]
        orientation: [0, 0, 0, 1]
      collision: false
      material_override: /LunarYard/Looks/LunarRegolith8k_antiTile
      transition_strip:
        enabled: true
        band_width: 10.0     # transition strip width in meters
        n_subdivisions: 8    # number of intermediate rows
```

`material` is taken from `material_override` automatically — no duplication needed.

---

## Key Considerations

- **World transform**: background mesh vertices are in local space; must use
  `ComputeLocalToWorldTransform()` before comparing to main terrain world bounds.
- **Median vs. mean**: median Z is preferred since the background mesh may have
  crater features near the edge — median is robust to these outliers.
- **z-fighting at outer edge**: the transition strip outer edge shares Z with the
  background landscape. Both surfaces are rendered at the same Z, which may cause
  z-fighting. If visible, the outer edge Z can be raised by a small epsilon
  (`outer_z + 0.01`) — not included in initial implementation, add if needed.
- **Guard**: if `terrain_manager` is `None` (landscape-only configs), skip silently.
- **No DEM file required**: unlike `LandscapeBuilder`, this approach works entirely
  from the already-loaded USD stage.
