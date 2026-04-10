# Landscape USD Crop-Scale Design

**Date**: 2026-04-10
**Status**: Approved

## Problem

The `background_landscape` static asset loads `landscape_cropped.usd` in its entirety. For most simulations, only a small region around the main terrain is visible, so loading the full mesh wastes GPU memory and increases load time.

## Goal

Generate three pre-cropped USD variants (5×, 10×, 20× the main terrain size) offline and select among them via a single YAML config field.

---

## Components

### 1. Offline Crop Script: `tools/crop_landscape_usd.py`

A standalone Python script that generates the cropped USD files. It does **not** depend on Isaac Sim — it uses only `pxr` (OpenUSD) and NumPy.

#### Inputs (CLI arguments)

| Argument | Type | Description |
|---|---|---|
| `--source` | path | Source USD file (e.g. `landscape_cropped.usd`) |
| `--terrain-size` | float | Main terrain size in meters (single value, assumes square) |
| `--pose-offset` | float×3 | Translation from YAML pose (x y z) applied to the USD prim |
| `--terrain-center` | float×2 | World-space center of main terrain (x y); defaults to `terrain_size/2 terrain_size/2` |
| `--scales` | int list | Crop scale multipliers to generate (default: `5 10 20`) |
| `--output-dir` | path | Directory for output USD files (default: same as source) |

#### Algorithm (per scale N)

1. Load source USD stage, traverse all `UsdGeom.Mesh` prims.
2. Extract `points` (Vf3) and `faceVertexIndices` (Vi) from each mesh prim.
3. Compute **world-space coordinates** of each vertex:
   `world_xy = local_xy + pose_offset_xy`
   (orientation is identity so no rotation needed; Z offset applied only to elevation, not to XY crop filter)
4. Compute **crop half-extent**:
   `half = N × terrain_size / 2`
5. Compute crop AABB in world space:
   `[center_x ± half, center_y ± half]`
6. For each triangle, keep it if **at least one vertex** falls inside the AABB.
7. Collect surviving vertex indices, remap to compact index space.
8. Write a new USD stage with a single `UsdGeom.Mesh` prim containing filtered points and indices.
9. Output filename: `{stem}_{N}x{suffix}` (e.g. `landscape_cropped_10x.usd`).

#### Triangle retention policy

"At least one vertex inside" is used (not "all vertices inside") so that triangles straddling the boundary are kept, avoiding visible holes at the crop edge.

#### Output

For each scale in `--scales`, one USD file written to `--output-dir`:
```
landscape_cropped_5x.usd
landscape_cropped_10x.usd
landscape_cropped_20x.usd
```

---

### 2. YAML Configuration

Add an optional `crop_scale` field to any static asset entry:

```yaml
static_assets_settings:
  root_path: "/StaticAssets"
  parameters:
    - asset_name: background_landscape
      usd_path: Terrains/landscape_cropped/landscape_cropped.usd
      crop_scale: 10          # optional: 5 / 10 / 20; omit to load original
      pose:
        position: [-20405.6, -9502.6, 561.8]
        orientation: [0, 0, 0, 1]
      collision: false
      material_override: /LunarYard/Looks/LunarRegolith8k_antiTile
```

If `crop_scale` is absent or `null`, the original `usd_path` is used unchanged.

---

### 3. Loader Change: `environments/lunar_yard.py`

In `_load_static_assets()`, before building `usd_path`, check for `crop_scale`:

```python
crop_scale = asset.get("crop_scale")
if crop_scale is not None:
    stem, suffix = os.path.splitext(usd_path)
    candidate = f"{stem}_{int(crop_scale)}x{suffix}"
    if os.path.isfile(candidate):
        usd_path = candidate
    else:
        logger.warning(
            "Cropped USD not found for scale %sx: %s — falling back to original",
            crop_scale, candidate,
        )
```

No other changes to the loading pipeline.

---

## File Locations

| File | Role |
|---|---|
| `tools/crop_landscape_usd.py` | Offline crop script (new) |
| `assets/Terrains/landscape_cropped/landscape_cropped_5x.usd` | Generated (not committed to git) |
| `assets/Terrains/landscape_cropped/landscape_cropped_10x.usd` | Generated (not committed to git) |
| `assets/Terrains/landscape_cropped/landscape_cropped_20x.usd` | Generated (not committed to git) |
| `environments/lunar_yard.py` | Loader (minimal change) |
| `config/environment/*.yaml` | Add `crop_scale` where desired |

---

## Error Handling

- `--source` file not found → script exits with clear error message
- Scale-specific USD not found at load time → warning log + fall back to original USD
- Source USD has no mesh prims → script exits with error
- Multiple mesh prims in source USD → each is cropped independently and all are written to the output stage

---

## Constraints

- Script requires `pxr` (OpenUSD) and `numpy` — both available in the Isaac Sim Python environment
- Generated USD files are not committed to git (add to `.gitignore`)
- Pose (translation/orientation) in YAML remains unchanged — cropped USDs keep the same coordinate origin as the original
- `crop_scale` value must match one of the pre-generated files; no runtime generation
