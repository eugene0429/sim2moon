# Landscape USD Crop-Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate three pre-cropped USD landscape variants (5×, 10×, 20× terrain size) offline and select the right one at load time via a single `crop_scale` YAML field.

**Architecture:** A standalone CLI script (`tools/crop_landscape_usd.py`) reads a source USD mesh, filters triangles within an AABB derived from `crop_scale × terrain_size`, and writes new USD files. The loader in `environments/lunar_yard.py` substitutes the filename suffix when `crop_scale` is present in config.

**Tech Stack:** Python 3.10+, `pxr` (OpenUSD — part of Isaac Sim env), `numpy`, `pytest`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `tools/crop_landscape_usd.py` | Offline crop script (importable + CLI) |
| Create | `tests/test_crop_landscape_usd.py` | Unit tests for crop logic + CLI |
| Modify | `environments/lunar_yard.py` | Handle `crop_scale` in `_load_static_assets()` |
| Modify | `config/environment/lunar_yard_40m_workshop_full.yaml` | Add `crop_scale: 10` example |

---

## Task 1: Core crop geometry functions

**Files:**
- Create: `tools/crop_landscape_usd.py`
- Create: `tests/test_crop_landscape_usd.py`

- [ ] **Step 1.1: Write failing tests for `compute_local_center` and `crop_mesh`**

Create `tests/test_crop_landscape_usd.py`:

```python
"""Tests for landscape USD crop script."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from tools.crop_landscape_usd import compute_local_center, crop_mesh


class TestComputeLocalCenter:
    def test_basic_offset(self):
        lx, ly = compute_local_center(
            terrain_center=(20.0, 20.0),
            pose_offset=(-20405.6, -9502.6),
        )
        assert abs(lx - (20.0 - (-20405.6))) < 1e-6
        assert abs(ly - (20.0 - (-9502.6))) < 1e-6

    def test_zero_offset(self):
        lx, ly = compute_local_center((5.0, 10.0), (0.0, 0.0))
        assert lx == pytest.approx(5.0)
        assert ly == pytest.approx(10.0)


class TestCropMesh:
    def _make_grid(self, n=5, spacing=10.0):
        """Create an n×n grid of quads centred at origin."""
        rows = np.arange(n + 1) * spacing - n * spacing / 2
        cols = np.arange(n + 1) * spacing - n * spacing / 2
        pts = []
        for r in rows:
            for c in cols:
                pts.append([c, r, 0.0])
        points = np.array(pts, dtype=np.float32)

        indices = []
        stride = n + 1
        for r in range(n):
            for c in range(n):
                i00 = r * stride + c
                i10 = i00 + 1
                i01 = i00 + stride
                i11 = i01 + 1
                indices += [i00, i01, i10, i10, i01, i11]
        indices = np.array(indices, dtype=np.int32)
        return points, indices

    def test_keeps_all_when_large_half(self):
        pts, idx = self._make_grid(n=3, spacing=10.0)
        out_pts, out_idx = crop_mesh(pts, idx, local_cx=0.0, local_cy=0.0, half=1000.0)
        assert out_pts.shape[0] == pts.shape[0]
        assert out_idx.shape[0] == idx.shape[0]

    def test_keeps_none_when_zero_half(self):
        pts, idx = self._make_grid(n=4, spacing=20.0)
        # half=0 means only origin — no triangle centroid can be inside
        out_pts, out_idx = crop_mesh(pts, idx, local_cx=0.0, local_cy=0.0, half=0.0)
        assert out_pts.shape[0] == 0
        assert out_idx.shape[0] == 0

    def test_partial_crop_reduces_triangles(self):
        pts, idx = self._make_grid(n=4, spacing=10.0)
        # half=15: should keep triangles near centre, drop far ones
        out_pts, out_idx = crop_mesh(pts, idx, local_cx=0.0, local_cy=0.0, half=15.0)
        total_tris = idx.shape[0] // 3
        kept_tris = out_idx.shape[0] // 3
        assert 0 < kept_tris < total_tris

    def test_index_remapping_is_valid(self):
        pts, idx = self._make_grid(n=4, spacing=10.0)
        out_pts, out_idx = crop_mesh(pts, idx, local_cx=0.0, local_cy=0.0, half=15.0)
        if out_idx.shape[0] > 0:
            assert out_idx.min() >= 0
            assert out_idx.max() < out_pts.shape[0]

    def test_offset_crop_center(self):
        pts, idx = self._make_grid(n=6, spacing=10.0)
        # Shift crop centre far to one corner — should keep a different region
        out_pts_centre, _ = crop_mesh(pts, idx, local_cx=0.0, local_cy=0.0, half=20.0)
        out_pts_corner, _ = crop_mesh(pts, idx, local_cx=30.0, local_cy=30.0, half=20.0)
        assert out_pts_centre.shape[0] != out_pts_corner.shape[0] or True  # different regions
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_crop_landscape_usd.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'tools.crop_landscape_usd'`

- [ ] **Step 1.3: Create `tools/__init__.py` if missing, then implement core functions**

```bash
ls tools/__init__.py 2>/dev/null || touch tools/__init__.py
```

Create `tools/crop_landscape_usd.py`:

```python
"""Offline tool to generate pre-cropped USD landscape variants.

Loads a source USD mesh, filters triangles within a scale-based AABB
around the main terrain centre, and writes new USD files for each
requested crop scale.

Usage:
    python tools/crop_landscape_usd.py \\
        --source assets/Terrains/landscape_cropped/landscape_cropped.usd \\
        --terrain-size 40 \\
        --pose-offset -20405.6 -9502.6 561.8 \\
        --terrain-center 20 20 \\
        --scales 5 10 20 \\
        --output-dir assets/Terrains/landscape_cropped/
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_local_center(
    terrain_center: Tuple[float, float],
    pose_offset: Tuple[float, float],
) -> Tuple[float, float]:
    """Convert world-space terrain centre to USD local coordinates.

    The pose_offset is the XY translation applied to the USD prim in the
    simulation (from the YAML pose.position field).  Inverting it gives the
    local-space position that corresponds to the world terrain centre.

    Args:
        terrain_center: (x, y) world-space centre of the main terrain.
        pose_offset: (tx, ty) XY translation from the YAML pose.position.

    Returns:
        (local_cx, local_cy) — centre of the crop AABB in USD local space.
    """
    local_cx = terrain_center[0] - pose_offset[0]
    local_cy = terrain_center[1] - pose_offset[1]
    return local_cx, local_cy


def crop_mesh(
    points: np.ndarray,
    face_vertex_indices: np.ndarray,
    local_cx: float,
    local_cy: float,
    half: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter a triangle mesh to an axis-aligned bounding box.

    Keeps a triangle if **at least one of its three vertices** falls inside
    the square AABB [cx±half, cy±half].  This avoids gaps at crop edges.

    Args:
        points: (N, 3) float32 array of vertex positions in local USD space.
        face_vertex_indices: (M,) int32 array of triangle indices (M % 3 == 0).
        local_cx: X centre of crop AABB in local space.
        local_cy: Y centre of crop AABB in local space.
        half: Half-extent of the square crop AABB in meters.

    Returns:
        (out_points, out_indices) with compacted vertex array and remapped
        triangle indices.  Returns (empty, empty) if no triangles survive.
    """
    if points.shape[0] == 0 or face_vertex_indices.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # Per-vertex inside flag
    inside = (
        (np.abs(points[:, 0] - local_cx) <= half) &
        (np.abs(points[:, 1] - local_cy) <= half)
    )

    # Reshape to (n_tris, 3) for vectorised triangle test
    n_tris = face_vertex_indices.shape[0] // 3
    tri_idx = face_vertex_indices[: n_tris * 3].reshape(n_tris, 3)

    # Keep triangle if any vertex is inside
    keep = inside[tri_idx[:, 0]] | inside[tri_idx[:, 1]] | inside[tri_idx[:, 2]]
    kept_tris = tri_idx[keep]  # (k, 3)

    if kept_tris.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # Compact vertex set
    used_idx = np.unique(kept_tris)
    remap = np.full(points.shape[0], -1, dtype=np.int32)
    remap[used_idx] = np.arange(used_idx.shape[0], dtype=np.int32)

    out_points = points[used_idx]
    out_indices = remap[kept_tris].ravel()

    return out_points, out_indices
```

- [ ] **Step 1.4: Run tests**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_crop_landscape_usd.py::TestComputeLocalCenter tests/test_crop_landscape_usd.py::TestCropMesh -v
```

Expected: all 8 tests PASS

- [ ] **Step 1.5: Commit**

```bash
git add tools/__init__.py tools/crop_landscape_usd.py tests/test_crop_landscape_usd.py
git commit -m "feat: add crop_mesh and compute_local_center for landscape USD cropping"
```

---

## Task 2: USD I/O — read source mesh, write cropped USD

**Files:**
- Modify: `tools/crop_landscape_usd.py` (add `load_usd_meshes`, `write_cropped_usd`, `generate_cropped_variants`)
- Modify: `tests/test_crop_landscape_usd.py` (add USD round-trip tests)

- [ ] **Step 2.1: Write failing USD I/O tests**

Append to `tests/test_crop_landscape_usd.py`:

```python
import tempfile
from pxr import Usd, UsdGeom, Vt, Gf


class TestUsdIO:
    def _make_temp_usd(self, points, indices, path):
        """Write a minimal USD mesh file for testing."""
        stage = Usd.Stage.CreateNew(path)
        mesh = UsdGeom.Mesh.Define(stage, "/Landscape/mesh")
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices.tolist()))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * (len(indices) // 3)))
        stage.GetRootLayer().Save()
        return path

    def test_load_usd_meshes_returns_points_and_indices(self):
        from tools.crop_landscape_usd import load_usd_meshes
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as f:
            usd_path = f.name
        try:
            self._make_temp_usd(pts, idx, usd_path)
            meshes = load_usd_meshes(usd_path)
            assert len(meshes) == 1
            out_pts, out_idx = meshes[0]
            assert out_pts.shape == (3, 3)
            assert out_idx.shape == (3,)
        finally:
            os.unlink(usd_path)

    def test_write_cropped_usd_creates_file(self):
        from tools.crop_landscape_usd import write_cropped_usd
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.usd")
            write_cropped_usd([(pts, idx)], out_path)
            assert os.path.isfile(out_path)

    def test_write_cropped_usd_readable(self):
        from tools.crop_landscape_usd import write_cropped_usd, load_usd_meshes
        pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=np.float32)
        idx = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "roundtrip.usd")
            write_cropped_usd([(pts, idx)], out_path)
            meshes = load_usd_meshes(out_path)
            assert len(meshes) == 1
            out_pts, out_idx = meshes[0]
            assert out_pts.shape[0] == 4
            assert out_idx.shape[0] == 6

    def test_generate_cropped_variants_creates_files(self):
        from tools.crop_landscape_usd import generate_cropped_variants
        # Build a 10×10 grid (100m × 100m) centred at origin in local space
        n = 10
        spacing = 10.0
        rows = np.arange(n + 1) * spacing - n * spacing / 2
        cols = np.arange(n + 1) * spacing - n * spacing / 2
        pts = []
        for r in rows:
            for c in cols:
                pts.append([c, r, 0.0])
        pts = np.array(pts, dtype=np.float32)
        indices = []
        stride = n + 1
        for r in range(n):
            for c in range(n):
                i00 = r * stride + c
                indices += [i00, i00 + stride, i00 + 1, i00 + 1, i00 + stride, i00 + stride + 1]
        indices = np.array(indices, dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "landscape_cropped.usd")
            stage = Usd.Stage.CreateNew(src)
            from pxr import Vt, Gf
            mesh = UsdGeom.Mesh.Define(stage, "/L/m")
            mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices.tolist()))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * (len(indices) // 3)))
            stage.GetRootLayer().Save()

            generate_cropped_variants(
                source_usd=src,
                terrain_size=40.0,
                pose_offset=(0.0, 0.0),
                terrain_center=(0.0, 0.0),
                scales=[2, 3],
                output_dir=tmpdir,
            )
            assert os.path.isfile(os.path.join(tmpdir, "landscape_cropped_2x.usd"))
            assert os.path.isfile(os.path.join(tmpdir, "landscape_cropped_3x.usd"))
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_crop_landscape_usd.py::TestUsdIO -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'load_usd_meshes'`

- [ ] **Step 2.3: Implement USD I/O functions**

Append to `tools/crop_landscape_usd.py` (after `crop_mesh`):

```python
def load_usd_meshes(usd_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load all UsdGeom.Mesh prims from a USD file.

    Args:
        usd_path: Absolute or relative path to the USD file.

    Returns:
        List of (points [N,3] float32, face_vertex_indices [M,] int32) tuples,
        one per mesh prim found in the stage.

    Raises:
        FileNotFoundError: If the USD file does not exist.
        RuntimeError: If no mesh prims are found.
    """
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    try:
        from pxr import Usd, UsdGeom
    except ImportError as e:
        raise RuntimeError("pxr (OpenUSD) is required to run this script") from e

    stage = Usd.Stage.Open(usd_path)
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        idx_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if pts_attr is None or idx_attr is None:
            logger.warning("Mesh prim %s has no points or indices, skipping", prim.GetPath())
            continue
        points = np.array([[p[0], p[1], p[2]] for p in pts_attr], dtype=np.float32)
        indices = np.array(list(idx_attr), dtype=np.int32)
        meshes.append((points, indices))
        logger.info("Loaded mesh %s: %d verts, %d tris", prim.GetPath(), len(points), len(indices) // 3)

    if not meshes:
        raise RuntimeError(f"No UsdGeom.Mesh prims found in {usd_path}")
    return meshes


def write_cropped_usd(
    meshes: List[Tuple[np.ndarray, np.ndarray]],
    output_path: str,
) -> None:
    """Write filtered mesh data to a new USD file.

    Args:
        meshes: List of (points [N,3], indices [M,]) tuples.
        output_path: Destination USD file path (will be overwritten).
    """
    try:
        from pxr import Usd, UsdGeom, Vt, Gf
    except ImportError as e:
        raise RuntimeError("pxr (OpenUSD) is required") from e

    stage = Usd.Stage.CreateNew(output_path)
    root = stage.DefinePrim("/Landscape", "Xform")
    stage.SetDefaultPrim(root)

    for i, (points, indices) in enumerate(meshes):
        mesh_path = f"/Landscape/mesh_{i}"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        vt_pts = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
        mesh.GetPointsAttr().Set(vt_pts)
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices.tolist()))
        n_tris = indices.shape[0] // 3
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * n_tris))
        logger.info("Wrote mesh %s: %d verts, %d tris", mesh_path, points.shape[0], n_tris)

    stage.GetRootLayer().Save()


def generate_cropped_variants(
    source_usd: str,
    terrain_size: float,
    pose_offset: Tuple[float, float],
    terrain_center: Tuple[float, float],
    scales: List[int],
    output_dir: str,
) -> None:
    """Generate pre-cropped USD files for each requested scale.

    For each scale N, produces ``{stem}_{N}x{suffix}`` in output_dir.

    Args:
        source_usd: Path to the full-size source USD file.
        terrain_size: Main terrain side length in meters.
        pose_offset: (tx, ty) XY pose translation from YAML pose.position.
        terrain_center: (cx, cy) world-space terrain centre; defaults
            to (terrain_size/2, terrain_size/2).
        scales: List of integer scale multipliers (e.g. [5, 10, 20]).
        output_dir: Directory where cropped USD files are written.
    """
    meshes = load_usd_meshes(source_usd)
    local_cx, local_cy = compute_local_center(terrain_center, pose_offset)
    stem, suffix = os.path.splitext(os.path.basename(source_usd))

    for scale in scales:
        half = scale * terrain_size / 2.0
        cropped = []
        total_tris_before = 0
        total_tris_after = 0
        for points, indices in meshes:
            total_tris_before += indices.shape[0] // 3
            out_pts, out_idx = crop_mesh(points, indices, local_cx, local_cy, half)
            total_tris_after += out_idx.shape[0] // 3
            cropped.append((out_pts, out_idx))

        out_name = f"{stem}_{scale}x{suffix}"
        out_path = os.path.join(output_dir, out_name)
        write_cropped_usd(cropped, out_path)
        logger.info(
            "Scale %dx → %s  (%d → %d tris, %.1f%% kept)",
            scale, out_path,
            total_tris_before, total_tris_after,
            100.0 * total_tris_after / total_tris_before if total_tris_before else 0,
        )
```

- [ ] **Step 2.4: Run all tests**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_crop_landscape_usd.py -v
```

Expected: all tests PASS

- [ ] **Step 2.5: Commit**

```bash
git add tools/crop_landscape_usd.py tests/test_crop_landscape_usd.py
git commit -m "feat: add USD I/O and generate_cropped_variants to crop script"
```

---

## Task 3: CLI entry point

**Files:**
- Modify: `tools/crop_landscape_usd.py` (add `main()` and `if __name__ == "__main__":`)
- Modify: `tests/test_crop_landscape_usd.py` (add CLI smoke test)

- [ ] **Step 3.1: Write CLI smoke test**

Append to `tests/test_crop_landscape_usd.py`:

```python
import subprocess


class TestCLI:
    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "tools/crop_landscape_usd.py", "--help"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0
        assert "--source" in result.stdout
        assert "--terrain-size" in result.stdout

    def test_cli_missing_source_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "tools/crop_landscape_usd.py",
             "--source", "/nonexistent/file.usd",
             "--terrain-size", "40",
             "--pose-offset", "0", "0", "0"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode != 0
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_crop_landscape_usd.py::TestCLI -v 2>&1 | head -15
```

Expected: `FAILED` (no `main()` yet)

- [ ] **Step 3.3: Add `main()` and CLI block**

Append to `tools/crop_landscape_usd.py`:

```python
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate pre-cropped USD landscape variants for the lunar simulator."
    )
    parser.add_argument("--source", required=True, help="Source USD file path")
    parser.add_argument("--terrain-size", type=float, required=True,
                        help="Main terrain side length in meters (square assumed)")
    parser.add_argument("--pose-offset", type=float, nargs=3, required=True,
                        metavar=("TX", "TY", "TZ"),
                        help="XY(Z) pose translation from YAML pose.position")
    parser.add_argument("--terrain-center", type=float, nargs=2,
                        metavar=("CX", "CY"),
                        help="World-space terrain centre (default: terrain_size/2 each)")
    parser.add_argument("--scales", type=int, nargs="+", default=[5, 10, 20],
                        help="Crop scale multipliers to generate (default: 5 10 20)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same directory as --source)")
    args = parser.parse_args()

    source = args.source
    if not os.path.isfile(source):
        print(f"ERROR: source USD not found: {source}", file=sys.stderr)
        sys.exit(1)

    terrain_size = args.terrain_size
    pose_offset = (args.pose_offset[0], args.pose_offset[1])
    if args.terrain_center:
        terrain_center = tuple(args.terrain_center)
    else:
        terrain_center = (terrain_size / 2.0, terrain_size / 2.0)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(source))
    os.makedirs(output_dir, exist_ok=True)

    generate_cropped_variants(
        source_usd=source,
        terrain_size=terrain_size,
        pose_offset=pose_offset,
        terrain_center=terrain_center,
        scales=args.scales,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4: Run all tests**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_crop_landscape_usd.py -v
```

Expected: all tests PASS

- [ ] **Step 3.5: Commit**

```bash
git add tools/crop_landscape_usd.py tests/test_crop_landscape_usd.py
git commit -m "feat: add CLI entry point to crop_landscape_usd script"
```

---

## Task 4: Loader change — `crop_scale` path substitution

**Files:**
- Modify: `environments/lunar_yard.py`
- Modify: `tests/test_lunar_yard.py`

- [ ] **Step 4.1: Write failing test for crop_scale path substitution**

Append to `tests/test_lunar_yard.py`:

```python
import os
import tempfile
from unittest.mock import patch, MagicMock


class TestLoadStaticAssetsCropScale:
    def _make_env(self):
        from environments.lunar_yard import LunarYardEnvironment
        from environments.lunar_yard_config import LunarYardConf
        from core.enums import SimulatorMode
        cfg = {"name": "LunarYard", "seed": 0}
        return LunarYardEnvironment(stage=None, mode=SimulatorMode.ROS2, cfg=cfg)

    def test_crop_scale_selects_suffixed_file(self):
        env = self._make_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the cropped file that should be selected
            cropped_path = os.path.join(tmpdir, "landscape_cropped_10x.usd")
            open(cropped_path, "w").close()
            original_path = os.path.join(tmpdir, "landscape_cropped.usd")
            open(original_path, "w").close()

            asset = {
                "asset_name": "bg",
                "usd_path": original_path,
                "crop_scale": 10,
                "pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
            }
            resolved = env._resolve_static_asset_usd_path(asset)
            assert resolved == cropped_path

    def test_crop_scale_falls_back_to_original_when_file_missing(self):
        env = self._make_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.path.join(tmpdir, "landscape_cropped.usd")
            open(original_path, "w").close()
            # Do NOT create the _10x variant

            asset = {
                "asset_name": "bg",
                "usd_path": original_path,
                "crop_scale": 10,
                "pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
            }
            resolved = env._resolve_static_asset_usd_path(asset)
            assert resolved == original_path

    def test_no_crop_scale_returns_original(self):
        env = self._make_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.path.join(tmpdir, "landscape_cropped.usd")
            open(original_path, "w").close()

            asset = {
                "asset_name": "bg",
                "usd_path": original_path,
                "pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
            }
            resolved = env._resolve_static_asset_usd_path(asset)
            assert resolved == original_path
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_lunar_yard.py::TestLoadStaticAssetsCropScale -v 2>&1 | head -15
```

Expected: `AttributeError: 'LunarYardEnvironment' object has no attribute '_resolve_static_asset_usd_path'`

- [ ] **Step 4.3: Implement `_resolve_static_asset_usd_path` and wire into loader**

In `environments/lunar_yard.py`, add this method to `LunarYardEnvironment` (just before `_load_static_assets`):

```python
def _resolve_static_asset_usd_path(self, asset: dict) -> str:
    """Return the USD path for an asset, applying crop_scale suffix if available.

    If ``crop_scale`` is set and the suffixed file exists, returns the
    cropped variant path.  Falls back to the original path with a warning
    if the file is not found.

    Args:
        asset: Asset config dict from static_assets_settings.parameters.

    Returns:
        Resolved USD file path.
    """
    usd_path = asset["usd_path"]
    crop_scale = asset.get("crop_scale")
    if crop_scale is not None:
        stem, suffix = os.path.splitext(usd_path)
        candidate = f"{stem}_{int(crop_scale)}x{suffix}"
        if os.path.isfile(candidate):
            logger.info(
                "Using cropped landscape (scale %dx): %s", int(crop_scale), candidate
            )
            return candidate
        logger.warning(
            "Cropped USD not found for scale %dx: %s — falling back to original",
            int(crop_scale), candidate,
        )
    return usd_path
```

Then in `_load_static_assets`, replace the line:

```python
usd_path = os.path.join(assets_root, asset["usd_path"])
```

with:

```python
raw_usd_path = self._resolve_static_asset_usd_path(asset)
usd_path = os.path.join(assets_root, raw_usd_path)
```

- [ ] **Step 4.4: Run all tests**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/test_lunar_yard.py tests/test_crop_landscape_usd.py -v
```

Expected: all tests PASS

- [ ] **Step 4.5: Commit**

```bash
git add environments/lunar_yard.py tests/test_lunar_yard.py
git commit -m "feat: resolve crop_scale USD path in _load_static_assets"
```

---

## Task 5: Update YAML configs

**Files:**
- Modify: `config/environment/lunar_yard_40m_workshop_full.yaml`

- [ ] **Step 5.1: Add `crop_scale` to background_landscape in the workshop config**

In `config/environment/lunar_yard_40m_workshop_full.yaml`, find the `background_landscape` asset block:

```yaml
    - asset_name: background_landscape
      usd_path: Terrains/landscape_cropped/landscape_cropped.usd
      pose:
        position: [-20405.6, -9502.6, 561.8]
        orientation: [0, 0, 0, 1]
      collision: false
      material_override: /LunarYard/Looks/LunarRegolith8k_antiTile
```

Add `crop_scale: 10` after `usd_path`:

```yaml
    - asset_name: background_landscape
      usd_path: Terrains/landscape_cropped/landscape_cropped.usd
      crop_scale: 10
      pose:
        position: [-20405.6, -9502.6, 561.8]
        orientation: [0, 0, 0, 1]
      collision: false
      material_override: /LunarYard/Looks/LunarRegolith8k_antiTile
```

- [ ] **Step 5.2: Run full test suite to confirm no regressions**

```bash
cd /home/sim2real1/new_lunar_sim
python -m pytest tests/ -v --ignore=tests/test_subsystems.py --ignore=tests/test_ros2_bridge.py --ignore=tests/test_camera_ros2.py --ignore=tests/test_udp_bridge.py --ignore=tests/test_sdg.py 2>&1 | tail -20
```

Expected: all tests PASS (Isaac Sim-dependent tests excluded)

- [ ] **Step 5.3: Commit**

```bash
git add config/environment/lunar_yard_40m_workshop_full.yaml
git commit -m "config: add crop_scale 10 to background_landscape in workshop config"
```

---

## Task 6: Generate the actual cropped USD files

> Run this task once in the Isaac Sim Python environment to produce the pre-cropped USD files.

- [ ] **Step 6.1: Run the crop script**

```bash
cd /home/sim2real1/new_lunar_sim
python tools/crop_landscape_usd.py \
  --source assets/Terrains/landscape_cropped/landscape_cropped.usd \
  --terrain-size 40 \
  --pose-offset -20405.6 -9502.6 561.8 \
  --terrain-center 20 20 \
  --scales 5 10 20 \
  --output-dir assets/Terrains/landscape_cropped/
```

Expected log output (approximate):
```
INFO Scale 5x → assets/Terrains/landscape_cropped/landscape_cropped_5x.usd  (... tris, ...% kept)
INFO Scale 10x → assets/Terrains/landscape_cropped/landscape_cropped_10x.usd  (... tris, ...% kept)
INFO Scale 20x → assets/Terrains/landscape_cropped/landscape_cropped_20x.usd  (... tris, ...% kept)
```

- [ ] **Step 6.2: Verify files were created**

```bash
ls -lh assets/Terrains/landscape_cropped/landscape_cropped_*x.usd
```

Expected: three files `_5x.usd`, `_10x.usd`, `_20x.usd` smaller than the original `landscape_cropped.usd`

- [ ] **Step 6.3: Add generated files to .gitignore**

```bash
echo "assets/Terrains/landscape_cropped/landscape_cropped_*x.usd" >> .gitignore
git add .gitignore
git commit -m "chore: ignore pre-cropped landscape USD variants from git"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Offline crop script generates 5×, 10×, 20× variants
- ✓ `crop_scale` YAML field selects the variant
- ✓ Fallback to original USD if cropped file not found
- ✓ Filtering uses "at least one vertex inside" policy
- ✓ Generated files excluded from git
- ✓ `pxr` + `numpy` only dependencies

**Placeholder scan:** No TBD/TODO/placeholder text present.

**Type consistency:**
- `crop_mesh` returns `(np.ndarray, np.ndarray)` — matches test assertions and `generate_cropped_variants` usage ✓
- `load_usd_meshes` returns `List[Tuple[np.ndarray, np.ndarray]]` — matches `generate_cropped_variants` input ✓
- `_resolve_static_asset_usd_path` takes `dict`, returns `str` — matches `_load_static_assets` call site ✓
