import sys
import os
import subprocess
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from pxr import Usd, UsdGeom, Vt, Gf
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
        out_pts, out_idx = crop_mesh(pts, idx, local_cx=0.0, local_cy=0.0, half=0.0)
        assert out_pts.shape[0] == 0
        assert out_idx.shape[0] == 0

    def test_partial_crop_reduces_triangles(self):
        pts, idx = self._make_grid(n=4, spacing=10.0)
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
        out_pts_centre, out_idx_centre = crop_mesh(pts, idx, local_cx=0.0, local_cy=0.0, half=20.0)
        out_pts_corner, out_idx_corner = crop_mesh(pts, idx, local_cx=30.0, local_cy=30.0, half=20.0)
        # Validate index integrity for both results
        if out_idx_centre.shape[0] > 0:
            assert out_idx_centre.max() < out_pts_centre.shape[0]
        if out_idx_corner.shape[0] > 0:
            assert out_idx_corner.max() < out_pts_corner.shape[0]
        # A 6×6 grid at spacing=10 with half=20 should yield different counts
        # for centre (0,0) vs corner (30,30) crops
        assert out_pts_centre.shape[0] != out_pts_corner.shape[0]


class TestUsdIO:
    def _make_temp_usd(self, points, indices, path):
        """Write a minimal USD mesh file for testing."""
        stage = Usd.Stage.CreateNew(path)
        mesh = UsdGeom.Mesh.Define(stage, "/Landscape/mesh")
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points]))
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

    def test_load_usd_meshes_raises_on_missing_file(self):
        from tools.crop_landscape_usd import load_usd_meshes
        with pytest.raises(FileNotFoundError):
            load_usd_meshes("/nonexistent/path/to/file.usd")

    def test_write_cropped_usd_creates_file(self):
        from tools.crop_landscape_usd import write_cropped_usd
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.usd")
            write_cropped_usd([(pts, idx)], out_path)
            assert os.path.isfile(out_path)

    def test_write_cropped_usd_roundtrip(self):
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
        n = 10
        spacing = 10.0
        rows = np.arange(n + 1) * spacing - n * spacing / 2
        cols = np.arange(n + 1) * spacing - n * spacing / 2
        pts = np.array([[c, r, 0.0] for r in rows for c in cols], dtype=np.float32)
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
            mesh = UsdGeom.Mesh.Define(stage, "/L/m")
            mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts]))
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
