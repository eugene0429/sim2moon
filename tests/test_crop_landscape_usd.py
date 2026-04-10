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
