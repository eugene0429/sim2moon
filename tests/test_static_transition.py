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
    assert np.allclose(verts[inner_mask, 2], 0.3 + z_offset, atol=1e-4)


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


def test_inner_ring_varies_with_non_flat_dem():
    """Inner ring heights must vary when DEM is not flat."""
    from terrain.static_transition import build_static_transition_arrays

    # DEM linearly increasing in Y: row i has height i * 0.01
    rows, cols = 100, 100
    dem = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        dem[i, :] = i * 0.01

    n_subdivisions = 4
    n_cols = n_subdivisions + 1

    verts, indices, uvs = build_static_transition_arrays(
        main_dem=dem,
        main_dem_resolution=0.1,
        main_pos=(0.0, 0.0, 0.0),
        main_size=(8.0, 8.0),
        outer_z=0.0,
        band_width=2.0,
        n_subdivisions=n_subdivisions,
    )
    inner_mask = np.arange(len(verts)) % n_cols == 0
    inner_z = verts[inner_mask, 2]
    # Heights must not all be equal (DEM varies with Y)
    assert inner_z.max() - inner_z.min() > 0.05
