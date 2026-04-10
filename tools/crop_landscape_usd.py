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
    terrain_center: tuple,   # (x, y) world-space centre of main terrain
    pose_offset: tuple,      # (tx, ty) XY translation from YAML pose.position
) -> tuple:                  # (local_cx, local_cy)
    """Convert world-space terrain centre to USD local coordinates.

    Inverting the pose offset gives the local-space position that corresponds
    to the world terrain centre: local = world - offset
    """
    local_cx = terrain_center[0] - pose_offset[0]
    local_cy = terrain_center[1] - pose_offset[1]
    return (local_cx, local_cy)


def crop_mesh(
    points: np.ndarray,            # (N, 3) float32 vertex positions in local USD space
    face_vertex_indices: np.ndarray,  # (M,) int32 triangle indices, M % 3 == 0
    local_cx: float,               # X centre of crop AABB in local space
    local_cy: float,               # Y centre of crop AABB in local space
    half: float,                   # half-extent of square crop AABB in meters
) -> tuple:                        # (out_points [K,3], out_indices [J,])
    """Filter a triangle mesh to an axis-aligned bounding box.

    Keeps a triangle if at least one of its three vertices falls inside
    the square AABB [cx±half, cy±half]. This avoids gaps at crop edges.

    Vertex arrays are compacted: only vertices used by kept triangles are
    returned, with indices remapped accordingly.

    Returns (empty [0,3], empty [0,]) if no triangles survive.
    """
    empty_pts = np.empty((0, 3), dtype=np.float32)
    empty_idx = np.empty((0,), dtype=np.int32)

    if face_vertex_indices.shape[0] == 0:
        return empty_pts, empty_idx

    # Determine which vertices fall inside the AABB.
    # Use strict inequality so that half=0.0 produces an empty box
    # (no vertex can satisfy x > cx and x < cx simultaneously).
    x = points[:, 0]
    y = points[:, 1]
    inside = (
        (x > local_cx - half) & (x < local_cx + half) &
        (y > local_cy - half) & (y < local_cy + half)
    )

    # Reshape indices into triangles: shape (T, 3)
    tris = face_vertex_indices.reshape(-1, 3)

    # Keep a triangle if any of its vertices is inside the AABB
    v0_in = inside[tris[:, 0]]
    v1_in = inside[tris[:, 1]]
    v2_in = inside[tris[:, 2]]
    keep_mask = v0_in | v1_in | v2_in

    kept_tris = tris[keep_mask]

    if kept_tris.shape[0] == 0:
        return empty_pts, empty_idx

    # Compact vertex array: collect unique used vertex indices
    used_indices = np.unique(kept_tris)

    # Build remapping from old index to new compacted index
    remap = np.full(points.shape[0], -1, dtype=np.int32)
    remap[used_indices] = np.arange(len(used_indices), dtype=np.int32)

    out_points = points[used_indices].astype(np.float32)
    out_indices = remap[kept_tris].flatten().astype(np.int32)

    return out_points, out_indices


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
        vt_pts = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])  # explicit float() cast required by Boost.Python binding
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

    For each scale N, produces {stem}_{N}x{suffix} in output_dir.

    Args:
        source_usd: Path to the full-size source USD file.
        terrain_size: Main terrain side length in meters.
        pose_offset: (tx, ty) XY pose translation from YAML pose.position.
        terrain_center: (cx, cy) world-space terrain centre.
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
