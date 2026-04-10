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

import numpy as np


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
