"""NumPy-based Perlin noise for crater shape deformation.

Provides 1D (periodic) and 2D noise without external dependencies.
Uses gradient-based Perlin algorithm with permutation tables.
"""

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    """Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation."""
    return a + t * (b - a)


def _make_permutation(seed: int, size: int = 256) -> np.ndarray:
    """Generate a seeded permutation table, doubled for overflow."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(size).astype(np.int32)
    return np.concatenate([p, p])


def perlin_1d(
    t: np.ndarray,
    freq: float = 1.0,
    period: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    1D Perlin noise.

    Args:
        t: Input coordinates, any shape.
        freq: Frequency multiplier.
        period: If > 0, noise wraps at this period (for θ ∈ [0, 2π]).
        seed: Random seed for reproducibility.

    Returns:
        Noise values in approximately [-1, 1], same shape as t.
    """
    perm = _make_permutation(seed)
    n = len(perm) // 2

    shape = t.shape
    x = (t.ravel() * freq).astype(np.float64)

    if period > 0:
        # Number of integer grid cells in one period (float, for accurate wrap)
        period_cells_f = period * freq
        period_cells = max(1, int(np.round(period_cells_f)))
        # Remap x so that the range [0, period) maps exactly to [0, period_cells)
        x = (x % period_cells_f) * (period_cells / period_cells_f)
    else:
        period_cells = 0

    xi = np.floor(x).astype(np.int32)
    xf = x - xi

    u = _fade(xf)

    if period_cells > 0:
        xi0 = xi % period_cells
        xi1 = (xi + 1) % period_cells
    else:
        xi0 = xi % n
        xi1 = (xi + 1) % n

    g0 = (perm[xi0 % n] % 2) * 2.0 - 1.0
    g1 = (perm[xi1 % n] % 2) * 2.0 - 1.0

    n0 = g0 * xf
    n1 = g1 * (xf - 1)

    result = _lerp(n0, n1, u)
    return result.reshape(shape)


def perlin_2d(
    x: np.ndarray,
    y: np.ndarray,
    freq: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    2D Perlin noise.

    Args:
        x: X coordinates (any shape, must match y).
        y: Y coordinates (any shape, must match x).
        freq: Frequency multiplier.
        seed: Random seed for reproducibility.

    Returns:
        Noise values in approximately [-1, 1], same shape as x.
    """
    perm = _make_permutation(seed)
    n = len(perm) // 2

    grads = np.array([
        [1, 1], [-1, 1], [1, -1], [-1, -1],
        [1, 0], [-1, 0], [0, 1], [0, -1],
    ], dtype=np.float64)

    shape = x.shape
    xf = (x.ravel() * freq).astype(np.float64)
    yf = (y.ravel() * freq).astype(np.float64)

    xi = np.floor(xf).astype(np.int32)
    yi = np.floor(yf).astype(np.int32)

    xd = xf - xi
    yd = yf - yi

    u = _fade(xd)
    v = _fade(yd)

    xi = xi % n
    yi = yi % n

    h00 = perm[(perm[xi % n] + yi) % n] % 8
    h10 = perm[(perm[(xi + 1) % n] + yi) % n] % 8
    h01 = perm[(perm[xi % n] + (yi + 1)) % n] % 8
    h11 = perm[(perm[(xi + 1) % n] + (yi + 1)) % n] % 8

    d00 = grads[h00, 0] * xd + grads[h00, 1] * yd
    d10 = grads[h10, 0] * (xd - 1) + grads[h10, 1] * yd
    d01 = grads[h01, 0] * xd + grads[h01, 1] * (yd - 1)
    d11 = grads[h11, 0] * (xd - 1) + grads[h11, 1] * (yd - 1)

    x1 = _lerp(d00, d10, u)
    x2 = _lerp(d01, d11, u)
    result = _lerp(x1, x2, v)

    return result.reshape(shape)
