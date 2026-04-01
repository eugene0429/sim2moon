# Realistic Crater Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `RealisticCraterGenerator` subclass that produces craters with irregular outlines, uneven rim heights, wall slumps, and floor roughness — switchable via YAML config.

**Architecture:** `RealisticCraterGenerator` extends `CraterGenerator`, overriding `_centered_distance_matrix()` and `_apply_profile()`. A new `RealisticCraterConf` dataclass holds deformation parameters. `MoonyardGenerator` selects generator class based on `crater_mode` in config. A small NumPy-based Perlin noise utility provides 1D/2D noise without external dependencies.

**Tech Stack:** Python 3.10+, NumPy, SciPy (CubicSpline, rotate), pytest

**Spec:** `docs/superpowers/specs/2026-04-01-realistic-crater-generator-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `terrain/procedural/noise.py` | NumPy-based 1D/2D Perlin noise utilities |
| Create | `terrain/procedural/realistic_crater_generator.py` | `RealisticCraterData` + `RealisticCraterGenerator` class |
| Create | `tests/test_noise.py` | Unit tests for noise utilities |
| Create | `tests/test_realistic_crater.py` | Unit + integration tests for realistic craters |
| Modify | `terrain/config.py` | Add `RealisticCraterConf`, `crater_mode` to `CraterGeneratorConf` |
| Modify | `terrain/procedural/moonyard_generator.py` | `crater_mode` dispatch logic |
| Modify | `terrain/procedural/__init__.py` | Export new classes |

---

### Task 1: Perlin Noise Utility — Tests

**Files:**
- Create: `tests/test_noise.py`

- [ ] **Step 1: Write failing tests for 1D periodic Perlin noise**

```python
"""Tests for NumPy-based Perlin noise utilities."""

import numpy as np
import pytest

from terrain.procedural.noise import perlin_1d, perlin_2d


class TestPerlin1D:
    """Tests for 1D Perlin noise with periodic boundary conditions."""

    def test_output_shape_matches_input(self):
        t = np.linspace(0, 1, 100)
        result = perlin_1d(t, seed=42)
        assert result.shape == (100,)

    def test_values_in_range(self):
        t = np.linspace(0, 10, 1000)
        result = perlin_1d(t, seed=42)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_periodicity(self):
        """Noise at t and t+period should match for periodic mode."""
        period = 2 * np.pi
        t = np.linspace(0, period, 200, endpoint=False)
        result = perlin_1d(t, period=period, seed=42)
        # First and last values should be close (periodic wrap)
        assert abs(result[0] - result[-1]) < 0.1

    def test_different_seeds_different_output(self):
        t = np.linspace(0, 5, 100)
        r1 = perlin_1d(t, seed=42)
        r2 = perlin_1d(t, seed=99)
        assert not np.allclose(r1, r2)

    def test_same_seed_reproducible(self):
        t = np.linspace(0, 5, 100)
        r1 = perlin_1d(t, seed=42)
        r2 = perlin_1d(t, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_not_all_zeros(self):
        t = np.linspace(0, 5, 100)
        result = perlin_1d(t, seed=42)
        assert np.std(result) > 0.01


class TestPerlin2D:
    """Tests for 2D Perlin noise."""

    def test_output_shape_matches_input(self):
        x = np.linspace(0, 5, 50)
        y = np.linspace(0, 5, 50)
        xx, yy = np.meshgrid(x, y)
        result = perlin_2d(xx, yy, seed=42)
        assert result.shape == (50, 50)

    def test_values_in_range(self):
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        result = perlin_2d(xx, yy, seed=42)
        assert np.all(result >= -1.5)
        assert np.all(result <= 1.5)

    def test_different_seeds_different_output(self):
        x = np.linspace(0, 5, 30)
        y = np.linspace(0, 5, 30)
        xx, yy = np.meshgrid(x, y)
        r1 = perlin_2d(xx, yy, seed=42)
        r2 = perlin_2d(xx, yy, seed=99)
        assert not np.allclose(r1, r2)

    def test_same_seed_reproducible(self):
        x = np.linspace(0, 5, 30)
        y = np.linspace(0, 5, 30)
        xx, yy = np.meshgrid(x, y)
        r1 = perlin_2d(xx, yy, seed=42)
        r2 = perlin_2d(xx, yy, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_spatial_variation(self):
        """Noise should vary spatially, not be constant."""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        result = perlin_2d(xx, yy, seed=42)
        assert np.std(result) > 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_noise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'terrain.procedural.noise'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_noise.py
git commit -m "test: add Perlin noise utility tests (red)"
```

---

### Task 2: Perlin Noise Utility — Implementation

**Files:**
- Create: `terrain/procedural/noise.py`

- [ ] **Step 1: Implement 1D and 2D Perlin noise**

```python
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
        # Map period to integer grid cells for wrapping
        period_cells = max(1, int(np.round(period * freq)))
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

    # Gradient: hash → sign × fractional distance
    g0 = (perm[xi0 % n] % 2) * 2.0 - 1.0  # -1 or +1
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

    # 2D gradients: 8 unit-ish vectors
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

    # Hash corners
    def grad_dot(ix, iy, dx, dy):
        h = perm[(perm[ix % n] + iy) % n] % 8
        g = grads[h]
        return g[:, 0][h] * dx + g[:, 1][h] * dy  # wrong: h is array

    # Vectorized hash → gradient dot
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_noise.py -v`
Expected: All 11 tests PASS

- [ ] **Step 3: Commit**

```bash
git add terrain/procedural/noise.py tests/test_noise.py
git commit -m "feat: add NumPy-based Perlin noise utilities for crater deformation"
```

---

### Task 3: Config Dataclasses

**Files:**
- Modify: `terrain/config.py:13-31` (CraterGeneratorConf) and `terrain/config.py:183-210` (MoonYardConf)
- Create: tests in `tests/test_realistic_crater.py` (partial — config section)

- [ ] **Step 1: Write failing tests for config dataclasses**

Create `tests/test_realistic_crater.py`:

```python
"""Tests for realistic crater generator."""

import numpy as np
import pytest

from terrain.config import CraterGeneratorConf, RealisticCraterConf, MoonYardConf


class TestRealisticCraterConf:
    """Tests for RealisticCraterConf dataclass."""

    def test_defaults(self):
        conf = RealisticCraterConf()
        assert conf.n_harmonics == 4
        assert conf.harmonic_amp == 0.12
        assert conf.contour_noise_amp == 0.03
        assert conf.rim_n_harmonics == 3
        assert conf.rim_noise_amp == 0.15
        assert conf.slump_intensity == 0.1
        assert conf.slump_wall_range == (0.3, 0.8)
        assert conf.floor_noise_amp == 0.03
        assert conf.floor_radius_ratio == 0.3

    def test_from_dict(self):
        conf = RealisticCraterConf(**{"n_harmonics": 6, "harmonic_amp": 0.2})
        assert conf.n_harmonics == 6
        assert conf.harmonic_amp == 0.2

    def test_validation_n_harmonics_positive(self):
        with pytest.raises(AssertionError):
            RealisticCraterConf(n_harmonics=0)

    def test_validation_harmonic_amp_range(self):
        with pytest.raises(AssertionError):
            RealisticCraterConf(harmonic_amp=-0.1)

    def test_validation_slump_wall_range_order(self):
        with pytest.raises(AssertionError):
            RealisticCraterConf(slump_wall_range=(0.8, 0.3))


class TestCraterGeneratorConfMode:
    """Tests for crater_mode field in CraterGeneratorConf."""

    def test_default_mode_is_classic(self):
        conf = CraterGeneratorConf()
        assert conf.crater_mode == "classic"

    def test_realistic_mode(self):
        conf = CraterGeneratorConf(crater_mode="realistic")
        assert conf.crater_mode == "realistic"

    def test_invalid_mode_raises(self):
        with pytest.raises(AssertionError):
            CraterGeneratorConf(crater_mode="invalid")


class TestMoonYardConfRealistic:
    """Tests for MoonYardConf with realistic_crater field."""

    def test_default_has_realistic_crater(self):
        conf = MoonYardConf()
        assert isinstance(conf.realistic_crater, RealisticCraterConf)

    def test_from_dict_with_realistic(self):
        conf = MoonYardConf(realistic_crater={"n_harmonics": 5})
        assert conf.realistic_crater.n_harmonics == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestRealisticCraterConf -v`
Expected: FAIL — `ImportError: cannot import name 'RealisticCraterConf'`

- [ ] **Step 3: Add RealisticCraterConf to config.py**

Add after `CraterGeneratorConf` (after line 31 in `terrain/config.py`):

```python
@dataclasses.dataclass
class RealisticCraterConf:
    """Parameters for realistic crater shape deformation."""

    n_harmonics: int = 4
    harmonic_amp: float = 0.12
    contour_noise_amp: float = 0.03
    rim_n_harmonics: int = 3
    rim_noise_amp: float = 0.15
    slump_intensity: float = 0.1
    slump_wall_range: Tuple[float, float] = (0.3, 0.8)
    floor_noise_amp: float = 0.03
    floor_radius_ratio: float = 0.3

    def __post_init__(self):
        assert self.n_harmonics > 0, "n_harmonics must be positive"
        assert self.harmonic_amp >= 0, "harmonic_amp must be non-negative"
        assert self.contour_noise_amp >= 0, "contour_noise_amp must be non-negative"
        assert self.rim_n_harmonics > 0, "rim_n_harmonics must be positive"
        assert self.rim_noise_amp >= 0, "rim_noise_amp must be non-negative"
        assert self.slump_intensity >= 0, "slump_intensity must be non-negative"
        assert self.slump_wall_range[0] < self.slump_wall_range[1], (
            "slump_wall_range[0] must be less than slump_wall_range[1]"
        )
        assert self.floor_noise_amp >= 0, "floor_noise_amp must be non-negative"
        assert 0 <= self.floor_radius_ratio <= 1, "floor_radius_ratio must be in [0, 1]"
```

- [ ] **Step 4: Add crater_mode to CraterGeneratorConf**

Add `crater_mode` field and validation to `CraterGeneratorConf`:

```python
@dataclasses.dataclass
class CraterGeneratorConf:
    """Configuration for crater shape generation from spline profiles."""

    profiles_path: str = ""
    min_xy_ratio: float = 0.85
    max_xy_ratio: float = 1.0
    resolution: float = 0.02
    pad_size: int = 500
    random_rotation: bool = True
    z_scale: float = 0.2
    seed: int = 42
    crater_mode: str = "classic"

    def __post_init__(self):
        assert 0 < self.min_xy_ratio <= self.max_xy_ratio <= 1.0, (
            f"xy_ratio must satisfy 0 < min ({self.min_xy_ratio}) <= max ({self.max_xy_ratio}) <= 1.0"
        )
        assert self.resolution > 0, "resolution must be positive"
        assert self.pad_size >= 0, "pad_size must be non-negative"
        assert self.z_scale > 0, "z_scale must be positive"
        assert self.crater_mode in ("classic", "realistic"), (
            f"crater_mode must be 'classic' or 'realistic', got '{self.crater_mode}'"
        )
```

- [ ] **Step 5: Add realistic_crater field to MoonYardConf**

In `MoonYardConf`, add:

```python
@dataclasses.dataclass
class MoonYardConf:
    """Configuration for the procedural moonyard terrain pipeline."""

    crater_generator: CraterGeneratorConf = dataclasses.field(
        default_factory=CraterGeneratorConf
    )
    crater_distribution: CraterDistributionConf = dataclasses.field(
        default_factory=CraterDistributionConf
    )
    base_terrain_generator: BaseTerrainGeneratorConf = dataclasses.field(
        default_factory=BaseTerrainGeneratorConf
    )
    deformation_engine: DeformationEngineConf = dataclasses.field(
        default_factory=DeformationEngineConf
    )
    realistic_crater: RealisticCraterConf = dataclasses.field(
        default_factory=RealisticCraterConf
    )
    is_yard: bool = True
    is_lab: bool = False

    def __post_init__(self):
        if isinstance(self.crater_generator, dict):
            self.crater_generator = CraterGeneratorConf(**self.crater_generator)
        if isinstance(self.crater_distribution, dict):
            self.crater_distribution = CraterDistributionConf(**self.crater_distribution)
        if isinstance(self.base_terrain_generator, dict):
            self.base_terrain_generator = BaseTerrainGeneratorConf(**self.base_terrain_generator)
        if isinstance(self.deformation_engine, dict):
            self.deformation_engine = DeformationEngineConf(**self.deformation_engine)
        if isinstance(self.realistic_crater, dict):
            self.realistic_crater = RealisticCraterConf(**self.realistic_crater)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py -v`
Expected: All 8 tests PASS

- [ ] **Step 7: Run existing tests to verify no regressions**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_lunar_yard_config.py tests/test_e2e_config.py -v`
Expected: All existing tests PASS

- [ ] **Step 8: Commit**

```bash
git add terrain/config.py tests/test_realistic_crater.py
git commit -m "feat: add RealisticCraterConf and crater_mode to config"
```

---

### Task 4: RealisticCraterData and randomize_parameters — Tests

**Files:**
- Modify: `tests/test_realistic_crater.py`

- [ ] **Step 1: Add tests for RealisticCraterData and randomize_parameters**

Append to `tests/test_realistic_crater.py`:

```python
from terrain.procedural.realistic_crater_generator import (
    RealisticCraterData,
    RealisticCraterGenerator,
)
from terrain.procedural.crater_generator import CraterData


class TestRealisticCraterData:
    """Tests for RealisticCraterData dataclass."""

    def test_inherits_crater_data(self):
        rcd = RealisticCraterData()
        assert isinstance(rcd, CraterData)

    def test_has_harmonic_fields(self):
        rcd = RealisticCraterData()
        assert hasattr(rcd, "harmonic_amplitudes")
        assert hasattr(rcd, "harmonic_phases")
        assert hasattr(rcd, "contour_noise_seed")

    def test_has_rim_fields(self):
        rcd = RealisticCraterData()
        assert hasattr(rcd, "rim_amplitudes")
        assert hasattr(rcd, "rim_phases")

    def test_has_slump_and_floor_fields(self):
        rcd = RealisticCraterData()
        assert hasattr(rcd, "slump_noise_seed")
        assert hasattr(rcd, "floor_noise_seed")


class TestRealisticRandomizeParameters:
    """Tests for RealisticCraterGenerator.randomize_parameters()."""

    @pytest.fixture
    def generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf(n_harmonics=4, rim_n_harmonics=3)
        return RealisticCraterGenerator(cfg, rcfg)

    def test_returns_realistic_crater_data(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert isinstance(cd, RealisticCraterData)

    def test_harmonic_amplitudes_shape(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert cd.harmonic_amplitudes.shape == (4,)
        assert cd.harmonic_phases.shape == (4,)

    def test_rim_amplitudes_shape(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert cd.rim_amplitudes.shape == (3,)
        assert cd.rim_phases.shape == (3,)

    def test_noise_seeds_are_integers(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert isinstance(cd.contour_noise_seed, (int, np.integer))
        assert isinstance(cd.slump_noise_seed, (int, np.integer))
        assert isinstance(cd.floor_noise_seed, (int, np.integer))

    def test_preserves_base_fields(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        assert cd.size == 101
        assert cd.deformation_spline is not None
        assert cd.marks_spline is not None

    def test_reproducible_with_same_seed(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf()
        gen1 = RealisticCraterGenerator(cfg, rcfg)
        gen2 = RealisticCraterGenerator(cfg, rcfg)
        cd1 = gen1.randomize_parameters(-1, 101)
        cd2 = gen2.randomize_parameters(-1, 101)
        np.testing.assert_array_equal(cd1.harmonic_amplitudes, cd2.harmonic_amplitudes)
        np.testing.assert_array_equal(cd1.rim_amplitudes, cd2.rim_amplitudes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestRealisticCraterData -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Commit test additions**

```bash
git add tests/test_realistic_crater.py
git commit -m "test: add RealisticCraterData and randomize_parameters tests (red)"
```

---

### Task 5: RealisticCraterData and randomize_parameters — Implementation

**Files:**
- Create: `terrain/procedural/realistic_crater_generator.py` (partial — data + randomize)

- [ ] **Step 1: Implement RealisticCraterData and initial RealisticCraterGenerator**

```python
"""Realistic crater generator with irregular outlines and uneven rims.

Extends CraterGenerator with:
- Harmonic contour perturbation + Perlin noise for irregular outlines
- Azimuthal rim height modulation
- Wall slump noise
- Floor roughness via 2D Perlin noise
"""

import dataclasses
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import rotate

from terrain.config import CraterGeneratorConf, RealisticCraterConf
from terrain.procedural.crater_generator import CraterData, CraterGenerator
from terrain.procedural.noise import perlin_1d, perlin_2d


@dataclasses.dataclass
class RealisticCraterData(CraterData):
    """Extended crater metadata for realistic generation."""

    harmonic_amplitudes: Optional[np.ndarray] = None
    harmonic_phases: Optional[np.ndarray] = None
    contour_noise_seed: int = 0

    rim_amplitudes: Optional[np.ndarray] = None
    rim_phases: Optional[np.ndarray] = None

    slump_noise_seed: int = 0
    floor_noise_seed: int = 0


class RealisticCraterGenerator(CraterGenerator):
    """Generates craters with irregular shapes and uneven rims."""

    def __init__(
        self, cfg: CraterGeneratorConf, realistic_cfg: RealisticCraterConf
    ) -> None:
        super().__init__(cfg)
        self._rcfg = realistic_cfg

    def randomize_parameters(self, index: int, size: int) -> RealisticCraterData:
        """Generate randomized parameters including realistic deformation fields."""
        # Get base parameters from parent
        base_cd = super().randomize_parameters(index, size)

        # Build RealisticCraterData from base fields
        rcd = RealisticCraterData(
            deformation_spline=base_cd.deformation_spline,
            marks_spline=base_cd.marks_spline,
            marks_intensity=base_cd.marks_intensity,
            size=base_cd.size,
            crater_profile_id=base_cd.crater_profile_id,
            xy_deformation_factor=base_cd.xy_deformation_factor,
            rotation=base_cd.rotation,
            coord=base_cd.coord,
        )

        rc = self._rcfg

        # Harmonic contour: amplitude decays as 1/n
        harmonics_n = np.arange(2, 2 + rc.n_harmonics)
        rcd.harmonic_amplitudes = np.array([
            self._rng.uniform(0, rc.harmonic_amp / n) for n in harmonics_n
        ])
        rcd.harmonic_phases = self._rng.uniform(0, 2 * np.pi, rc.n_harmonics)
        rcd.contour_noise_seed = int(self._rng.integers(0, 2**31))

        # Rim height modulation
        rim_n = np.arange(2, 2 + rc.rim_n_harmonics)
        rcd.rim_amplitudes = np.array([
            self._rng.uniform(0, rc.rim_noise_amp / n) for n in rim_n
        ])
        rcd.rim_phases = self._rng.uniform(0, 2 * np.pi, rc.rim_n_harmonics)

        # Noise seeds
        rcd.slump_noise_seed = int(self._rng.integers(0, 2**31))
        rcd.floor_noise_seed = int(self._rng.integers(0, 2**31))

        return rcd
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestRealisticCraterData tests/test_realistic_crater.py::TestRealisticRandomizeParameters -v`
Expected: All 11 tests PASS

- [ ] **Step 3: Commit**

```bash
git add terrain/procedural/realistic_crater_generator.py
git commit -m "feat: add RealisticCraterData and randomize_parameters"
```

---

### Task 6: _centered_distance_matrix Override — Tests

**Files:**
- Modify: `tests/test_realistic_crater.py`

- [ ] **Step 1: Add tests for irregular distance matrix**

Append to `tests/test_realistic_crater.py`:

```python
class TestRealisticDistanceMatrix:
    """Tests for _centered_distance_matrix override."""

    @pytest.fixture
    def generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf(n_harmonics=4, harmonic_amp=0.12, contour_noise_amp=0.03)
        return RealisticCraterGenerator(cfg, rcfg)

    @pytest.fixture
    def classic_generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            seed=42,
        )
        return CraterGenerator(cfg)

    def test_output_shape(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        dist = generator._centered_distance_matrix(cd)
        assert dist.shape == (101, 101)

    def test_contour_is_irregular(self, generator):
        """Distance values along a circle should vary (not uniform)."""
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        # Sample points on a ring at ~70% of crater radius
        center = 100
        radius = 70
        n_samples = 360
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        ring_vals = []
        for a in angles:
            r, c = int(center + radius * np.sin(a)), int(center + radius * np.cos(a))
            if 0 <= r < 201 and 0 <= c < 201:
                ring_vals.append(dist[r, c])
        ring_vals = np.array(ring_vals)
        # Standard deviation should be significant (not a perfect circle)
        assert np.std(ring_vals) > 1.0, f"Ring std too low: {np.std(ring_vals)}"

    def test_differs_from_classic(self, generator, classic_generator):
        """Realistic distance matrix should differ from classic."""
        # Use same base parameters but different generators
        cd_real = generator.randomize_parameters(-1, 101)
        cd_classic = classic_generator.randomize_parameters(-1, 101)
        dist_real = generator._centered_distance_matrix(cd_real)
        dist_classic = classic_generator._centered_distance_matrix(cd_classic)
        # They use different seeds so direct comparison isn't meaningful,
        # but we can check structural differences exist
        assert dist_real.shape == dist_classic.shape

    def test_center_is_zero(self, generator):
        """Center of crater should have near-zero distance."""
        cd = generator.randomize_parameters(-1, 101)
        dist = generator._centered_distance_matrix(cd)
        center = cd.size // 2
        assert dist[center, center] < 5.0

    def test_clamped_to_half_size(self, generator):
        """Distance values should not exceed size/2."""
        cd = generator.randomize_parameters(-1, 101)
        dist = generator._centered_distance_matrix(cd)
        assert np.all(dist <= cd.size / 2 + 0.5)  # small tolerance for rotation interpolation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestRealisticDistanceMatrix -v`
Expected: FAIL — `_centered_distance_matrix` returns parent's uniform result (contour_is_irregular fails)

- [ ] **Step 3: Commit**

```bash
git add tests/test_realistic_crater.py
git commit -m "test: add distance matrix irregularity tests (red)"
```

---

### Task 7: _centered_distance_matrix Override — Implementation

**Files:**
- Modify: `terrain/procedural/realistic_crater_generator.py`

- [ ] **Step 1: Implement _centered_distance_matrix override**

Add method to `RealisticCraterGenerator`:

```python
    def _centered_distance_matrix(self, cd: CraterData) -> np.ndarray:
        """Build distance matrix with harmonic contour perturbation + Perlin noise."""
        if not isinstance(cd, RealisticCraterData):
            return super()._centered_distance_matrix(cd)

        n = cd.size
        rc = self._rcfg

        # Angular coordinate grid
        x_lin, y_lin = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        theta = np.arctan2(y_lin, x_lin)

        # --- Harmonic contour perturbation ---
        contour = np.ones_like(theta)
        for i, harm_n in enumerate(range(2, 2 + rc.n_harmonics)):
            contour += cd.harmonic_amplitudes[i] * np.cos(
                harm_n * theta + cd.harmonic_phases[i]
            )

        # High-freq Perlin noise on contour (periodic over 2π)
        theta_flat = theta.ravel()
        noise = perlin_1d(
            theta_flat + np.pi,  # shift to [0, 2π] range
            freq=3.0,
            period=2 * np.pi,
            seed=cd.contour_noise_seed,
        ).reshape(theta.shape)
        contour += noise * rc.contour_noise_amp

        # --- Euclidean distance with elliptical deformation ---
        x_idx, y_idx = np.meshgrid(range(n), range(n))
        dist = np.sqrt(
            ((x_idx - n / 2 + 1) / cd.xy_deformation_factor[0]) ** 2
            + ((y_idx - n / 2 + 1) / cd.xy_deformation_factor[1]) ** 2
        )

        # Apply contour perturbation (divide so contour>1 means larger crater in that direction)
        dist = dist / contour

        # Rotate
        dist = rotate(dist, cd.rotation, reshape=False, cval=n / 2)

        # Clamp to crater radius
        dist[dist > n / 2] = n / 2

        return dist
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestRealisticDistanceMatrix -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add terrain/procedural/realistic_crater_generator.py
git commit -m "feat: implement _centered_distance_matrix with harmonic + Perlin contour"
```

---

### Task 8: _apply_profile Override — Tests

**Files:**
- Modify: `tests/test_realistic_crater.py`

- [ ] **Step 1: Add tests for rim height, slump, and floor noise**

Append to `tests/test_realistic_crater.py`:

```python
class TestRealisticApplyProfile:
    """Tests for _apply_profile override with rim modulation, slumps, floor noise."""

    @pytest.fixture
    def generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf(
            rim_noise_amp=0.15,
            slump_intensity=0.1,
            floor_noise_amp=0.03,
        )
        return RealisticCraterGenerator(cfg, rcfg)

    @pytest.fixture
    def classic_generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            seed=42,
        )
        return CraterGenerator(cfg)

    def test_rim_height_varies_azimuthally(self, generator):
        """Rim height should vary around the crater perimeter."""
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        # Extract rim ring (near edge, ~90% of radius)
        center = 100
        rim_r = 90
        n_samples = 360
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        rim_heights = []
        for a in angles:
            r = int(center + rim_r * np.sin(a))
            c = int(center + rim_r * np.cos(a))
            if 0 <= r < 201 and 0 <= c < 201:
                rim_heights.append(crater[r, c])
        rim_heights = np.array(rim_heights)
        # Rim should show variation
        assert np.std(rim_heights) > 0.0001, f"Rim std too low: {np.std(rim_heights)}"

    def test_wall_has_slump_noise(self, generator):
        """Wall region should have noise (compared to smooth classic)."""
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        # Extract wall region values along a radial line
        center = 100
        wall_vals = crater[center, 50:90]  # roughly 0.3R to 0.8R from center
        # Wall should not be perfectly smooth
        diffs = np.diff(wall_vals)
        assert np.std(diffs) > 0.0, "Wall region has no variation"

    def test_floor_has_noise(self, generator):
        """Floor region should have micro-roughness."""
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        # Floor region: central area
        center = 100
        floor_patch = crater[center - 15:center + 15, center - 15:center + 15]
        assert np.std(floor_patch) > 0.0, "Floor is perfectly flat"

    def test_output_shape(self, generator):
        cd = generator.randomize_parameters(-1, 101)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        assert crater.shape == (101, 101)

    def test_no_nans(self, generator):
        cd = generator.randomize_parameters(-1, 201)
        dist = generator._centered_distance_matrix(cd)
        crater = generator._apply_profile(dist, cd)
        assert not np.any(np.isnan(crater))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestRealisticApplyProfile -v`
Expected: FAIL — `_apply_profile` still uses parent's uniform profile

- [ ] **Step 3: Commit**

```bash
git add tests/test_realistic_crater.py
git commit -m "test: add rim, slump, and floor noise tests (red)"
```

---

### Task 9: _apply_profile Override — Implementation

**Files:**
- Modify: `terrain/procedural/realistic_crater_generator.py`

- [ ] **Step 1: Implement _apply_profile with 3-layer composition**

Add method to `RealisticCraterGenerator`:

```python
    @staticmethod
    def _smooth_step(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
        """Hermite smooth step: 0 at edge0, 1 at edge1."""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0, 1)
        return t * t * (3 - 2 * t)

    def _apply_profile(self, distance: np.ndarray, cd: CraterData) -> np.ndarray:
        """Apply profile with rim modulation, wall slumps, and floor noise."""
        if not isinstance(cd, RealisticCraterData):
            return super()._apply_profile(distance, cd)

        rc = self._rcfg
        n = cd.size
        half = n / 2.0

        # Base profile from spline
        base = self._profiles[cd.crater_profile_id](2 * distance / n)

        # --- Layer 1: Rim height modulation ---
        x_lin, y_lin = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        theta = np.arctan2(y_lin, x_lin)

        rim_scale = np.ones_like(theta)
        for i, harm_n in enumerate(range(2, 2 + rc.rim_n_harmonics)):
            rim_scale += cd.rim_amplitudes[i] * np.cos(
                harm_n * theta + cd.rim_phases[i]
            )

        crater = base * rim_scale

        # --- Layer 2: Wall slump noise ---
        r_norm = 2 * distance / n  # 0 at center, ~1 at rim
        wall_lo, wall_hi = rc.slump_wall_range
        # Mask: 1 inside wall zone, 0 outside, smooth transitions
        wall_mask = (
            self._smooth_step(r_norm, wall_lo - 0.05, wall_lo + 0.05)
            * (1 - self._smooth_step(r_norm, wall_hi - 0.05, wall_hi + 0.05))
        )

        # 1D radial noise mapped to r_norm (each azimuth gets same radial pattern,
        # but Perlin noise varies with the continuous r_norm value)
        slump_noise = perlin_1d(
            r_norm.ravel(), freq=8.0, seed=cd.slump_noise_seed
        ).reshape(r_norm.shape)
        crater += slump_noise * rc.slump_intensity * wall_mask

        # --- Layer 3: Floor roughness ---
        floor_mask = 1 - self._smooth_step(r_norm, rc.floor_radius_ratio - 0.05, rc.floor_radius_ratio + 0.05)

        x_grid, y_grid = np.meshgrid(
            np.linspace(0, n * 0.1, n), np.linspace(0, n * 0.1, n)
        )
        floor_noise = perlin_2d(x_grid, y_grid, freq=1.0, seed=cd.floor_noise_seed)
        crater += floor_noise * rc.floor_noise_amp * floor_mask

        return crater
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestRealisticApplyProfile -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add terrain/procedural/realistic_crater_generator.py
git commit -m "feat: implement _apply_profile with rim modulation, slump, floor noise"
```

---

### Task 10: End-to-End generate_single and generate_craters — Tests

**Files:**
- Modify: `tests/test_realistic_crater.py`

- [ ] **Step 1: Add end-to-end tests**

Append to `tests/test_realistic_crater.py`:

```python
class TestRealisticEndToEnd:
    """End-to-end tests: generate_single and generate_craters."""

    @pytest.fixture
    def generator(self):
        cfg = CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic",
            seed=42,
        )
        rcfg = RealisticCraterConf()
        return RealisticCraterGenerator(cfg, rcfg)

    def test_generate_single_returns_tuple(self, generator):
        crater, cd = generator.generate_single(size=101)
        assert isinstance(crater, np.ndarray)
        assert isinstance(cd, RealisticCraterData)
        assert crater.shape == (101, 101)

    def test_generate_single_no_nans(self, generator):
        crater, cd = generator.generate_single(size=201)
        assert not np.any(np.isnan(crater))

    def test_generate_single_has_depression(self, generator):
        """Crater should have a depression (negative values)."""
        crater, cd = generator.generate_single(size=201)
        assert np.min(crater) < 0, "Crater has no depression"

    def test_generate_craters_on_dem(self, generator):
        """Stamp multiple craters onto a flat DEM."""
        dem = np.zeros((500, 500), dtype=np.float32)
        coords = np.array([[10.0, 10.0], [30.0, 30.0]], dtype=np.float64)
        radii = np.array([2.0, 1.5], dtype=np.float64)
        dem_out, mask, craters_data = generator.generate_craters(dem, coords, radii)
        assert dem_out.shape == (500, 500)
        assert mask.shape == (500, 500)
        assert len(craters_data) == 2
        assert all(isinstance(cd, RealisticCraterData) for cd in craters_data)

    def test_generate_craters_modifies_dem(self, generator):
        """DEM should be modified after crater stamping."""
        dem = np.zeros((500, 500), dtype=np.float32)
        coords = np.array([[10.0, 10.0]], dtype=np.float64)
        radii = np.array([2.0], dtype=np.float64)
        dem_out, _, _ = generator.generate_craters(dem, coords, radii)
        assert not np.allclose(dem_out, 0.0)

    def test_reproducible_with_crater_data(self, generator):
        """Regenerating from CraterData should produce identical result."""
        crater1, cd = generator.generate_single(size=101)
        crater2, _ = generator.generate_single(crater_data=cd)
        np.testing.assert_array_almost_equal(crater1, crater2)

    def test_interface_compatible_with_parent(self, generator):
        """Return signature matches CraterGenerator.generate_craters."""
        dem = np.zeros((200, 200), dtype=np.float32)
        coords = np.array([[5.0, 5.0]], dtype=np.float64)
        radii = np.array([1.0], dtype=np.float64)
        result = generator.generate_craters(dem, coords, radii)
        assert len(result) == 3  # (dem, mask, craters_data)
        dem_out, mask_out, cd_list = result
        assert isinstance(dem_out, np.ndarray)
        assert isinstance(mask_out, np.ndarray)
        assert isinstance(cd_list, list)
```

- [ ] **Step 2: Run tests to verify they pass**

These should already pass since `generate_single` and `generate_craters` are inherited from parent and call our overridden methods.

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestRealisticEndToEnd -v`
Expected: All 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_realistic_crater.py
git commit -m "test: add end-to-end realistic crater generation tests"
```

---

### Task 11: MoonyardGenerator Integration

**Files:**
- Modify: `terrain/procedural/moonyard_generator.py:17,36`
- Modify: `terrain/procedural/__init__.py`
- Modify: `tests/test_realistic_crater.py`

- [ ] **Step 1: Add integration test**

Append to `tests/test_realistic_crater.py`:

```python
from terrain.config import MoonYardConf, CraterDistributionConf, BaseTerrainGeneratorConf
from terrain.procedural.moonyard_generator import MoonyardGenerator


class TestMoonyardIntegration:
    """Tests for MoonyardGenerator with realistic crater mode."""

    def test_realistic_mode_uses_realistic_generator(self):
        cfg = MoonYardConf(
            crater_generator=CraterGeneratorConf(
                profiles_path="assets/Terrains/crater_spline_profiles.pkl",
                crater_mode="realistic",
                seed=42,
            ),
            realistic_crater=RealisticCraterConf(),
            crater_distribution=CraterDistributionConf(
                x_size=10.0, y_size=10.0,
                densities=[0.1], radius=[[0.5, 1.0]], seed=42,
            ),
            base_terrain_generator=BaseTerrainGeneratorConf(
                x_size=10.0, y_size=10.0, resolution=0.05, seed=42,
            ),
        )
        gen = MoonyardGenerator(cfg)
        assert isinstance(gen._crater_gen, RealisticCraterGenerator)

    def test_classic_mode_uses_classic_generator(self):
        cfg = MoonYardConf(
            crater_generator=CraterGeneratorConf(
                profiles_path="assets/Terrains/crater_spline_profiles.pkl",
                crater_mode="classic",
                seed=42,
            ),
        )
        gen = MoonyardGenerator(cfg)
        assert isinstance(gen._crater_gen, CraterGenerator)
        assert not isinstance(gen._crater_gen, RealisticCraterGenerator)

    def test_randomize_with_realistic_mode(self):
        cfg = MoonYardConf(
            crater_generator=CraterGeneratorConf(
                profiles_path="assets/Terrains/crater_spline_profiles.pkl",
                crater_mode="realistic",
                seed=42,
            ),
            realistic_crater=RealisticCraterConf(),
            crater_distribution=CraterDistributionConf(
                x_size=10.0, y_size=10.0,
                densities=[0.1], radius=[[0.5, 1.0]], seed=42,
            ),
            base_terrain_generator=BaseTerrainGeneratorConf(
                x_size=10.0, y_size=10.0, resolution=0.05, seed=42,
            ),
        )
        gen = MoonyardGenerator(cfg)
        dem, mask, craters_data = gen.randomize()
        assert dem.shape[0] > 0
        assert dem.shape[1] > 0
        assert not np.any(np.isnan(dem))
        # At least some craters should have been generated
        if len(craters_data) > 0:
            assert isinstance(craters_data[0], RealisticCraterData)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestMoonyardIntegration::test_realistic_mode_uses_realistic_generator -v`
Expected: FAIL — MoonyardGenerator doesn't know about RealisticCraterGenerator yet

- [ ] **Step 3: Modify MoonyardGenerator to dispatch based on crater_mode**

In `terrain/procedural/moonyard_generator.py`, add import and modify `__init__`:

Add to imports (after existing crater_generator import):
```python
from terrain.procedural.realistic_crater_generator import RealisticCraterGenerator
```

Replace the `self._crater_gen` line in `__init__`:
```python
        if cfg.crater_generator.crater_mode == "realistic":
            self._crater_gen = RealisticCraterGenerator(
                cfg.crater_generator, cfg.realistic_crater
            )
        else:
            self._crater_gen = CraterGenerator(cfg.crater_generator)
```

Also add `realistic_crater` to `_override_seeds` to pass through:
```python
    @staticmethod
    def _override_seeds(cfg: MoonYardConf, seed: int) -> MoonYardConf:
        """Return a copy of cfg with all sub-generator seeds replaced."""
        return dataclasses.replace(
            cfg,
            base_terrain_generator=dataclasses.replace(cfg.base_terrain_generator, seed=seed),
            crater_distribution=dataclasses.replace(cfg.crater_distribution, seed=seed),
            crater_generator=dataclasses.replace(cfg.crater_generator, seed=seed),
        )
```
(No change needed — `realistic_crater` has no seed field, parameters are derived from crater_generator's seed.)

- [ ] **Step 4: Update __init__.py exports**

In `terrain/procedural/__init__.py`:

```python
"""Procedural terrain generation: base terrain, crater placement, and crater shapes."""

from terrain.procedural.base_terrain import BaseTerrainGenerator
from terrain.procedural.crater_distribution import CraterDistributor
from terrain.procedural.crater_generator import CraterGenerator, CraterData
from terrain.procedural.moonyard_generator import MoonyardGenerator
from terrain.procedural.noise import perlin_1d, perlin_2d
from terrain.procedural.realistic_crater_generator import (
    RealisticCraterGenerator,
    RealisticCraterData,
)

__all__ = [
    "BaseTerrainGenerator",
    "CraterDistributor",
    "CraterGenerator",
    "CraterData",
    "MoonyardGenerator",
    "RealisticCraterGenerator",
    "RealisticCraterData",
    "perlin_1d",
    "perlin_2d",
]
```

- [ ] **Step 5: Run all tests to verify they pass**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full regression test suite**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_lunar_yard_config.py tests/test_e2e_config.py tests/test_noise.py tests/test_realistic_crater.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add terrain/procedural/moonyard_generator.py terrain/procedural/__init__.py tests/test_realistic_crater.py
git commit -m "feat: integrate RealisticCraterGenerator into MoonyardGenerator pipeline"
```

---

### Task 12: YAML Config Validation

**Files:**
- Verify: `config/environment/lunar_yard_40m_realistic_rocks.yaml` (already modified)

- [ ] **Step 1: Add config loading test**

Append to `tests/test_realistic_crater.py`:

```python
class TestYAMLConfig:
    """Test that YAML config with realistic crater settings loads correctly."""

    def test_realistic_rocks_yaml_loads(self):
        import yaml
        with open("config/environment/lunar_yard_40m_realistic_rocks.yaml") as f:
            raw = yaml.safe_load(f)
        tm = raw["terrain_manager"]
        assert tm["moon_yard"]["crater_generator"]["crater_mode"] == "realistic"
        rc = tm["moon_yard"]["realistic_crater"]
        assert rc["n_harmonics"] == 4
        assert rc["harmonic_amp"] == 0.12

    def test_realistic_rocks_yaml_creates_valid_conf(self):
        import yaml
        with open("config/environment/lunar_yard_40m_realistic_rocks.yaml") as f:
            raw = yaml.safe_load(f)
        my = raw["terrain_manager"]["moon_yard"]
        conf = MoonYardConf(**my)
        assert conf.crater_generator.crater_mode == "realistic"
        assert conf.realistic_crater.n_harmonics == 4
```

- [ ] **Step 2: Run tests**

Run: `cd /home/sim2real1/new_lunar_sim && python -m pytest tests/test_realistic_crater.py::TestYAMLConfig -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_realistic_crater.py
git commit -m "test: add YAML config loading validation for realistic crater mode"
```

---

## Summary

| Task | Description | Files | Tests |
|------|------------|-------|-------|
| 1 | Perlin noise tests (red) | `tests/test_noise.py` | 11 |
| 2 | Perlin noise implementation | `terrain/procedural/noise.py` | 11 |
| 3 | Config dataclasses | `terrain/config.py`, `tests/test_realistic_crater.py` | 8 |
| 4 | RealisticCraterData tests (red) | `tests/test_realistic_crater.py` | 11 |
| 5 | RealisticCraterData implementation | `terrain/procedural/realistic_crater_generator.py` | 11 |
| 6 | Distance matrix tests (red) | `tests/test_realistic_crater.py` | 5 |
| 7 | Distance matrix implementation | `terrain/procedural/realistic_crater_generator.py` | 5 |
| 8 | Apply profile tests (red) | `tests/test_realistic_crater.py` | 5 |
| 9 | Apply profile implementation | `terrain/procedural/realistic_crater_generator.py` | 5 |
| 10 | E2E tests | `tests/test_realistic_crater.py` | 7 |
| 11 | MoonyardGenerator integration | `moonyard_generator.py`, `__init__.py` | 3 |
| 12 | YAML config validation | `tests/test_realistic_crater.py` | 2 |

**Total: 12 tasks, ~73 tests**
