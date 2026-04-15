"""Realistic crater generator with irregular outlines and uneven rim shoulders.

Extends CraterGenerator with:
- Harmonic contour perturbation for irregular outlines
- Low-frequency azimuthal noise at the slope-break boundary (shoulder)
"""

import dataclasses
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import rotate

from terrain.config import CraterGeneratorConf, RealisticCraterConf
from terrain.procedural.crater_generator import CraterData, CraterGenerator
from terrain.procedural.noise import perlin_1d


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
        base_cd = super().randomize_parameters(index, size)

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

        harmonics_n = np.arange(2, 2 + rc.n_harmonics)
        rcd.harmonic_amplitudes = np.array([
            self._rng.uniform(0, rc.harmonic_amp / n) for n in harmonics_n
        ])
        rcd.harmonic_phases = self._rng.uniform(0, 2 * np.pi, rc.n_harmonics)
        rcd.contour_noise_seed = int(self._rng.integers(0, 2**31))

        rim_n = np.arange(2, 2 + rc.rim_n_harmonics)
        rcd.rim_amplitudes = np.array([
            self._rng.uniform(0, rc.rim_noise_amp / n) for n in rim_n
        ])
        rcd.rim_phases = self._rng.uniform(0, 2 * np.pi, rc.rim_n_harmonics)

        rcd.slump_noise_seed = int(self._rng.integers(0, 2**31))
        rcd.floor_noise_seed = int(self._rng.integers(0, 2**31))

        return rcd

    def _centered_distance_matrix(self, cd: CraterData) -> np.ndarray:
        """Build distance matrix with harmonic contour perturbation + Perlin noise."""
        if not isinstance(cd, RealisticCraterData):
            return super()._centered_distance_matrix(cd)

        n = cd.size
        rc = self._rcfg

        # Angular coordinate grid
        x_lin, y_lin = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        theta = np.arctan2(y_lin, x_lin)

        # Low-frequency harmonic contour perturbation (outline deformation only)
        contour = np.ones_like(theta)
        for i, harm_n in enumerate(range(2, 2 + rc.n_harmonics)):
            contour += cd.harmonic_amplitudes[i] * np.cos(
                harm_n * theta + cd.harmonic_phases[i]
            )

        # Euclidean distance with elliptical deformation
        x_idx, y_idx = np.meshgrid(range(n), range(n))
        dist = np.sqrt(
            ((x_idx - n / 2 + 1) / cd.xy_deformation_factor[0]) ** 2
            + ((y_idx - n / 2 + 1) / cd.xy_deformation_factor[1]) ** 2
        )

        # Apply contour perturbation
        dist = dist / contour

        # Rotate
        dist = rotate(dist, cd.rotation, reshape=False, cval=n / 2)

        # Clamp to crater radius
        dist[dist > n / 2] = n / 2

        return dist

    @staticmethod
    def _smooth_step(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
        """Hermite smooth step: 0 at edge0, 1 at edge1."""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0, 1)
        return t * t * (3 - 2 * t)

    def _apply_profile(self, distance: np.ndarray, cd: CraterData) -> np.ndarray:
        """Apply profile with additive positive noise at the shoulder only.

        Adds positive-only height perturbation around the shoulder region.
        Beyond the shoulder (rim and outer edge) the base profile is preserved.
        """
        if not isinstance(cd, RealisticCraterData):
            return super()._apply_profile(distance, cd)

        rc = self._rcfg
        n = cd.size
        profile_fn = self._profiles[cd.crater_profile_id]

        # --- Base profile (untouched) ---
        r_norm = 2 * distance / n  # 0 at center, ~1 at edge
        base = profile_fn(np.clip(r_norm, 0, 1))

        # --- Find shoulder: slope-break on outer wall ---
        r_1d = np.linspace(0.05, 0.95, 200)
        profile_1d = profile_fn(r_1d)
        slope_1d = np.abs(np.gradient(profile_1d, r_1d))
        peak_idx = np.argmax(slope_1d)
        threshold = slope_1d[peak_idx] * 0.3
        outer_offset = np.argmax(slope_1d[peak_idx:] < threshold)
        shoulder_r = r_1d[peak_idx + outer_offset]

        # --- Azimuthal noise (always positive) ---
        x_lin, y_lin = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        theta = np.arctan2(y_lin, x_lin)

        noise = np.zeros_like(theta)
        for i, harm_n in enumerate(range(2, 2 + rc.rim_n_harmonics)):
            noise += cd.rim_amplitudes[i] * np.cos(
                harm_n * theta + cd.rim_phases[i]
            )
        # fBm Perlin for natural detail
        amp = rc.rim_noise_amp * 0.5
        for octave in range(3):
            freq = 2.0 * (2 ** octave)
            noise += amp * perlin_1d(
                (theta + np.pi).ravel(), freq=freq,
                period=2 * np.pi,
                seed=cd.contour_noise_seed + octave,
            ).reshape(theta.shape)
            amp *= 0.5

        # Only positive perturbation (material accumulation at shoulder)
        noise = np.abs(noise)

        # --- Smooth height blending around shoulder ---
        # At the shoulder, full noise is added. Away from the shoulder,
        # the HEIGHT DIFFERENCE between (base + noise) and base is smoothly
        # reduced to zero — not the noise pattern itself.
        # This avoids radial grooves extending into the crater interior.

        # Compute per-azimuth noise value at exactly the shoulder radius.
        # This is a 1D value per angle (no radial structure).
        shoulder_noise = noise  # noise is already azimuth-only (no r dependence)

        # Blend factor: 1 at shoulder, 0 far away (both sides)
        falloff_inner = 0.01
        falloff_outer = 0.03
        blend = np.exp(-0.5 * np.where(
            r_norm <= shoulder_r,
            ((r_norm - shoulder_r) / falloff_inner) ** 2,
            ((r_norm - shoulder_r) / falloff_outer) ** 2,
        ))

        return base + shoulder_noise * blend
