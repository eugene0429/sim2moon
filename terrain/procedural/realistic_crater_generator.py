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
