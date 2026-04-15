"""Terramechanics parameter dataclasses.

Defines robot mechanical dimensions and terrain soil parameters
for the Bekker-Janosi terramechanics model.

Lunar regolith defaults are based on Apollo-era measurements and
GRC-1 simulant bevameter tests (Oravec et al., 2010).
"""

import dataclasses
import math


@dataclasses.dataclass
class RobotParameter:
    """Robot mechanical dimension parameters for terramechanics.

    Attributes:
        mass: Total rover mass (kg).
        num_wheels: Number of driven wheels.
        wheel_radius: Wheel outer radius (m).
        wheel_width: Wheel contact patch width (m).  Used as the 'b'
            dimension in the Bekker pressure-sinkage equation.
        wheel_lug_height: Grouser (lug) height protruding from wheel surface (m).
        wheel_lug_count: Number of grousers around wheel circumference.
    """

    mass: float = 20.0
    num_wheels: int = 4
    wheel_radius: float = 0.09
    wheel_width: float = 0.1
    wheel_lug_height: float = 0.02
    wheel_lug_count: int = 16

    def __post_init__(self):
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        if self.num_wheels <= 0:
            raise ValueError(f"num_wheels must be positive, got {self.num_wheels}")
        if self.wheel_radius <= 0:
            raise ValueError(f"wheel_radius must be positive, got {self.wheel_radius}")
        if self.wheel_width <= 0:
            raise ValueError(f"wheel_width must be positive, got {self.wheel_width}")
        if self.wheel_lug_height < 0:
            raise ValueError(f"wheel_lug_height must be non-negative, got {self.wheel_lug_height}")
        if self.wheel_lug_count < 0:
            raise ValueError(f"wheel_lug_count must be non-negative, got {self.wheel_lug_count}")


@dataclasses.dataclass
class TerrainMechanicalParameter:
    """Terrain soil mechanical parameters for Bekker-Janosi model.

    Defaults represent a medium-density lunar regolith at ~5-15 cm depth,
    derived from Apollo measurements and GRC-1 simulant data.

    Attributes:
        k_c: Cohesive modulus of deformation (N/m^(n+1)).
        k_phi: Frictional modulus of deformation (N/m^(n+2)).
        n: Sinkage exponent (dimensionless).
        c: Soil cohesion (Pa).
        phi: Internal friction angle (rad).
        K: Shear deformation modulus (m).
        rho: Soil bulk density (kg/m^3).
        gravity: Surface gravitational acceleration (m/s^2).
        a_0, a_1: Wong-Reece coefficients for theta_m computation.
        slip_sinkage_coeff: Slip-sinkage coupling coefficient (dimensionless).
            Effective sinkage = z * (1 + slip_sinkage_coeff * |slip_ratio|).
    """

    k_c: float = 1400.0
    k_phi: float = 820000.0
    n: float = 1.0
    c: float = 170.0
    phi: float = 35.0 * math.pi / 180.0
    K: float = 0.018
    rho: float = 1500.0
    gravity: float = 1.625
    a_0: float = 0.4
    a_1: float = 0.3
    slip_sinkage_coeff: float = 0.5

    # Spatial heterogeneity (non-uniform soil)
    heterogeneity: float = 0.0
    """Soil variation intensity (0 = uniform, 1 = max variation).
    When > 0, K, phi, and c vary spatially across the terrain using
    coherent noise, simulating patches of loose/compacted regolith."""

    heterogeneity_scale: float = 2.0
    """Spatial scale of soil patches in meters.
    Smaller = fine-grained variation, larger = broad patches."""

    heterogeneity_seed: int = 42
    """Random seed for reproducible soil variation patterns."""

    physx_mu: float = 0.0
    """PhysX rigid-surface friction coefficient. When > 0 and > tan(phi),
    a traction deficit correction is applied: soft soil provides less grip
    than a rigid surface, so at high slip PhysX overestimates traction.
    Set to the PhysX material friction value (typically 0.7 for rubber on rock).
    0 = no correction (legacy behavior)."""

    def __post_init__(self):
        if self.K <= 0:
            raise ValueError(f"K (shear modulus) must be positive, got {self.K}")
        if self.rho <= 0:
            raise ValueError(f"rho (density) must be positive, got {self.rho}")
        if self.gravity <= 0:
            raise ValueError(f"gravity must be positive, got {self.gravity}")
        if self.slip_sinkage_coeff < 0:
            raise ValueError(f"slip_sinkage_coeff must be non-negative, got {self.slip_sinkage_coeff}")
        if not 0.0 <= self.heterogeneity <= 1.0:
            raise ValueError(f"heterogeneity must be in [0, 1], got {self.heterogeneity}")
