"""Bekker-Janosi terramechanics solver (vectorized).

Computes wheel-terrain interaction forces and torques based on
sinkage, velocity, and angular velocity per wheel.

Forces computed:
    Fx: Drawbar pull (longitudinal traction)
    Fy: Lateral resistance force from side-slip (Mohr-Coulomb shear)
    Fz: Normal force (not applied — PhysX handles this)
Torques computed:
    My: Resistance/overturn torque

Improvements over OmniLRS reference implementation:
    - Configurable gravity (lunar 1.625 m/s^2 default)
    - Realistic lunar regolith Bekker parameters
    - Gauss-Legendre quadrature instead of scipy.integrate.quad
    - Fully vectorized across all wheels (no Python for-loop)
    - Slip-sinkage coupling (Lyasko 2010)
    - Grouser (lug) correction for effective radius, shear width, and soil trapping
    - Lateral force (Fy) from side-slip angle via Mohr-Coulomb soil shear

Reference: OmniLRS src/physics/terramechanics_solver.py
"""

import logging

import numpy as np

from physics.terramechanics_parameters import RobotParameter, TerrainMechanicalParameter

logger = logging.getLogger(__name__)

# Pre-compute Gauss-Legendre nodes and weights (12-point, sufficient for
# the smooth integrands in the Bekker-Janosi model).
_GL_ORDER = 12
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_ORDER)


def _gl_integrate(f_vals: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gauss-Legendre quadrature over [a, b] for pre-evaluated function values.

    Args:
        f_vals: Function values at quadrature nodes, shape (..., _GL_ORDER).
        a: Lower bounds, shape (...).
        b: Upper bounds, shape (...).

    Returns:
        Integral estimates, shape (...).
    """
    half_width = (b - a) / 2.0  # (...)
    return half_width * np.einsum("...q,q->...", f_vals, _GL_WEIGHTS)


def _gl_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Map GL nodes from [-1, 1] to [a, b].

    Args:
        a: Lower bounds, shape (N,).
        b: Upper bounds, shape (N,).

    Returns:
        Quadrature points, shape (N, _GL_ORDER).
    """
    mid = (a + b) / 2.0
    half = (b - a) / 2.0
    # (N, 1) + (N, 1) * (Q,) -> (N, Q)
    return mid[:, None] + half[:, None] * _GL_NODES[None, :]


class TerramechanicsSolver:
    """Vectorized Bekker-Janosi terramechanics model.

    All per-wheel computations are batched via NumPy broadcasting.
    No Python for-loops over wheels.

    Interface:
        compute_force_and_torque(velocity, omega, sinkage)
            -> (forces [N,3], torques [N,3])
    """

    def __init__(
        self,
        robot_param: RobotParameter = None,
        terrain_param: TerrainMechanicalParameter = None,
    ) -> None:
        if robot_param is None:
            robot_param = RobotParameter()
        if terrain_param is None:
            terrain_param = TerrainMechanicalParameter()

        self._robot = robot_param
        self._terrain = terrain_param
        self._num_wheels = robot_param.num_wheels

    def compute_force_and_torque(
        self,
        velocity: np.ndarray,
        omega: np.ndarray,
        sinkage: np.ndarray,
        soil_multipliers: np.ndarray = None,
        lateral_speed: np.ndarray = None,
        contact_fz: np.ndarray = None,
        physx_mu: float | None = None,
    ) -> tuple:
        """Compute forces and torques for all wheels (vectorized).

        Args:
            velocity: Forward velocity per wheel (num_wheels,).
            omega: Angular velocity per wheel (num_wheels,).
            sinkage: Sinkage depth per wheel (num_wheels,).
            soil_multipliers: Per-wheel soil variation factors (num_wheels,).
                Multiplies K and divides (c, tan_phi) to create softer patches.
                1.0 = nominal, >1 = looser/slippier, <1 = more compact/grippy.
                None = uniform soil (all 1.0).
            lateral_speed: Sideways velocity per wheel (num_wheels,).
                Positive = wheel sliding to the right in its local frame.
                Used to compute lateral resistance force (Fy).
                None = no lateral force computed.
            contact_fz: Normal contact force per wheel (num_wheels,), positive.
                Used to cap Fx and Fy to the friction limit (mu * Fz).
                None = no friction limit applied.
            physx_mu: PhysX rigid-surface friction coefficient.
                When provided, computes a traction deficit correction:
                soft soil provides less traction than rigid surface (tan(phi) < physx_mu),
                so a negative force is applied at high slip to reduce effective traction
                to what the soil can actually sustain. None = no correction.

        Returns:
            forces: (num_wheels, 3) -- [Fx, Fy, 0] per wheel.
            torques: (num_wheels, 3) -- [0, My, 0] per wheel.
        """
        N = self._num_wheels
        r_base = self._robot.wheel_radius
        b = self._robot.wheel_width
        h_lug = self._robot.wheel_lug_height
        n_lug = self._robot.wheel_lug_count
        t = self._terrain

        # --- Grouser correction ---
        # Grousers protrude outward from the wheel surface, increasing the
        # effective radius and the shear failure area in the soil.
        #
        # 1) Effective radius: the grouser tips define the outer contact circle.
        r = r_base + h_lug
        #
        # 2) Effective shear width: soil shears along both side walls of each
        #    grouser, adding 2*h_lug to the contact patch width.
        b_shear = b + 2.0 * h_lug
        #
        # 3) Soil trapping ratio: soil trapped between grousers acts as part
        #    of the wheel, reducing the shear deformation modulus K.
        #    The ratio is based on the fraction of the circumference covered
        #    by the grouser gaps that can retain soil.
        if n_lug > 0 and h_lug > 0:
            # Approximate grouser spacing along the circumference.
            # Assume each grouser is thin; the gap between grousers traps soil
            # up to a depth of h_lug. The trapping efficiency depends on
            # h_lug relative to the gap width.
            gap = (2.0 * np.pi * r_base) / n_lug  # arc-length between grousers
            # Soil retention ratio: deeper lugs relative to gap retain more soil.
            # Saturates at 1.0 (fully trapped) when h_lug >= gap.
            trap_ratio = min(h_lug / gap, 1.0) if gap > 0 else 0.0
        else:
            trap_ratio = 0.0

        velocity = np.asarray(velocity, dtype=np.float64).ravel()[:N]
        omega = np.asarray(omega, dtype=np.float64).ravel()[:N]
        sinkage = np.asarray(sinkage, dtype=np.float64).ravel()[:N]

        if soil_multipliers is None:
            soil_multipliers = np.ones(N)
        else:
            soil_multipliers = np.asarray(soil_multipliers, dtype=np.float64).ravel()[:N]

        # --- Slip ratio (N,) ---
        slip = self._compute_slip_ratio_vec(velocity, omega)

        # --- Slip-sinkage coupling (Lyasko 2010) ---
        # Effective sinkage increases with slip magnitude
        z_eff = sinkage * (1.0 + t.slip_sinkage_coeff * np.abs(slip))
        z_eff = np.clip(z_eff, 0.0, r * 0.99)

        # --- Contact angles (N,) ---
        # theta_f: entry angle from sinkage depth
        theta_f = np.where(z_eff > 0, np.arccos(1.0 - z_eff / r), 0.0)
        theta_r = np.zeros(N)
        theta_m_raw = (t.a_0 + t.a_1 * slip) * theta_f
        theta_m = np.clip(theta_m_raw, theta_r, theta_f)

        # --- sigma_max: standard Bekker pressure-sinkage ---
        # p(z) = (k_c/b + k_phi) * z^n, where z = r*(cos(theta) - cos(theta_f))
        # We factor out: sigma_max = (k_c/b + k_phi) * r^n
        # Then sigma(theta) = sigma_max * (cos(theta) - cos(theta_f))^n
        sigma_max = (t.k_c / b + t.k_phi) * r ** t.n  # scalar

        # Skip computation for zero-contact wheels
        active = theta_f > 1e-12  # (N,)
        if not np.any(active):
            return np.zeros((N, 3)), np.zeros((N, 3))

        # --- Quadrature points ---
        # Lower region: [theta_r, theta_m], Upper region: [theta_m, theta_f]
        pts_lower = _gl_points(theta_r, theta_m)  # (N, Q)
        pts_upper = _gl_points(theta_m, theta_f)   # (N, Q)

        # --- Normal stress at quadrature points ---
        # Bekker: sigma = (k_c/b + k_phi) * [r*(cos(theta) - cos(theta_f))]^n
        #       = sigma_max * (cos(theta) - cos(theta_f))^n

        # Lower region (theta_r to theta_m): Wong-Reece rear stress
        denom_lower = np.where(
            np.abs(theta_m - theta_r) > 1e-12,
            theta_m - theta_r,
            1.0,
        )  # (N,)
        ratio_lower = (pts_lower - theta_r[:, None]) / denom_lower[:, None]  # (N, Q)
        delta_cos_lower = np.maximum(
            np.cos(theta_f[:, None] - ratio_lower * (theta_f[:, None] - theta_m[:, None]))
            - np.cos(theta_f[:, None]),
            0.0,
        )  # (N, Q)
        sigma_lower = sigma_max * delta_cos_lower ** t.n  # (N, Q)

        # Upper region (theta_m to theta_f): Wong-Reece front stress
        delta_cos_upper = np.maximum(
            np.cos(pts_upper) - np.cos(theta_f[:, None]),
            0.0,
        )  # (N, Q)
        sigma_upper = sigma_max * delta_cos_upper ** t.n  # (N, Q)

        # --- Shear displacement j(theta) ---
        j_lower = r * (
            theta_f[:, None] - pts_lower
            - (1 - slip[:, None]) * (np.sin(theta_f[:, None]) - np.sin(pts_lower))
        )  # (N, Q)
        j_upper = r * (
            theta_f[:, None] - pts_upper
            - (1 - slip[:, None]) * (np.sin(theta_f[:, None]) - np.sin(pts_upper))
        )  # (N, Q)

        # --- Shear stress (Janosi-Hanamoto) with soil heterogeneity + grouser ---
        # soil_multipliers > 1 = looser soil: K increases, shear strength decreases
        sm = soil_multipliers[:, None]  # (N, 1) for broadcasting with (N, Q)
        # Grouser soil-trapping effect: trapped soil between grousers reduces
        # the shear deformation modulus K, meaning traction mobilizes faster.
        K_grouser = t.K * (1.0 - 0.5 * trap_ratio)  # up to 50% reduction
        K_eff = K_grouser * sm         # combine with spatial heterogeneity
        c_eff = t.c / sm               # looser -> less cohesion
        tan_phi_eff = np.tan(t.phi) / sm  # looser -> lower friction angle

        # Janosi-Hanamoto saturation factor: (1 - exp(-j/K))
        # For positive j (driving): saturates to +1 → tau approaches +(c + σ·tanφ)
        # For negative j (braking): the raw formula diverges exponentially,
        # producing |tau| >> Mohr-Coulomb capacity. This is non-physical —
        # soil cannot resist more than its shear strength in either direction.
        # Fix: clamp the saturation factor to [-1, +1].
        sat_lower = np.clip(1.0 - np.exp(-j_lower / K_eff), -1.0, 1.0)
        tau_lower = (c_eff + sigma_lower * tan_phi_eff) * sat_lower

        sat_upper = np.clip(1.0 - np.exp(-j_upper / K_eff), -1.0, 1.0)
        tau_upper = (c_eff + sigma_upper * tan_phi_eff) * sat_upper

        # --- Integrand evaluation ---
        cos_lower = np.cos(pts_lower)
        sin_lower = np.sin(pts_lower)
        cos_upper = np.cos(pts_upper)
        sin_upper = np.sin(pts_upper)

        # Fx integrands: tau*cos(theta) - sigma*sin(theta)
        fx_integrand_lower = tau_lower * cos_lower - sigma_lower * sin_lower
        fx_integrand_upper = tau_upper * cos_upper - sigma_upper * sin_upper

        # My integrands: tau(theta)
        my_integrand_lower = tau_lower
        my_integrand_upper = tau_upper

        # --- Integration ---
        # Fx uses b_shear (grouser side walls add shear area for traction).
        # My uses b_shear as well (resistance torque from the wider shear zone).
        # sigma (normal stress) in the Bekker equation still uses the original
        # wheel width b, since ground pressure depends on the physical footprint.
        fx_lower = _gl_integrate(fx_integrand_lower, theta_r, theta_m)
        fx_upper = _gl_integrate(fx_integrand_upper, theta_m, theta_f)
        fx = r * b_shear * (fx_lower + fx_upper)  # (N,)

        my_lower = _gl_integrate(my_integrand_lower, theta_r, theta_m)
        my_upper = _gl_integrate(my_integrand_upper, theta_m, theta_f)
        my = r * r * b_shear * (my_lower + my_upper)  # (N,)

        # --- Lateral force (Fy) from side-slip ---
        # When a wheel slides sideways, the soil resists via Mohr-Coulomb
        # shear along the contact patch. The lateral force opposes the
        # side-slip direction.
        #
        # Uses slip-angle based saturation (not raw velocity, which has
        # wrong dimensions and causes force spikes from PhysX jitter).
        # Slip angle: alpha = atan2(|v_lat|, |v_fwd|)
        # Saturation: 1 - exp(-alpha / alpha_sat), where alpha_sat ~ 0.15 rad
        # (typical soil lateral shear mobilization angle, ~8.6 degrees).
        if lateral_speed is not None:
            v_lat = np.asarray(lateral_speed, dtype=np.float64).ravel()[:N]
        else:
            v_lat = np.zeros(N)

        # Dead zone: ignore lateral velocities below noise floor to prevent
        # PhysX vibration from generating spurious lateral forces.
        _LAT_DEADZONE = 0.05  # m/s — below this, no lateral force
        has_lateral = np.abs(v_lat) > _LAT_DEADZONE
        if np.any(has_lateral & active):
            # Lateral shear uses the same normal stress distribution.
            sigma_int_lower = _gl_integrate(sigma_lower, theta_r, theta_m)
            sigma_int_upper = _gl_integrate(sigma_upper, theta_m, theta_f)
            sigma_integral = sigma_int_lower + sigma_int_upper  # (N,)

            # Mohr-Coulomb lateral shear capacity per unit width:
            c_lat = t.c / soil_multipliers
            tan_phi_lat = np.tan(t.phi) / soil_multipliers
            arc_length = theta_f - theta_r  # (N,)
            tau_lat_capacity = r * (c_lat * arc_length + tan_phi_lat * sigma_integral)

            # Slip-angle based saturation (dimensionally consistent).
            # alpha_sat: characteristic angle at which ~63% of lateral
            # shear capacity is mobilized. Typical range: 0.1-0.3 rad.
            _ALPHA_SAT = 0.15  # rad (~8.6 degrees)
            v_fwd_abs = np.maximum(np.abs(velocity), 0.05)  # floor to avoid div-by-zero
            slip_angle = np.arctan2(np.abs(v_lat), v_fwd_abs)
            sat_factor = 1.0 - np.exp(-slip_angle / _ALPHA_SAT)

            # Fy opposes the lateral sliding direction
            fy = -np.sign(v_lat) * b_shear * tau_lat_capacity * sat_factor
            fy = np.where(active & has_lateral, fy, 0.0)
        else:
            fy = np.zeros(N)

        # Zero out inactive wheels
        fx = np.where(active, fx, 0.0)
        fy = np.where(active, fy, 0.0)
        my = np.where(active, my, 0.0)

        # --- Friction limit: Fy cannot exceed mu*Fz ---
        # This prevents lateral force from exceeding available friction.
        if contact_fz is not None:
            fz = np.asarray(contact_fz, dtype=np.float64).ravel()[:N]
            mu = np.tan(t.phi)
            friction_limit = mu * np.maximum(fz, 0.0)
            fy = np.clip(fy, -friction_limit, friction_limit)

        # --- Compaction resistance (bulldozing drag) ---
        # When a wheel sinks into soil, it must push soil ahead of it to
        # advance. This force opposes body translation (not wheel rotation),
        # so PhysX motor controllers cannot compensate for it.
        # Rc = b * z^2 * (k_c/b + k_phi) / (n+1)  [Bekker compaction theory]
        # Applied as a backward force on the wheel in the forward direction.
        bekker_mod = t.k_c / b + t.k_phi
        rc = b * sinkage ** 2 * bekker_mod / (t.n + 1.0)
        # Only apply when wheel is actively moving forward (not reversing)
        rc = np.where((active) & (np.abs(velocity) > 0.01), rc, 0.0)
        # Oppose forward motion direction
        rc_signed = -np.sign(velocity) * rc

        # --- Traction deficit correction ---
        # PhysX treats terrain as rigid, giving traction up to mu_physx * Fz.
        # Soft soil (Bekker model) provides less: max traction ≈ tan(phi) * Fz.
        # When physx_mu > tan(phi), PhysX overestimates traction. We correct
        # by applying a negative force that scales with slip — at high slip
        # the wheel is at the traction limit and the deficit matters most.
        #
        # traction_deficit = (physx_mu - soil_mu) * Fz * slip_factor * drive_dir
        #   → negative when driving on soft soil → reduces net forward force
        traction_deficit = np.zeros(N)
        if physx_mu is not None and contact_fz is not None:
            fz = np.asarray(contact_fz, dtype=np.float64).ravel()[:N]
            soil_mu = np.tan(t.phi)  # effective soil friction from Mohr-Coulomb
            mu_gap = max(physx_mu - soil_mu, 0.0)
            if mu_gap > 0:
                omega_r = omega * r
                relative_speed = omega_r - velocity
                # Drive direction: PhysX pushes body forward when wheel spins
                # faster than body moves (driving), backward when braking.
                drive_dir = np.where(
                    np.abs(relative_speed) > 0.05, np.sign(relative_speed), 0.0
                )
                # Slip-modulated: at low slip, PhysX and Bekker agree;
                # at high slip (>0.3), full deficit applies.
                slip_factor = np.clip(np.abs(slip) * 3.0, 0.0, 1.0)
                traction_deficit = -mu_gap * fz * drive_dir * slip_factor

        # --- Assemble output ---
        # Forces applied on top of PhysX rigid-body contact:
        #   Rc: compaction resistance (soil bulldozing drag)
        #   traction_deficit: reduces PhysX traction to soft-soil level
        #   Fy: lateral soil shear resistance
        #   My: rolling resistance torque (opposes wheel rotation)
        #
        # My sign convention:
        #   my = r²*b*∫tau dθ > 0 when driving (positive tau).
        #   This represents the torque the motor must OVERCOME to spin the
        #   wheel in deformable soil. Applied as -my so it OPPOSES wheel
        #   rotation (resistance), not aids it.
        forces = np.zeros((N, 3))
        torques = np.zeros((N, 3))
        forces[:, 0] = rc_signed + traction_deficit
        forces[:, 1] = fy
        torques[:, 1] = -my  # resistance opposes wheel rotation

        # Clamp to safe range — NaN/Inf forces crash PhysX.
        # Use per-wheel weight as reference: max force = 5x static wheel load.
        static_wheel_load = self._robot.mass * t.gravity / N
        _MAX_FORCE = max(static_wheel_load * 5.0, 50.0)  # at least 50 N floor
        np.nan_to_num(forces, copy=False, nan=0.0, posinf=_MAX_FORCE, neginf=-_MAX_FORCE)
        np.nan_to_num(torques, copy=False, nan=0.0, posinf=_MAX_FORCE, neginf=-_MAX_FORCE)
        np.clip(forces, -_MAX_FORCE, _MAX_FORCE, out=forces)
        np.clip(torques, -_MAX_FORCE, _MAX_FORCE, out=torques)

        return forces, torques

    def _compute_slip_ratio_vec(self, v: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Vectorized slip ratio computation for all wheels.

        Args:
            v: Forward velocity (N,).
            omega: Angular velocity (N,).

        Returns:
            Slip ratio (N,), clipped to [-1, 1].
        """
        r = self._robot.wheel_radius
        v = np.asarray(v, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)

        omega_r = omega * r
        both_small = (np.abs(omega_r) < 1e-6) & (np.abs(v) < 1e-6)
        pure_slide = (~both_small) & (np.abs(omega_r) < 1e-6)
        driving = (~both_small) & (~pure_slide) & (v <= omega_r)
        braking = (~both_small) & (~pure_slide) & (v > omega_r)

        slip = np.zeros_like(v)
        # driving: s = 1 - v/(omega*r)
        slip = np.where(driving, 1.0 - v / np.where(np.abs(omega_r) > 1e-6, omega_r, 1.0), slip)
        # braking: s = omega*r/v - 1
        slip = np.where(braking, omega_r / np.where(np.abs(v) > 1e-6, v, 1.0) - 1.0, slip)
        # pure sliding: sign depends on velocity direction.
        # v > 0, omega=0 → braking (slip=-1), soil resists forward motion
        # v < 0, omega=0 → backward drag (slip=+1), soil resists backward motion
        pure_slide_sign = -np.sign(np.where(np.abs(v) > 1e-6, v, 1.0))
        slip = np.where(pure_slide, pure_slide_sign, slip)

        return np.clip(slip, -1.0, 1.0)
