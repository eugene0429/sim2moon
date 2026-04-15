# Terramechanics: Wheel-Terrain Interaction Model

## 1. Overview

This project implements a **Bekker-Janosi terramechanics model** on top of Isaac Sim's PhysX rigid-body physics engine. The model computes realistic wheel-terrain interaction forces for deformable regolith that PhysX alone cannot reproduce.

PhysX treats all surfaces as rigid bodies with uniform friction coefficients. On the Moon, terrain is soft regolith that deforms under wheel load, causing sinkage, slip, and traction loss behaviors fundamentally different from rigid contact. The terramechanics layer corrects for this by computing additional forces and applying them as external wrench inputs to each wheel's rigid body every physics step.

**Key files:**

| File | Role |
|------|------|
| [physics/terramechanics.py](../physics/terramechanics.py) | Core solver: Bekker-Janosi force/torque computation |
| [physics/terramechanics_parameters.py](../physics/terramechanics_parameters.py) | Parameter dataclasses for robot and soil |
| [environments/lunar_yard.py](../environments/lunar_yard.py) | Integration layer: reads PhysX state, calls solver, applies forces |
| [robots/rigid_body_group.py](../robots/rigid_body_group.py) | Physics interface: contact forces, velocities, force application |

---

## 2. System Architecture

```
PhysX simulation step
        │
        ▼
RobotRigidGroup.get_net_contact_forces()   ← contact Fz per wheel
RobotRigidGroup.get_velocities()           ← linear + angular velocity per wheel
RobotRigidGroup.get_pose()                 ← orientation per wheel (heading only)
        │
        ▼
LunarYardEnvironment.apply_terramechanics()
  ├── EMA filter on contact forces  (reduce PhysX jitter)
  ├── Sinkage computation           (invert Bekker pressure-sinkage)
  ├── Velocity projection           (world frame → wheel-local frame)
  ├── Soil heterogeneity sampling   (spatial noise per wheel position)
  └── TerramechanicsSolver.compute_force_and_torque()
            │
            ▼  (returns local-frame forces/torques)
        Frame transform             (local → world frame)
            │
            ▼
RobotRigidGroup.apply_force_torque()       ← applied to PhysX rigid bodies
        │
        ▼
PhysX integrates new state (next step)
```

The cycle runs every simulation step (default: 60 Hz at `physics_dt = 0.0166 s`).

---

## 3. Mathematical Model

### 3.1 Bekker Pressure-Sinkage Equation

When a wheel sinks into deformable soil to depth *z*, the normal stress σ beneath the contact patch follows Bekker's pressure-sinkage law:

```
σ(z) = (k_c/b + k_phi) · z^n
```

where:
- `k_c` — cohesive modulus of deformation (N/m^(n+1))
- `k_phi` — frictional modulus of deformation (N/m^(n+2))
- `b` — wheel width (m)
- `n` — sinkage exponent (dimensionless)
- `z` — sinkage depth (m)

For a cylindrical wheel of radius *r* entering the soil at entry angle θ_f, the sinkage at angle θ is:

```
z(θ) = r · (cos θ − cos θ_f)
```

### 3.2 Wong-Reece Stress Distribution

The normal stress is not symmetric across the contact arc. The maximum stress occurs at angle θ_m, computed from the Wong-Reece model:

```
θ_m = (a_0 + a_1 · slip) · θ_f
```

where `a_0 = 0.4`, `a_1 = 0.3` are empirical coefficients. This produces an asymmetric pressure distribution — higher at the front arc when driving, shifting rearward under braking.

- **Front arc** [θ_m → θ_f]: Standard Bekker formula applied directly
- **Rear arc** [θ_r → θ_m]: Stress interpolated from zero at θ_r to maximum at θ_m

### 3.3 Janosi-Hanamoto Shear Stress

Soil shear stress τ is governed by the Janosi-Hanamoto equation:

```
τ(θ) = (c + σ · tan φ) · (1 − exp(−j / K))
```

where:
- `c` — soil cohesion (Pa)
- `φ` — internal friction angle (rad)
- `K` — shear deformation modulus (m) — smaller = faster traction mobilization
- `j(θ)` — shear displacement (m)

Shear displacement accumulates along the contact arc:

```
j(θ) = r · [θ_f − θ − (1 − slip) · (sin θ_f − sin θ)]
```

**Braking correction:** The raw Janosi-Hanamoto formula diverges for negative *j* (braking), producing `|τ| >> Mohr-Coulomb capacity`. This is non-physical. The implementation clamps the saturation factor to [−1, +1]:

```python
sat = np.clip(1.0 - np.exp(-j / K), -1.0, 1.0)
```

### 3.4 Longitudinal Force and Resistance Torque

Integration over the contact arc (from θ_r to θ_f) yields:

**Drawbar pull (Fx):**
```
Fx = r · b · ∫[θ_r → θ_f] (τ · cos θ − σ · sin θ) dθ
```

**Rolling resistance torque (My):**
```
My = r² · b · ∫[θ_r → θ_f] τ(θ) dθ
```

My is applied as a negative torque on the wheel (opposes rotation). It represents the effort the motor must overcome to spin through deformable soil.

### 3.5 Slip Ratio

Slip ratio *s* quantifies how much the wheel is spinning relative to body motion:

```
Driving (v ≤ ω·r):   s = 1 − v / (ω · r)       ∈ [0, 1]
Braking (v > ω·r):   s = (ω · r) / v − 1         ∈ [−1, 0]
Pure slide (ω = 0):  s = −sign(v)                 (±1)
```

where `v` is forward wheel velocity (m/s) and `ω` is wheel angular velocity (rad/s).

---

## 4. Improvements Over Reference Implementation

This implementation significantly extends the OmniLRS reference (`src/physics/terramechanics_solver.py`):

| Feature | Reference | This Implementation |
|---------|-----------|---------------------|
| Gravity | Hard-coded 9.81 m/s² | Configurable (default: lunar 1.625 m/s²) |
| Soil defaults | Earth soil | Apollo/GRC-1 lunar regolith |
| Numerical integration | `scipy.integrate.quad` (per-wheel Python loop) | 12-point Gauss-Legendre quadrature, fully vectorized |
| Slip-sinkage coupling | Not implemented | Lyasko (2010) model |
| Grouser correction | Not implemented | Effective radius, shear width, soil trapping ratio |
| Lateral force (Fy) | Not implemented (zero) | Mohr-Coulomb with slip-angle saturation |
| Compaction resistance | Not implemented | Bekker bulldozing drag formula |
| PhysX traction correction | Not implemented | Traction deficit correction (physx_mu vs tan φ) |
| Friction limit | Not implemented | Fy capped at μ·Fz |
| Contact force filtering | Not implemented | Exponential moving average (EMA) |
| Soil heterogeneity | Not implemented | Multi-octave value noise per wheel position |
| Per-robot skip list | Not implemented | Selective enable/disable per robot name |

---

## 5. Key Implementation Features

### 5.1 Gauss-Legendre Quadrature (Vectorized)

The contact arc integrals are solved with 12-point Gauss-Legendre quadrature. All wheels are processed simultaneously using NumPy broadcasting — there are no Python for-loops over wheels.

```python
_GL_ORDER = 12
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_ORDER)
```

Integration over [a, b] for N wheels:
```python
# f_vals: (N, 12) — integrand at quadrature points for each wheel
half_width = (b - a) / 2.0   # (N,)
result = half_width * np.einsum("...q,q->...", f_vals, _GL_WEIGHTS)
```

### 5.2 Grouser (Lug) Correction

Grousers are the metal protrusions on rover wheels that dig into soil. The model accounts for three grouser effects:

**1. Effective radius:**
```
r_eff = r_base + h_lug
```
Grouser tips define the outer contact circle.

**2. Effective shear width:**
```
b_shear = b + 2 · h_lug
```
Soil shears along both side walls of each grouser, increasing the effective traction area.

**3. Soil trapping ratio:**
```
gap = 2π · r_base / n_lug
trap_ratio = min(h_lug / gap, 1.0)
K_grouser = K · (1 − 0.5 · trap_ratio)   ← up to 50% reduction in K
```
Soil trapped between grousers reduces the shear deformation modulus *K*, meaning traction is mobilized more quickly with less slip required.

### 5.3 Slip-Sinkage Coupling (Lyasko 2010)

Sinkage increases with slip because spinning wheels excavate soil:

```
z_eff = z · (1 + slip_sinkage_coeff · |slip|)
```

This is capped at 99% of wheel radius to prevent numerical instability. On loose lunar regolith with `slip_sinkage_coeff = 2.0`, a wheel at 50% slip sinks twice as deep as static sinkage.

### 5.4 Lateral Force — Mohr-Coulomb Shear (Fy)

When a wheel slides sideways, the soil resists via Mohr-Coulomb shear:

```
τ_lat_capacity = r · (c · arc_length + tan φ · ∫σ dθ)
```

The force is saturated by slip angle rather than raw lateral velocity (avoids force spikes from PhysX vibration):

```
α = atan2(|v_lat|, |v_fwd|)                    ← slip angle
sat = 1 − exp(−α / α_sat),  α_sat = 0.15 rad   ← ~8.6° characteristic angle
Fy = −sign(v_lat) · b_shear · τ_lat_capacity · sat
```

A 0.05 m/s dead zone ignores small lateral velocities from PhysX numerical noise.

Additionally, Fy is capped by the friction limit:
```
|Fy| ≤ tan(φ) · Fz
```

### 5.5 Compaction Resistance (Rc)

As the wheel advances, it must push soil ahead (bulldozing drag). This force opposes forward motion and cannot be compensated by motor controllers:

```
Rc = b · z² · (k_c/b + k_phi) / (n + 1)
```

Applied as: `force_x += −sign(v) · Rc`

This is the Bekker compaction resistance formula. It scales with sinkage squared — on very soft regolith with deep sinkage, this is a significant drag.

### 5.6 Traction Deficit Correction

PhysX treats the terrain as a rigid surface with friction coefficient `physx_mu` (typically 0.7 for rubber on rock). Soft lunar regolith provides much less traction — approximately `tan(φ)` times normal force.

At high slip, the gap between PhysX traction and actual soil capacity creates unrealistically high forward thrust. The correction applies a counter-force:

```
soil_mu = tan(φ)
mu_gap = max(physx_mu − soil_mu, 0)

slip_factor = clip(|slip| · 3.0, 0, 1)    ← ramps in over slip range [0, 0.33]
drive_dir = sign(ω·r − v)                  ← +1 driving, −1 braking

traction_deficit = −mu_gap · Fz · drive_dir · slip_factor
```

This creates a negative force at high slip that reduces net thrust to what soft soil can actually sustain. At low slip, both models agree and no correction is needed.

**Example:** For `physx_mu = 0.7`, `phi = 20°` → `tan(phi) = 0.364`:
- `mu_gap = 0.7 − 0.364 = 0.336`
- At full slip, deficit force ≈ −0.336 × Fz per wheel

### 5.7 Sinkage Computation from Contact Force

Since PhysX treats terrain as rigid, wheels don't geometrically penetrate — they bounce off. Sinkage is inferred by inverting the Bekker pressure-sinkage equation using the measured PhysX contact force:

```
Fz ≈ 2·b·(k_c/b + k_phi)·√(2r) · z^(n + 0.5)
z = (Fz / denom)^(1/(n + 0.5))
```

Contact force is clamped to `3 × static_wheel_load` to reject PhysX impact/bounce spikes while allowing weight transfer on slopes. Sinkage is capped at `0.5 × wheel_radius`.

### 5.8 EMA Filter on Contact Forces

PhysX contact forces are inherently noisy. An exponential moving average (EMA) smooths them:

```
EMA(t) = α · EMA(t−1) + (1 − α) · raw(t)
```

- `α = 0` — no filtering (raw PhysX values)
- `α = 0.7` (default) — moderate smoothing, lag ≈ 3 frames

Initialized to zero so the initial landing impact ramps in gradually rather than causing a first-frame spike.

A separate EMA state is maintained per robot name (`_contact_ema` dict), enabling independent filtering for multi-robot scenarios.

### 5.9 Soil Heterogeneity

Spatially varying soil properties simulate patches of loose and compacted regolith. At each wheel position, a multi-octave value noise generates a `soil_multiplier`:

```
multiplier > 1.0 → looser soil: K increases, c and tan(φ) decrease
multiplier < 1.0 → more compact: K decreases, c and tan(φ) increase
```

Three octaves are summed (1/scale, 2/scale, 4/scale frequencies) with amplitudes 1, 0.5, 0.25, producing natural-looking broad patches with fine detail. The hash function is deterministic — the same world position always produces the same soil type (no temporal flickering).

Configuration:
```yaml
heterogeneity: 0.5       # variation intensity (0 = uniform)
heterogeneity_scale: 1.5 # spatial scale of patches (meters)
heterogeneity_seed: 42   # reproducibility seed
```

### 5.10 Per-Robot Skip List

Certain robots can be excluded from terramechanics, keeping PhysX-only rigid-body behavior. This is useful for side-by-side comparison experiments:

```yaml
terramechanics_settings:
  skip_robots: ["husky_physx"]
```

All robots NOT in `skip_robots` receive Bekker-Janosi forces. Robots in `skip_robots` use only PhysX friction.

---

## 6. Force Pipeline Summary

Each simulation step, the following forces and torques are assembled and applied to wheel rigid bodies:

| Component | Direction | Description |
|-----------|-----------|-------------|
| `Rc` (compaction resistance) | −forward | Bulldozing drag from sinking into soil |
| `traction_deficit` | −forward | Corrects PhysX over-traction on soft soil |
| `Fy` (lateral shear) | lateral | Mohr-Coulomb resistance to side slip |
| `My` (rolling resistance) | around axle | Torque cost of spinning through deformable soil |

Note: **Fz is NOT applied** — PhysX rigid-body collision handles the normal reaction. The terramechanics layer only adds the deformable-soil corrections on top.

Forces are computed in wheel-local frame (Fx along forward, Fy along lateral, My around axle), then rotated to world frame before application:

```python
world_forces  = Fx · forward_dir + Fy · lateral_dir
world_torques = My · axle_dir
```

A per-wheel safety clamp prevents NaN/Inf from crashing PhysX:
```
max_force = max(5 × static_wheel_load, 50 N)
```

---

## 7. Parameter Reference

### 7.1 Soil Parameters (`TerrainMechanicalParameter`)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `k_c` | 1400 | N/m^(n+1) | Cohesive modulus of deformation |
| `k_phi` | 820000 | N/m^(n+2) | Frictional modulus of deformation |
| `n` | 1.0 | — | Sinkage exponent |
| `c` | 170 | Pa | Soil cohesion |
| `phi` | 35° | rad | Internal friction angle |
| `K` | 0.018 | m | Shear deformation modulus |
| `rho` | 1500 | kg/m³ | Soil bulk density |
| `gravity` | 1.625 | m/s² | Surface gravity |
| `a_0` | 0.4 | — | Wong-Reece θ_m coefficient |
| `a_1` | 0.3 | — | Wong-Reece slip coefficient |
| `slip_sinkage_coeff` | 0.5 | — | Lyasko slip-sinkage coupling |
| `heterogeneity` | 0.0 | — | Soil variation intensity [0–1] |
| `heterogeneity_scale` | 2.0 | m | Spatial scale of soil patches |
| `heterogeneity_seed` | 42 | — | Reproducibility seed |
| `physx_mu` | 0.0 | — | PhysX friction coefficient (0 = no correction) |

Defaults represent medium-density lunar regolith at ~5–15 cm depth (Apollo measurements + GRC-1 simulant data, Oravec et al. 2010).

### 7.2 Robot Parameters (`RobotParameter`)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `mass` | 20.0 | kg | Total rover mass |
| `num_wheels` | 4 | — | Number of driven wheels |
| `wheel_radius` | 0.09 | m | Wheel outer radius |
| `wheel_width` | 0.1 | m | Contact patch width (b in Bekker eq.) |
| `wheel_lug_height` | 0.02 | m | Grouser protrusion height |
| `wheel_lug_count` | 16 | — | Number of grousers per wheel |

---

## 8. YAML Configuration

```yaml
terramechanics_settings:
  enable: true
  contact_force_ema: 0.7          # EMA smoothing (0 = raw, 0.7 = smooth)
  skip_robots: ["husky_physx"]    # PhysX-only robots (baseline comparison)

  robot:
    mass: 50.0                    # Husky rover mass (kg)
    num_wheels: 4
    wheel_radius: 0.165
    wheel_width: 0.1
    wheel_lug_height: 0.01
    wheel_lug_count: 16

  soil:
    # Loose lunar regolith (playground config — tuned for visible slip)
    k_c: 200.0
    k_phi: 20000.0
    n: 0.7
    c: 10.0
    phi_deg: 20.0                 # converted to radians internally
    K: 0.025
    rho: 1300.0
    gravity: 1.625
    physx_mu: 0.7                 # enables traction deficit correction
    slip_sinkage_coeff: 2.0
    heterogeneity: 0.5
    heterogeneity_scale: 1.5
    heterogeneity_seed: 42
```

---

## 9. Dual-Rover Comparison Setup

The `lunar_yard_20m_playground.yaml` configuration places two Husky rovers side-by-side:

- **`husky`** — Bekker-Janosi terramechanics enabled. Experiences sinkage, compaction drag, and traction deficit on soft regolith.
- **`husky_physx`** — Listed in `skip_robots`. PhysX rigid-body friction only (simulates driving on hard rock at 0.7 friction coefficient).

Observable behavioral differences on soft lunar soil:
1. **Sinkage drag:** Terramechanics rover requires more motor effort to maintain speed
2. **Slope climbing:** Physx rover maintains grip on inclines where terramechanics rover slips
3. **Turning radius:** Terramechanics rover experiences lateral soil resistance during skid-steer turns
4. **Wheel digging:** At high slip, terramechanics rover sinks deeper (slip-sinkage coupling)

---

## 10. Limitations

1. **Fz handled by PhysX:** The model does not apply a computed normal force — it relies on PhysX rigid-body collision. This causes a mismatch: traction follows Bekker soft-soil physics, but wheel bounce follows rigid-body impulse. This can produce jitter on very soft soils at high slip.

2. **Classic Bekker model limits at 1/6g:** The Bekker-Janosi model is a macro-level semi-empirical model developed from Earth-gravity bevameter tests. At lunar gravity (1.625 m/s²), low contact pressures push behavior into regimes the model was not calibrated for (particle-scale slip, low-pressure granular flow).

3. **Single soil layer:** The model uses one set of soil parameters per simulation. Depth-dependent soil stratification (hard pan beneath loose surface) is not modeled.

4. **Approximate heterogeneity:** Soil variation uses value noise (hash-based), not measured DEM-derived soil property maps. Patch geometry does not correlate with surface features like craters.

---

## 11. References

- Bekker, M.G. (1969). *Introduction to Terrain-Vehicle Systems*. University of Michigan Press.
- Wong, J.Y. & Reece, A.R. (1967). Prediction of rigid wheel performance. *Journal of Terramechanics*, 4(1), 81–98.
- Janosi, Z. & Hanamoto, B. (1961). Analytical determination of drawbar pull as a function of slip. *1st ISTVS Conf.*
- Lyasko, M. (2010). Slip sinkage effect in soil–vehicle mechanics. *Journal of Terramechanics*, 47(1), 21–31.
- Oravec, H.A. et al. (2010). Design and characterization of GRC-1: A soil for lunar terramechanics testing. *Journal of Terramechanics*, 47(6), 361–377.
