# 테라메카닉스: 바퀴-지형 상호작용 모델

## 1. 개요

이 프로젝트는 Isaac Sim의 PhysX 강체 물리 엔진 위에 **Bekker-Janosi 테라메카닉스 모델**을 구현합니다. PhysX만으로는 재현할 수 없는, 변형 가능한 월면 레골리스에서의 현실적인 바퀴-지형 상호작용 힘을 계산합니다.

PhysX는 모든 표면을 일정한 마찰 계수를 가진 강체로 취급합니다. 달 표면의 레골리스는 바퀴 하중에 의해 변형되는 연질 토양으로, 침하(sinkage), 슬립, 트랙션 손실 등의 거동이 강체 접촉과 근본적으로 다릅니다. 테라메카닉스 레이어는 이를 보정하기 위해 추가 힘을 계산하고 매 물리 스텝마다 각 바퀴의 강체에 외력(external wrench)으로 적용합니다.

**핵심 파일:**

| 파일 | 역할 |
|------|------|
| [physics/terramechanics.py](../physics/terramechanics.py) | 핵심 솔버: Bekker-Janosi 힘/토크 계산 |
| [physics/terramechanics_parameters.py](../physics/terramechanics_parameters.py) | 로봇 및 토양 파라미터 데이터클래스 |
| [environments/lunar_yard.py](../environments/lunar_yard.py) | 통합 레이어: PhysX 상태 읽기, 솔버 호출, 힘 적용 |
| [robots/rigid_body_group.py](../robots/rigid_body_group.py) | 물리 인터페이스: 접촉력, 속도, 힘 적용 |

---

## 2. 시스템 아키텍처

```
PhysX 시뮬레이션 스텝
        │
        ▼
RobotRigidGroup.get_net_contact_forces()   ← 바퀴별 접촉 Fz
RobotRigidGroup.get_velocities()           ← 바퀴별 선속도 + 각속도
RobotRigidGroup.get_pose()                 ← 바퀴별 방향 (heading only)
        │
        ▼
LunarYardEnvironment.apply_terramechanics()
  ├── EMA 필터 (PhysX 지터 감소)
  ├── 침하량 계산 (Bekker 압력-침하 역산)
  ├── 속도 투영 (월드 프레임 → 바퀴 로컬 프레임)
  ├── 토양 이질성 샘플링 (바퀴 위치별 공간 노이즈)
  └── TerramechanicsSolver.compute_force_and_torque()
            │
            ▼  (로컬 프레임 힘/토크 반환)
        프레임 변환 (로컬 → 월드 프레임)
            │
            ▼
RobotRigidGroup.apply_force_torque()       ← PhysX 강체에 적용
        │
        ▼
PhysX가 새 상태 적분 (다음 스텝)
```

이 사이클은 매 시뮬레이션 스텝(기본값: `physics_dt = 0.0166 s`, 약 60 Hz)마다 실행됩니다.

---

## 3. 수학 모델

### 3.1 Bekker 압력-침하 방정식

바퀴가 변형 가능한 토양에 깊이 *z* 만큼 침하할 때, 접촉면 아래의 법선 응력 σ는 Bekker의 압력-침하 법칙을 따릅니다:

```
σ(z) = (k_c/b + k_phi) · z^n
```

여기서:
- `k_c` — 점착 변형 계수 (N/m^(n+1))
- `k_phi` — 마찰 변형 계수 (N/m^(n+2))
- `b` — 바퀴 폭 (m)
- `n` — 침하 지수 (무차원)
- `z` — 침하 깊이 (m)

진입각 θ_f로 토양에 진입하는 반경 *r*의 원통형 바퀴에서, 각도 θ에서의 침하량은:

```
z(θ) = r · (cos θ − cos θ_f)
```

### 3.2 Wong-Reece 응력 분포

법선 응력은 접촉호(contact arc) 전반에 걸쳐 대칭적이지 않습니다. 최대 응력은 Wong-Reece 모델로 계산되는 θ_m에서 발생합니다:

```
θ_m = (a_0 + a_1 · slip) · θ_f
```

경험적 계수는 `a_0 = 0.4`, `a_1 = 0.3`입니다. 이를 통해 비대칭 압력 분포를 모사합니다 — 구동 시 앞쪽 호에서 높고, 제동 시 뒤쪽으로 이동합니다.

- **앞쪽 호** [θ_m → θ_f]: 표준 Bekker 공식 직접 적용
- **뒤쪽 호** [θ_r → θ_m]: θ_r에서 0, θ_m에서 최대값으로 보간

### 3.3 Janosi-Hanamoto 전단 응력

토양 전단 응력 τ는 Janosi-Hanamoto 방정식으로 결정됩니다:

```
τ(θ) = (c + σ · tan φ) · (1 − exp(−j / K))
```

여기서:
- `c` — 토양 점착력 (Pa)
- `φ` — 내부 마찰각 (rad)
- `K` — 전단 변형 계수 (m) — 작을수록 트랙션이 빠르게 발현
- `j(θ)` — 전단 변위 (m)

전단 변위는 접촉호를 따라 누적됩니다:

```
j(θ) = r · [θ_f − θ − (1 − slip) · (sin θ_f − sin θ)]
```

**제동 시 보정:** 원래 Janosi-Hanamoto 공식은 음의 *j* (제동)에서 발산하여 `|τ| >> Mohr-Coulomb 한계`를 만들어냅니다. 이는 물리적으로 불가능합니다. 구현에서는 포화 인수를 [−1, +1]로 클램핑합니다:

```python
sat = np.clip(1.0 - np.exp(-j / K), -1.0, 1.0)
```

### 3.4 종방향 힘과 저항 토크

접촉호(θ_r ~ θ_f)에 대한 적분으로 다음을 구합니다:

**견인력 Fx (Drawbar Pull):**
```
Fx = r · b · ∫[θ_r → θ_f] (τ · cos θ − σ · sin θ) dθ
```

**롤링 저항 토크 My:**
```
My = r² · b · ∫[θ_r → θ_f] τ(θ) dθ
```

My는 바퀴에 음의 토크로 적용됩니다 (회전에 저항). 변형 가능한 토양에서 바퀴를 회전시키기 위해 모터가 극복해야 하는 토크를 의미합니다.

### 3.5 슬립비 (Slip Ratio)

슬립비 *s*는 차체 이동에 비해 바퀴가 얼마나 헛도는지를 나타냅니다:

```
구동 (v ≤ ω·r):   s = 1 − v / (ω · r)       ∈ [0, 1]
제동 (v > ω·r):   s = (ω · r) / v − 1         ∈ [−1, 0]
순수 슬라이드 (ω = 0):  s = −sign(v)           (±1)
```

여기서 `v`는 바퀴 전진 속도 (m/s), `ω`는 바퀴 각속도 (rad/s)입니다.

---

## 4. 기존 구현 대비 개선 사항

이 구현은 OmniLRS 레퍼런스 구현(`src/physics/terramechanics_solver.py`)을 크게 확장합니다:

| 기능 | 기존 구현 | 본 구현 |
|------|-----------|---------|
| 중력 | 하드코딩 9.81 m/s² | 설정 가능 (기본값: 달 1.625 m/s²) |
| 토양 기본값 | 지구 토양 | 아폴로/GRC-1 월면 레골리스 |
| 수치 적분 | `scipy.integrate.quad` (바퀴별 Python 루프) | 12점 Gauss-Legendre 구적법, 완전 벡터화 |
| 슬립-침하 결합 | 미구현 | Lyasko (2010) 모델 |
| 그라우저 보정 | 미구현 | 유효 반경, 전단 폭, 흙 포획 비율 |
| 횡방향력 Fy | 미구현 (0으로 고정) | Mohr-Coulomb + 슬립각 포화 |
| 다짐 저항력 | 미구현 | Bekker 불도저 드래그 공식 |
| PhysX 트랙션 보정 | 미구현 | 트랙션 결핍 보정 (physx_mu vs tan φ) |
| 마찰 한계 | 미구현 | Fy를 μ·Fz로 제한 |
| 접촉력 필터링 | 미구현 | 지수 이동 평균 (EMA) |
| 토양 이질성 | 미구현 | 바퀴 위치별 다중 옥타브 값 노이즈 |
| 로봇별 스킵 | 미구현 | 로봇 이름별 선택적 활성화/비활성화 |

---

## 5. 핵심 구현 특징

### 5.1 Gauss-Legendre 구적법 (벡터화)

접촉호 적분은 12점 Gauss-Legendre 구적법으로 계산됩니다. 모든 바퀴를 NumPy 브로드캐스팅으로 동시에 처리합니다 — 바퀴에 대한 Python for-loop가 전혀 없습니다.

```python
_GL_ORDER = 12
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_ORDER)
```

N개 바퀴에 대한 [a, b] 구간 적분:
```python
# f_vals: (N, 12) — 각 바퀴의 구적점에서의 피적분 함수 값
half_width = (b - a) / 2.0   # (N,)
result = half_width * np.einsum("...q,q->...", f_vals, _GL_WEIGHTS)
```

### 5.2 그라우저(Lug) 보정

그라우저는 로버 바퀴에서 토양 속으로 파고드는 금속 돌기입니다. 모델은 세 가지 그라우저 효과를 반영합니다:

**1. 유효 반경:**
```
r_eff = r_base + h_lug
```
그라우저 끝단이 외부 접촉원을 정의합니다.

**2. 유효 전단 폭:**
```
b_shear = b + 2 · h_lug
```
각 그라우저의 양쪽 측벽을 따라 토양이 전단되어 유효 트랙션 면적이 증가합니다.

**3. 흙 포획 비율:**
```
gap = 2π · r_base / n_lug           ← 그라우저 간 호 길이
trap_ratio = min(h_lug / gap, 1.0)
K_grouser = K · (1 − 0.5 · trap_ratio)   ← K 최대 50% 감소
```
그라우저 사이에 포획된 흙은 전단 변형 계수 *K*를 감소시켜, 더 적은 슬립으로 더 빠르게 트랙션이 발현됩니다.

### 5.3 슬립-침하 결합 (Lyasko 2010)

헛도는 바퀴가 토양을 굴착하기 때문에 슬립이 증가하면 침하량도 증가합니다:

```
z_eff = z · (1 + slip_sinkage_coeff · |slip|)
```

수치 불안정성을 방지하기 위해 바퀴 반경의 99%로 클램핑됩니다. `slip_sinkage_coeff = 2.0`인 느슨한 월면 레골리스에서 50% 슬립 상태의 바퀴는 정적 침하량의 두 배까지 침하됩니다.

### 5.4 횡방향력 — Mohr-Coulomb 전단 (Fy)

바퀴가 옆으로 미끄러질 때, 토양은 Mohr-Coulomb 전단으로 저항합니다:

```
τ_lat_capacity = r · (c · arc_length + tan φ · ∫σ dθ)
```

힘의 포화는 원시 횡속도가 아닌 슬립각으로 계산합니다 (PhysX 진동으로 인한 힘 스파이크 방지):

```
α = atan2(|v_lat|, |v_fwd|)                    ← 슬립각
sat = 1 − exp(−α / α_sat),  α_sat = 0.15 rad   ← 특성각 약 8.6°
Fy = −sign(v_lat) · b_shear · τ_lat_capacity · sat
```

PhysX 수치 노이즈로 인한 작은 횡속도를 무시하기 위해 0.05 m/s 데드존을 적용합니다.

또한 Fy는 마찰 한계로 제한됩니다:
```
|Fy| ≤ tan(φ) · Fz
```

### 5.5 다짐 저항력 (Compaction Resistance, Rc)

바퀴가 전진할 때 앞쪽 흙을 밀어내야 합니다 (불도저 효과). 이 힘은 전진을 방해하며 모터 컨트롤러가 보상할 수 없습니다:

```
Rc = b · z² · (k_c/b + k_phi) / (n + 1)
```

적용 방식: `force_x += −sign(v) · Rc`

이는 Bekker의 다짐 저항 공식입니다. 침하량의 제곱에 비례하므로, 깊은 침하가 발생하는 매우 연한 레골리스에서 상당한 저항력이 됩니다.

### 5.6 트랙션 결핍 보정 (Traction Deficit Correction)

PhysX는 지형을 `physx_mu` (고무-암석의 경우 일반적으로 0.7)의 마찰 계수를 가진 강체 표면으로 취급합니다. 연한 월면 레골리스가 제공할 수 있는 트랙션은 이보다 훨씬 작아 약 `tan(φ) × Fz`입니다.

고슬립 상태에서 PhysX 트랙션과 실제 토양 한계 사이의 차이가 비현실적으로 높은 추진력을 만들어냅니다. 이를 보정하는 역방향 힘을 적용합니다:

```
soil_mu = tan(φ)
mu_gap = max(physx_mu − soil_mu, 0)

slip_factor = clip(|slip| · 3.0, 0, 1)    ← 슬립 범위 [0, 0.33]에서 선형 증가
drive_dir = sign(ω·r − v)                  ← +1 구동, −1 제동

traction_deficit = −mu_gap · Fz · drive_dir · slip_factor
```

고슬립에서 실제 토양이 감당할 수 있는 수준으로 순 추진력을 감소시킵니다. 저슬립에서는 두 모델이 일치하므로 보정이 필요 없습니다.

**예시:** `physx_mu = 0.7`, `phi = 20°` → `tan(phi) = 0.364`:
- `mu_gap = 0.7 − 0.364 = 0.336`
- 완전 슬립 시, 바퀴당 결핍 힘 ≈ −0.336 × Fz

### 5.7 접촉력으로부터 침하량 계산

PhysX는 지형을 강체로 취급하므로 바퀴가 기하학적으로 침투하지 않습니다. 침하량은 측정된 PhysX 접촉력으로부터 Bekker 압력-침하 방정식을 역산하여 추정합니다:

```
Fz ≈ 2·b·(k_c/b + k_phi)·√(2r) · z^(n + 0.5)
z = (Fz / denom)^(1/(n + 0.5))
```

접촉력은 PhysX 충돌/바운스 스파이크를 거부하면서 경사면에서의 하중 이동을 허용하기 위해 `3 × 정적 바퀴 하중`으로 클램핑됩니다. 침하량은 `0.5 × 바퀴 반경`으로 제한됩니다.

### 5.8 접촉력에 대한 EMA 필터

PhysX 접촉력은 본질적으로 노이즈가 많습니다. 지수 이동 평균(EMA)으로 평활화합니다:

```
EMA(t) = α · EMA(t−1) + (1 − α) · raw(t)
```

- `α = 0` — 필터링 없음 (PhysX 원시값)
- `α = 0.7` (기본값) — 적당한 평활화, 지연 약 3 프레임

처음 착지 시 충격 스파이크를 방지하기 위해 0으로 초기화됩니다.

다중 로봇 시나리오에서 독립적인 필터링을 위해 로봇 이름별로 별도의 EMA 상태(`_contact_ema` 딕셔너리)를 유지합니다.

### 5.9 토양 이질성 (Soil Heterogeneity)

공간적으로 변하는 토양 특성이 느슨한 레골리스와 다져진 레골리스의 패치를 시뮬레이션합니다. 각 바퀴 위치에서 다중 옥타브 값 노이즈가 `soil_multiplier`를 생성합니다:

```
multiplier > 1.0 → 더 느슨한 토양: K 증가, c와 tan(φ) 감소
multiplier < 1.0 → 더 다져진 토양: K 감소, c와 tan(φ) 증가
```

세 개의 옥타브(1/scale, 2/scale, 4/scale 주파수)를 진폭 1, 0.5, 0.25로 합산하여 미세한 디테일이 있는 자연스러운 넓은 패치를 생성합니다. 해시 함수는 결정론적이므로 같은 월드 위치는 항상 같은 토양 타입을 생성합니다 (시간적 깜빡임 없음).

설정:
```yaml
heterogeneity: 0.5       # 변동 강도 (0 = 균일)
heterogeneity_scale: 1.5 # 패치 공간 스케일 (미터)
heterogeneity_seed: 42   # 재현성 시드
```

### 5.10 로봇별 스킵 리스트

특정 로봇을 테라메카닉스에서 제외하여 PhysX만의 강체 거동을 유지할 수 있습니다. 이는 나란히 비교하는 실험에 유용합니다:

```yaml
terramechanics_settings:
  skip_robots: ["husky_physx"]
```

`skip_robots`에 없는 모든 로봇은 Bekker-Janosi 힘을 받습니다. `skip_robots`에 있는 로봇은 PhysX 마찰만 사용합니다.

---

## 6. 힘 파이프라인 요약

매 시뮬레이션 스텝마다 다음 힘과 토크가 조합되어 바퀴 강체에 적용됩니다:

| 구성요소 | 방향 | 설명 |
|---------|------|------|
| `Rc` (다짐 저항력) | −전진 방향 | 토양에 침하하며 발생하는 불도저 드래그 |
| `traction_deficit` (트랙션 결핍) | −전진 방향 | 연한 토양에서 PhysX 과잉 트랙션 보정 |
| `Fy` (횡방향 전단력) | 횡방향 | 횡 슬립에 대한 Mohr-Coulomb 저항 |
| `My` (롤링 저항 토크) | 차축 주위 | 변형 가능한 토양에서 바퀴 회전 비용 |

참고: **Fz는 적용하지 않습니다** — PhysX 강체 충돌이 법선 반력을 처리합니다. 테라메카닉스 레이어는 그 위에 변형 가능한 토양 보정만 추가합니다.

힘은 바퀴 로컬 프레임(Fx: 전진, Fy: 횡방향, My: 차축)으로 계산된 후, 적용 전에 월드 프레임으로 변환됩니다:

```python
world_forces  = Fx · forward_dir + Fy · lateral_dir
world_torques = My · axle_dir
```

NaN/Inf로 PhysX가 충돌하는 것을 방지하기 위해 바퀴별 안전 클램프를 적용합니다:
```
max_force = max(5 × 정적 바퀴 하중, 50 N)
```

---

## 7. 파라미터 레퍼런스

### 7.1 토양 파라미터 (`TerrainMechanicalParameter`)

| 파라미터 | 기본값 | 단위 | 설명 |
|---------|--------|------|------|
| `k_c` | 1400 | N/m^(n+1) | 점착 변형 계수 |
| `k_phi` | 820000 | N/m^(n+2) | 마찰 변형 계수 |
| `n` | 1.0 | — | 침하 지수 |
| `c` | 170 | Pa | 토양 점착력 |
| `phi` | 35° | rad | 내부 마찰각 |
| `K` | 0.018 | m | 전단 변형 계수 |
| `rho` | 1500 | kg/m³ | 토양 겉보기 밀도 |
| `gravity` | 1.625 | m/s² | 표면 중력 가속도 |
| `a_0` | 0.4 | — | Wong-Reece θ_m 계수 |
| `a_1` | 0.3 | — | Wong-Reece 슬립 계수 |
| `slip_sinkage_coeff` | 0.5 | — | Lyasko 슬립-침하 결합 계수 |
| `heterogeneity` | 0.0 | — | 토양 변동 강도 [0–1] |
| `heterogeneity_scale` | 2.0 | m | 토양 패치 공간 스케일 |
| `heterogeneity_seed` | 42 | — | 재현성 시드 |
| `physx_mu` | 0.0 | — | PhysX 마찰 계수 (0 = 보정 없음) |

기본값은 아폴로 탐사 측정값과 GRC-1 시뮬런트 데이터(Oravec et al. 2010)를 기반으로 한 깊이 약 5–15 cm의 중간 밀도 월면 레골리스를 나타냅니다.

### 7.2 로봇 파라미터 (`RobotParameter`)

| 파라미터 | 기본값 | 단위 | 설명 |
|---------|--------|------|------|
| `mass` | 20.0 | kg | 로버 총 질량 |
| `num_wheels` | 4 | — | 구동 바퀴 수 |
| `wheel_radius` | 0.09 | m | 바퀴 외경 반지름 |
| `wheel_width` | 0.1 | m | 접촉 패치 폭 (Bekker 방정식의 b) |
| `wheel_lug_height` | 0.02 | m | 그라우저 돌출 높이 |
| `wheel_lug_count` | 16 | — | 바퀴당 그라우저 수 |

---

## 8. YAML 설정

```yaml
terramechanics_settings:
  enable: true
  contact_force_ema: 0.7          # EMA 평활화 (0 = 원시값, 0.7 = 평활)
  skip_robots: ["husky_physx"]    # PhysX 전용 로봇 (비교 기준)

  robot:
    mass: 50.0                    # Husky 로버 질량 (kg)
    num_wheels: 4
    wheel_radius: 0.165
    wheel_width: 0.1
    wheel_lug_height: 0.01
    wheel_lug_count: 16

  soil:
    # 느슨한 월면 레골리스 (슬립이 눈에 보이도록 튜닝된 playground 설정)
    k_c: 200.0
    k_phi: 20000.0
    n: 0.7
    c: 10.0
    phi_deg: 20.0                 # 내부에서 라디안으로 변환
    K: 0.025
    rho: 1300.0
    gravity: 1.625
    physx_mu: 0.7                 # 트랙션 결핍 보정 활성화
    slip_sinkage_coeff: 2.0
    heterogeneity: 0.5
    heterogeneity_scale: 1.5
    heterogeneity_seed: 42
```

---

## 9. 이중 로버 비교 설정

`lunar_yard_20m_playground.yaml` 설정은 두 대의 Husky 로버를 나란히 배치합니다:

- **`husky`** — Bekker-Janosi 테라메카닉스 활성화. 연한 레골리스에서 침하, 다짐 드래그, 트랙션 결핍을 경험합니다.
- **`husky_physx`** — `skip_robots`에 등록. PhysX 강체 마찰만 사용 (암석 위를 0.7 마찰 계수로 주행하는 것과 동일).

연한 월면 토양에서 관찰 가능한 거동 차이:
1. **침하 드래그:** 테라메카닉스 로버는 속도 유지에 더 많은 모터 출력이 필요
2. **경사 등반:** PhysX 로버는 테라메카닉스 로버가 미끄러지는 경사에서도 그립 유지
3. **선회 반경:** 테라메카닉스 로버는 스키드-스티어 선회 시 횡방향 토양 저항 경험
4. **바퀴 파묻힘:** 고슬립에서 테라메카닉스 로버는 슬립-침하 결합으로 더 깊이 침하

---

## 10. 한계점

1. **Fz를 PhysX에 의존:** 모델은 계산된 법선력을 직접 적용하지 않고 PhysX 강체 충돌에 의존합니다. 트랙션은 Bekker 연질 토양 물리를 따르지만, 바퀴의 바운스는 강체 임펄스 방식을 따르는 불일치가 발생합니다. 고슬립의 매우 연한 토양에서 지터가 발생할 수 있습니다.

2. **1/6g에서 클래식 Bekker 모델의 한계:** Bekker-Janosi 모델은 지구 중력(1g) 베바미터 테스트에서 개발된 매크로 수준의 반경험적 모델입니다. 달의 중력(1.625 m/s²)에서는 낮은 접촉 압력이 모델이 보정되지 않은 영역으로 거동을 밀어냅니다 (입자 규모의 슬립, 저압 과립 흐름).

3. **단일 토양 레이어:** 시뮬레이션당 하나의 토양 파라미터 세트를 사용합니다. 느슨한 표면 아래의 단단한 층과 같은 깊이 의존적 토양 층상화는 모델링되지 않습니다.

4. **근사적 이질성:** 토양 변동은 측정된 DEM 기반 토양 특성 지도가 아닌 값 노이즈(해시 기반)를 사용합니다. 패치 형상은 크레이터와 같은 표면 지형과 상관관계가 없습니다.

---

## 11. 참고 문헌

- Bekker, M.G. (1969). *Introduction to Terrain-Vehicle Systems*. University of Michigan Press.
- Wong, J.Y. & Reece, A.R. (1967). Prediction of rigid wheel performance. *Journal of Terramechanics*, 4(1), 81–98.
- Janosi, Z. & Hanamoto, B. (1961). Analytical determination of drawbar pull as a function of slip. *1st ISTVS Conf.*
- Lyasko, M. (2010). Slip sinkage effect in soil–vehicle mechanics. *Journal of Terramechanics*, 47(1), 21–31.
- Oravec, H.A. et al. (2010). Design and characterization of GRC-1: A soil for lunar terramechanics testing. *Journal of Terramechanics*, 47(6), 361–377.
