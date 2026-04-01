# Realistic Crater Generator Design

## Summary

기존 `CraterGenerator`의 단순 원형 + 균일 림 크레이터에 더해, 자연스러운 찌그러짐과 울퉁불퉁한 림을 가진 현실적 크레이터를 생성하는 `RealisticCraterGenerator`를 추가한다. 기존 코드는 수정하지 않으며, 서브클래스로 확장한다.

## Decisions

| 결정 사항 | 선택 | 근거 |
|-----------|------|------|
| 현실감 수준 | 중간 (Moderate Realism) | 현재 스케일(40m, 반경 0.25~2.5m)에서 central peak 등은 불필요 |
| 모드 전환 | YAML config `crater_mode` | 단순하고 기존 파이프라인 변경 최소 |
| 외곽선 변형 | Low-freq harmonics + High-freq Perlin noise | 전체 형태와 미세 질감을 독립 조절 가능 |
| 벽면/바닥 | 림 높이 변조 → 프로파일 스케일링 + radial slump noise | 기존 스플라인 프로파일을 그대로 활용 |
| 구현 방식 | `RealisticCraterGenerator` 서브클래스 | 기존 코드 수정 없음, 테스트 독립 |
| 암석 배치 | 기존 원형 방식 유지 | 스코프 최소화, 현재 스케일에서 괴리 미미 |

## Architecture

### Class Hierarchy

```
CraterGenerator (기존 — 수정 없음)
├── __init__(cfg: CraterGeneratorConf)
├── _load_profiles()
├── _centered_distance_matrix(cd) ← override target
├── _apply_profile(dist, cd)      ← override target
├── randomize_parameters(index, size)
├── generate_single(size, index, crater_data)
└── generate_craters(dem, coords, radii, craters_data)

RealisticCraterGenerator(CraterGenerator) ← NEW
├── __init__(cfg: CraterGeneratorConf)  # super().__init__ + realistic params
├── _centered_distance_matrix(cd)        # OVERRIDE
├── _apply_profile(dist, cd)             # OVERRIDE
└── randomize_parameters(index, size)    # OVERRIDE (확장 필드 추가)
```

### File Structure

```
terrain/
├── config.py                          # + RealisticCraterConf 추가
├── procedural/
│   ├── crater_generator.py            # 기존 — 수정 없음
│   ├── realistic_crater_generator.py  # NEW
│   └── moonyard_generator.py          # crater_mode 분기 추가
```

### Config Changes

`CraterGeneratorConf`에 `crater_mode` 필드 추가:

```python
@dataclasses.dataclass
class CraterGeneratorConf:
    # ... 기존 필드 그대로 ...
    crater_mode: str = "classic"  # "classic" | "realistic"
```

새로운 `RealisticCraterConf` dataclass:

```python
@dataclasses.dataclass
class RealisticCraterConf:
    """Parameters for realistic crater shape deformation."""
    # 외곽선 찌그러짐
    n_harmonics: int = 4              # 조화함수 차수 (2~6)
    harmonic_amp: float = 0.12        # 전체 찌그러짐 강도 (반경의 ±12%)
    contour_noise_amp: float = 0.03   # 고주파 Perlin 노이즈 진폭 (±3%)

    # 림 높이 변조
    rim_n_harmonics: int = 3          # 림 높이 변화 주파수 성분 수
    rim_noise_amp: float = 0.15       # 림 높이 변동 (±15%)

    # 벽면 슬럼프
    slump_intensity: float = 0.1      # 슬럼프 강도 (크레이터 깊이의 ±10%)
    slump_wall_range: Tuple[float, float] = (0.3, 0.8)  # 슬럼프 적용 반경 범위

    # 바닥 불규칙성
    floor_noise_amp: float = 0.03     # 바닥 2D noise 진폭
    floor_radius_ratio: float = 0.3   # 바닥 영역 범위 (반경 비율)
```

`MoonYardConf`에 `RealisticCraterConf` 포함:

```python
@dataclasses.dataclass
class MoonYardConf:
    # ... 기존 필드 ...
    realistic_crater: RealisticCraterConf = dataclasses.field(
        default_factory=RealisticCraterConf
    )
```

### YAML Config Example

```yaml
terrain_manager:
  moon_yard:
    crater_generator:
      crater_mode: "realistic"  # "classic" for original behavior
      profiles_path: assets/Terrains/crater_spline_profiles.pkl
      z_scale: 0.2
      seed: 42
    realistic_crater:
      n_harmonics: 4
      harmonic_amp: 0.12
      contour_noise_amp: 0.03
      rim_n_harmonics: 3
      rim_noise_amp: 0.15
      slump_intensity: 0.1
      floor_noise_amp: 0.03
```

## Algorithm Detail

### 1. _centered_distance_matrix() Override

기존 타원 변형 + 마크를 대체하여 harmonic perturbation + Perlin noise로 불규칙한 윤곽을 생성한다.

**Step 1 — Harmonic contour 생성:**

```python
# 방위각 θ에 대한 반경 변조 함수
contour(θ) = 1.0
for n in range(2, 2 + n_harmonics):
    a_n = rng.uniform(0, harmonic_amp / n)  # 고차 harmonics은 진폭 감소
    phi_n = rng.uniform(0, 2π)
    contour += a_n * cos(n * θ + phi_n)
```

- `n=2`: 타원형 변형 (가장 지배적)
- `n=3`: 삼각형 변형
- `n=4,5`: 세부 불규칙성
- 진폭이 `1/n`으로 감쇠하여 자연스러운 형태 유지

**Step 2 — 고주파 Perlin noise 추가:**

```python
# 1D Perlin noise를 θ에 매핑 (주기적 경계 조건)
noise(θ) = perlin_1d(θ * freq, octaves=2) * contour_noise_amp
contour(θ) += noise(θ)
```

- 크레이터마다 noise seed를 달리하여 개별 형태 보장
- 주기적 경계 조건으로 θ=0과 θ=2π에서 매끄럽게 연결

**Step 3 — 거리 행렬 변조:**

```python
# 기존 타원 변형은 유지 (xy_deformation_factor)
dist = sqrt(((x - cx) / sx)² + ((y - cy) / sy)²)

# contour 함수로 방위각별 반경 변조
dist = dist / contour(θ_grid)

# 기존 회전 적용
dist = rotate(dist, rotation, reshape=False, cval=n/2)

# 클램핑
dist[dist > n/2] = n/2
```

### 2. _apply_profile() Override

기존의 단일 프로파일 적용을 확장하여 방위각별 림 높이 변조, 벽면 슬럼프, 바닥 노이즈를 합성한다.

> **데이터 흐름:** `randomize_parameters()`가 `RealisticCraterData`에 모든 harmonic/noise 파라미터를 저장하고, `generate_single()`이 `cd` 인자로 전달한다. `_apply_profile(cd)`는 `cd.rim_amplitudes`, `cd.rim_phases`, `cd.slump_noise_seed`, `cd.floor_noise_seed`를 읽어 사용한다. `_centered_distance_matrix(cd)`도 마찬가지로 `cd.harmonic_amplitudes`, `cd.harmonic_phases`, `cd.contour_noise_seed`를 사용한다.

**Layer 1 — 림 높이 변조:**

```python
# 방위각별 림 높이 스케일 함수
rim_scale(θ) = 1.0
for n in range(2, 2 + rim_n_harmonics):
    b_n = rng.uniform(0, rim_noise_amp / n)
    psi_n = rng.uniform(0, 2π)
    rim_scale += b_n * cos(n * θ + psi_n)

# 기존 프로파일에 림 높이 스케일 적용
crater = base_profile(dist) * rim_scale(θ_grid) * size/2 * z_scale * resolution
```

- `rim_scale > 1` 방향: 림이 높고 벽이 가파름
- `rim_scale < 1` 방향: 림이 낮고 벽이 완만함
- 비대칭적인 크레이터 단면을 자연스럽게 생성

**Layer 2 — 벽면 슬럼프:**

```python
# 벽면 영역 마스크 (smooth transition)
r_norm = 2 * dist / size  # 0~1 정규화
wall_mask = smooth_step(r_norm, slump_wall_range[0], slump_wall_range[1])

# 방위각-반경 평면의 radial noise
slump = noise_1d(r_norm * freq) * slump_intensity * size/2 * z_scale * resolution

# 벽면에만 적용
crater += slump * wall_mask
```

- smooth_step mask로 벽면 영역에서만 부드럽게 적용
- 계단식 붕괴 효과: 벽면의 특정 반경에서 급격한 고도 변화

**Layer 3 — 바닥 불규칙성:**

```python
# 바닥 영역 마스크
floor_mask = smooth_step(1 - r_norm / floor_radius_ratio, 0, 1)

# 2D Perlin noise
floor_noise = perlin_2d(x_grid, y_grid, freq=4) * floor_noise_amp * size/2 * z_scale * resolution

# 바닥에만 적용
crater += floor_noise * floor_mask
```

- 바닥(r < 0.3R)에서만 활성화, 가장자리에서 0으로 페이드
- 미세한 언덕/요철로 단조로운 평탄 바닥을 개선

### 3. randomize_parameters() Override

기존 `CraterData`를 확장한 `RealisticCraterData`를 반환한다.

```python
@dataclasses.dataclass
class RealisticCraterData(CraterData):
    """Extended crater metadata for realistic generation."""
    # Harmonic contour parameters
    harmonic_amplitudes: np.ndarray = None   # shape: (n_harmonics,)
    harmonic_phases: np.ndarray = None       # shape: (n_harmonics,)
    contour_noise_seed: int = 0

    # Rim height modulation
    rim_amplitudes: np.ndarray = None        # shape: (rim_n_harmonics,)
    rim_phases: np.ndarray = None            # shape: (rim_n_harmonics,)

    # Slump parameters
    slump_noise_seed: int = 0

    # Floor noise
    floor_noise_seed: int = 0
```

- `CraterData`를 상속하므로 기존 rock distribution과 호환
- `generate_craters()`에서 `CraterData` 리스트에 그대로 추가 가능
- noise seed를 저장하여 동일한 크레이터를 재현 가능

## Integration

### MoonyardGenerator 변경

```python
class MoonyardGenerator:
    def __init__(self, cfg: MoonYardConf, seed=None):
        # ...
        if cfg.crater_generator.crater_mode == "realistic":
            self._crater_gen = RealisticCraterGenerator(
                cfg.crater_generator, cfg.realistic_crater
            )
        else:
            self._crater_gen = CraterGenerator(cfg.crater_generator)
```

- `randomize()`, `augment()` 등은 변경 없음 — 둘 다 `self._crater_gen.generate_craters()` 호출
- `RealisticCraterGenerator`가 `CraterGenerator`의 서브클래스이므로 인터페이스 동일

### Perlin Noise 구현

1D/2D Perlin noise가 필요. 선택지:
- `noise` 패키지 (pnoise1, pnoise2) — 가장 간단하지만 외부 의존성
- NumPy 기반 자체 구현 — 의존성 없음, 코드량 약간 증가

**결정:** NumPy 기반 자체 구현. 이유:
- 이미 프로젝트가 NumPy에만 의존하는 구조
- 필요한 noise는 단순 1D/2D이므로 구현 부담 적음
- `terrain/procedural/noise.py`에 유틸리티로 배치

## Testing Strategy

### Unit Tests (`tests/test_realistic_crater.py`)

1. **형상 검증**: RealisticCraterGenerator가 원형이 아닌 불규칙한 윤곽을 생성하는지
   - 거리 행렬에서 동일 반경 원 위의 값들이 일정하지 않은지 확인
   - 표준편차가 임계값 이상인지 검증

2. **림 높이 변동**: 림 높이가 방위각에 따라 변화하는지
   - 크레이터 가장자리 링에서 높이값 추출 → 표준편차 검증

3. **슬럼프 존재**: 벽면 영역에 noise가 적용되는지
   - classic 모드 결과와 비교, 벽면 영역에서 차이가 있는지

4. **바닥 불규칙성**: 바닥 영역이 평탄하지 않은지
   - 바닥 영역 높이값의 표준편차 검증

5. **재현성**: 동일 seed로 동일 결과
   - `RealisticCraterData`의 noise seed를 통한 재생성 검증

6. **하위 호환**: `CraterGenerator`와 동일 인터페이스
   - `generate_craters()`가 동일 시그니처로 동작하는지
   - 반환값이 `(DEM, mask, List[CraterData])` 형태인지

7. **Config 전환**: `crater_mode: "classic"`일 때 기존 동작과 동일한지

### Integration Tests

- `MoonyardGenerator`에 realistic 모드 config 전달 → `randomize()` 정상 동작
- 생성된 DEM이 유효한 범위 내인지 (NaN 없음, 합리적 elevation)

## Scope Boundaries

**포함:**
- `RealisticCraterGenerator` 클래스
- `RealisticCraterConf`, `RealisticCraterData` dataclass
- NumPy 기반 Perlin noise 유틸리티
- `MoonyardGenerator` 모드 분기
- Config/YAML 확장
- Unit/Integration 테스트

**제외:**
- 기존 `CraterGenerator` 코드 수정
- 암석 배치 로직 변경 (`rock_distribution.py`)
- Central peak, ejecta blanket 등 고수준 현실감 기능
- 크레이터 나이/침식 시뮬레이션
