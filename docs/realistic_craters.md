# 현실적 크레이터 생성 시스템

## 개요

기존 `CraterGenerator`의 단일 스플라인 프로파일 기반 원형 크레이터를 확장하여, 실제 충돌 크레이터에 가까운 불규칙한 외곽선과 비균일한 림 shoulder 높이를 생성합니다. 기존 29개 프로파일과 타원 변형, 회전, 표면 마크 등의 기능은 100% 유지하면서, 두 가지 추가 변형을 적용합니다:

1. **외곽선 불규칙화**: 저주파 harmonic으로 크레이터 윤곽을 찌그러뜨림
2. **Shoulder 높이 변조**: 벽면-림 경계(shoulder)에 방위각별 양수 노이즈를 additive로 추가

## 아키텍처

```
CraterGenerator (기존)
  └── RealisticCraterGenerator (확장)
        ├── _centered_distance_matrix()  ← 외곽선 harmonic 변형
        └── _apply_profile()             ← shoulder 높이 변조
```

- **설정**: `terrain/config.py` (`CraterGeneratorConf`, `RealisticCraterConf`)
- **구현**: `terrain/procedural/realistic_crater_generator.py`
- **노이즈 유틸**: `terrain/procedural/noise.py` (`perlin_1d`)
- **모드 선택**: `terrain/procedural/crater_generator.py`의 `crater_mode` 분기

---

## 1. 외곽선 불규칙화 (Contour Perturbation)

### 작동 원리

기존 `CraterGenerator`의 거리 행렬에 방위각 기반 harmonic perturbation을 적용하여, 크레이터 외곽이 완벽한 원/타원이 아닌 불규칙한 형태가 되도록 합니다.

```
dist_original = sqrt(dx² + dy²)
contour(θ) = 1 + Σ aₙ cos(nθ + φₙ)    (n = 2, 3, ..., 2+n_harmonics)
dist_perturbed = dist_original / contour(θ)
```

- `n=2`: 타원 → 약간 비대칭한 난형
- `n=3`: 삼각형 성분
- `n=4,5`: 더 세밀한 윤곽 변화

각 harmonic의 진폭은 `harmonic_amp / n`을 상한으로 랜덤 생성되어, 고차 성분일수록 진폭이 자연스럽게 감소합니다.

### 기존 방식과의 차이

| | 기존 (`classic`) | 현실적 (`realistic`) |
|---|---|---|
| 외곽선 형태 | 타원 (xy_ratio) + 9점 스플라인 변형 | 타원 + harmonic contour perturbation |
| 변형 주파수 | 매우 저주파 (9개 제어점) | 2~5차 harmonic (중주파) |
| 변형 진폭 | 0.95~1.0 범위 (최대 5%) | `harmonic_amp` 파라미터로 제어 |

---

## 2. Shoulder 높이 변조

### Shoulder란?

프로파일의 **변곡점** — 급경사 벽면이 완만한 림으로 전환되는 지점입니다.

```
프로파일 단면:

    rim crest ──╮
                │  ← shoulder (기울기 변곡점)
                │
                ╰──── crater wall (급경사)
                        │
                        ╰── floor (평탄)
```

기존 방식에서는 이 shoulder의 높이가 360° 전 방위각에서 동일하여 인공적으로 보입니다. 현실적 모드에서는 방위각마다 shoulder 높이를 다르게 하여, 어떤 방향의 림은 높고 어떤 방향은 낮은 자연스러운 형태를 만듭니다.

### 작동 원리

#### Step 1: Shoulder 위치 자동 탐지

프로파일의 기울기를 분석하여 slope-break 지점을 찾습니다:

```
1. 프로파일을 200점으로 샘플링
2. |gradient| 계산 → 최대 기울기 지점(peak) 찾기
3. peak 이후 기울기가 peak의 30%로 떨어지는 지점 = shoulder_r
```

#### Step 2: 방위각별 노이즈 생성

두 레이어의 합성으로 자연스러운 변화를 만듭니다:

1. **저주파 Harmonics** (전체적 형태): `Σ aₙ cos(nθ + φₙ)`, n = 2~4
2. **fBm Perlin** (자연스러운 디테일): 3 옥타브, 주파수 2→4→8, 진폭 ½씩 감쇠

최종 노이즈는 `abs()`로 **양수만** 취합니다. 크레이터 shoulder에 물질이 쌓이는 물리적 현상을 모사합니다.

#### Step 3: 비대칭 Gaussian 블렌딩

shoulder 지점에서 최대, 양쪽으로 Gaussian 감쇠하되 inner/outer 폭이 다릅니다:

```
blend(r) = exp(-0.5 × ((r - shoulder_r) / σ)²)

σ = falloff_inner  (r ≤ shoulder_r, 크레이터 안쪽 방향)
σ = falloff_outer  (r > shoulder_r, 림 바깥 방향)
```

```
최종 높이 = base_profile(r) + noise(θ) × blend(r)
```

- 크레이터 내부 (`r << shoulder_r`): blend ≈ 0 → 기존 프로파일 그대로
- shoulder (`r = shoulder_r`): blend = 1 → 노이즈 100% 적용
- 림 바깥 (`r >> shoulder_r`): blend ≈ 0 → 기존 프로파일 그대로

### 핵심 설계 결정

| 결정 | 이유 |
|---|---|
| 양수 노이즈만 (`abs`) | 음수 노이즈는 shoulder를 파내어 프로파일 연속성이 깨짐. 물질 퇴적이 물리적으로 더 자연스러움 |
| Additive (높이 더하기) | 프로파일 자체를 변형하지 않으므로 기존 29개 프로파일의 형상이 보존됨 |
| 비대칭 Gaussian | 안쪽은 좁게(벽면 프로파일 보존), 바깥쪽은 약간 넓게(림과 자연스럽게 합류) |
| 방위각만 의존하는 노이즈 | 반경 방향 구조가 없어 방사형 골(groove) 아티팩트가 발생하지 않음 |

---

## 파라미터

### `CraterGeneratorConf`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `crater_mode` | `"classic"` | `"classic"` 또는 `"realistic"`. realistic 선택 시 `RealisticCraterGenerator` 사용 |
| `profiles_path` | `""` | 스플라인 프로파일 pkl 경로 |
| `z_scale` | `0.2` | 크레이터 깊이 스케일. **프로파일 + shoulder 노이즈 모두에 적용됨** |
| `min_xy_ratio` | `0.85` | 타원 변형 최소 비율 |
| `max_xy_ratio` | `1.0` | 타원 변형 최대 비율 |
| `seed` | `42` | 난수 시드 |

### `RealisticCraterConf`

#### 외곽선 불규칙화

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `n_harmonics` | `4` | 외곽선 harmonic 차수 (2차부터 시작, 4이면 2~5차) |
| `harmonic_amp` | `0.08` | 최대 harmonic 진폭. 높을수록 외곽이 더 찌그러짐 |
| `contour_noise_amp` | `0.02` | (현재 미사용, 예약) |

#### Shoulder 높이 변조

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `rim_n_harmonics` | `3` | shoulder 노이즈의 harmonic 차수 (2차부터 시작) |
| `rim_noise_amp` | `0.04` | shoulder 노이즈 진폭. **높이 변조의 주요 제어 파라미터** |
| `slump_intensity` | `0.02` | (현재 미사용, 예약) |
| `floor_noise_amp` | `0.01` | (현재 미사용, 예약) |

---

## 튜닝 가이드

### 외곽선 찌그러짐

```
harmonic_amp: 0.02~0.04  →  미세한 불규칙 (거의 원형)
harmonic_amp: 0.08       →  눈에 띄는 불규칙 (권장)
harmonic_amp: 0.15+      →  강한 변형 (별 모양에 가까워질 수 있음)
n_harmonics: 2~3         →  부드러운 변형
n_harmonics: 5~6         →  세밀한 변형 (작은 크레이터에 적합)
```

### Shoulder 높이 변화

```
rim_noise_amp: 0.02      →  미세한 높이 차이
rim_noise_amp: 0.04~0.08 →  자연스러운 림 불균일 (권장)
rim_noise_amp: 0.15+     →  강한 높이 차이 (디버깅/시각화용)
```

### Shoulder 블렌딩 폭

현재 코드에서 하드코딩된 값 (필요 시 `RealisticCraterConf`에 파라미터 추가 가능):

| 변수 | 현재 값 | 설명 |
|---|---|---|
| `falloff_inner` | `0.01` | 크레이터 안쪽 감쇠 폭 (좁음 = 벽면 프로파일 보존) |
| `falloff_outer` | `0.03` | 림 바깥쪽 감쇠 폭 (넓음 = 림과 자연스럽게 합류) |

### z_scale과의 상호작용

`z_scale`은 `generate_single()`에서 프로파일 결과 전체에 곱해지므로, shoulder 노이즈 포함 최종 높이에 영향을 줍니다:

```
z_scale: 0.2  →  크레이터 직경의 20%가 깊이 (얕은 크레이터)
z_scale: 0.4  →  크레이터 직경의 40%가 깊이 (깊은 크레이터)
z_scale: 0.8  →  크레이터 직경의 80%가 깊이 (매우 깊은 크레이터)
```

`rim_noise_amp`와 `z_scale`을 함께 조정해야 원하는 시각적 비율을 유지할 수 있습니다.

---

## YAML 설정 예시

### 기본 realistic 모드

```yaml
terrain_manager:
  moon_yard:
    crater_generator:
      crater_mode: "realistic"
      profiles_path: assets/Terrains/crater_spline_profiles.pkl
      z_scale: 0.4
      seed: 42
    realistic_crater:
      n_harmonics: 4
      harmonic_amp: 0.08
      rim_n_harmonics: 3
      rim_noise_amp: 0.04
```

### 강한 불규칙 (시각화/디버깅)

```yaml
    realistic_crater:
      n_harmonics: 4
      harmonic_amp: 0.15
      rim_n_harmonics: 3
      rim_noise_amp: 0.15
```

### 미세한 불규칙 (원형에 가깝게)

```yaml
    realistic_crater:
      n_harmonics: 2
      harmonic_amp: 0.03
      rim_n_harmonics: 2
      rim_noise_amp: 0.02
```

### Classic 모드로 되돌리기

```yaml
    crater_generator:
      crater_mode: "classic"
      # realistic_crater 블록은 있어도 무시됨
```

---

## 파일 구조

| 파일 | 역할 |
|---|---|
| `terrain/config.py` | `CraterGeneratorConf`, `RealisticCraterConf` 데이터클래스 |
| `terrain/procedural/crater_generator.py` | 기존 `CraterGenerator` (29개 스플라인 프로파일 기반) |
| `terrain/procedural/realistic_crater_generator.py` | `RealisticCraterGenerator` (harmonic contour + shoulder 높이 변조) |
| `terrain/procedural/noise.py` | 주기적 Perlin noise 유틸리티 (`perlin_1d`) |
| `assets/Terrains/crater_spline_profiles.pkl` | 29개 half-crater 스플라인 프로파일 |

---

## 기술 참고

### 프로파일 로딩

`crater_spline_profiles.pkl`은 `CubicSpline` 객체를 pickle로 저장한 파일이지만, scipy 버전 간 호환성 문제를 방지하기 위해 `PPoly`로 재구성합니다:

```python
raw = SafeUnpickler(f).load()          # CubicSpline → 더미 객체로 역직렬화
profiles = [PPoly(p.c, p.x) for p in raw]  # 계수 + 분할점으로 PPoly 재구성
```

### Shoulder 탐지 알고리즘

프로파일의 기울기 변곡점을 자동 탐지합니다. 29개 프로파일마다 shoulder 위치가 다르므로, 하드코딩 대신 기울기 분석으로 동적으로 결정합니다:

```
shoulder_r ≈ 0.65~0.85 (프로파일에 따라 다름)
```

기울기 peak 이후 30% threshold에 도달하는 첫 지점을 shoulder로 정의합니다. 이 threshold는 실험적으로 대부분의 프로파일에서 시각적으로 올바른 지점을 찾는 것으로 확인되었습니다.

### 노이즈 구성

방위각 노이즈는 두 레이어의 합성입니다:

| 레이어 | 역할 | 주파수 | 진폭 |
|---|---|---|---|
| Harmonics (2~4차) | 전체적 형태 변화 | 저주파 | `rim_noise_amp / n` |
| fBm Perlin (3 옥타브) | 자연스러운 미세 디테일 | 중~고주파 (2, 4, 8) | `rim_noise_amp × 0.5^octave` |

두 레이어 모두 **주기적** (period = 2π)이므로 θ = -π와 θ = π에서 이음새가 없습니다.
