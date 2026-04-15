# 천체 시각 효과 시스템

## 개요

달 시뮬레이션의 야간 장면을 구성하는 두 가지 시각 효과 시스템: **별**과 **지구광 (Earthshine)**. 두 시스템 모두 태양 고도에 따라 동적으로 밝기가 조절되며, `StellarEngine`이 제공하는 천체 위치 데이터를 기반으로 매 프레임 업데이트됩니다.

## 아키텍처

```
StellarEngine (천체 위치 계산)
  ├── get_sun_alt_az()  ──→  Starfield.update(sun_alt)
  └── get_earth_alt_az() ─→  Earthshine.update(sun_alt, earth_alt, earth_az, sun_az)
```

- **설정**: `effects/config.py` (`StarfieldConf`, `EarthshineConf`)
- **구현**: `effects/starfield.py`, `effects/earthshine.py`
- **통합**: `environments/lunar_yard.py` → `build_scene()` / `update()`

---

## 1. 별 (Starfield)

### 렌더링 방식

DomeLight + HDR 텍스처 방식을 사용합니다. `UsdGeom.Points`는 RTX 모드에서 렌더링되지 않기 때문에, 절차적으로 생성한 lat-lon HDR 텍스처를 DomeLight에 매핑합니다.

### 별 카탈로그 생성

1. **공간 분포**: 단위 구 위에 균일 분포 (`uniform on sphere`)
2. **등급 분포**: 누적 분포 함수(CDF) 기반 — `log N(m) ∝ slope × m`
3. **색 온도**: 등급 기반 스펙트럼 근사 (밝은 별 = 파랑, 어두운 별 = 빨강)
4. **HDR 텍스처**: 별 flux를 lat-lon 이미지에 베이킹, 밝은 별(mag < 2)은 multi-pixel glow 적용

### 태양 고도별 밝기 변화

카메라 노출 적응을 시뮬레이션합니다:

| 태양 고도 | 별 밝기 | 상태 |
|---|---|---|
| ≥ `sun_fade_start` (5°) | 0.0 | 낮 — 별 안 보임 |
| `sun_fade_end` ~ `sun_fade_start` (0°~5°) | 선형 보간 | 황혼 |
| ≤ `sun_fade_end` (0°) | `base_brightness` | 밤 — 최대 밝기 |

달에는 대기가 없어 실제로는 태양이 떠 있어도 별이 보여야 하지만, 카메라 다이나믹 레인지 한계로 인해 태양 조명이 강한 환경에서는 별이 노출되지 않는 것이 사실적입니다.

### 파라미터 (`StarfieldConf`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `enable` | `true` | 활성화 여부 |
| `num_stars` | `4000000` | 생성할 별 수 |
| `magnitude_limit` | `7.5` | 최대 등급 (어두운 한계). 달 표면은 대기가 없어 7.5등급까지 관측 가능 |
| `magnitude_slope` | `0.8` | 등급 분포 기울기. **낮을수록 밝은 별 비율 증가** |
| `sphere_radius` | `5000.0` | 천구 반경 (m) |
| `base_brightness` | `1.0` | 최대 밝기 계수 |
| `texture_resolution` | `4096` | HDR 텍스처 폭 (높이 = 폭/2) |
| `sun_fade_start` | `5.0` | 별이 사라지기 시작하는 태양 고도 (°) |
| `sun_fade_end` | `0.0` | 별이 최대 밝기가 되는 태양 고도 (°) |
| `color_temperature_min` | `3000.0` | 가장 차가운 별 색 온도 (K, 적색) |
| `color_temperature_max` | `30000.0` | 가장 뜨거운 별 색 온도 (K, 청색) |
| `seed` | `42` | 난수 시드 (재현성 보장) |

#### `magnitude_slope` 튜닝 가이드

```
0.3~0.4  →  밝은 별 비율 높음 (화려한 밤하늘)
0.6      →  천문학적 관측 분포 (log N ∝ 0.6m)
0.8~1.0  →  어두운 별 비율 높음 (희미한 밤하늘)
```

### YAML 설정 예시

```yaml
starfield_settings:
  enable: true
  num_stars: 4000000
  magnitude_limit: 7.5
  magnitude_slope: 0.8
  texture_resolution: 4096
  base_brightness: 1.0
  sun_fade_start: 5.0
  sun_fade_end: 0.0
  seed: 42
```

---

## 2. 지구광 (Earthshine)

### 렌더링 방식

`DistantLight`를 사용하며, `SunController`와 동일한 **2단계 Xform 계층 구조**를 따릅니다:
- **Parent Xform**: 지구의 alt/az 방향으로 회전 (매 프레임 갱신)
- **Child DistantLight**: 고정 pre-rotation으로 광선 방향 정렬

`SphereLight`를 사용하지 않는 이유: SphereLight는 거리 제곱에 반비례하여 감쇠되므로, 800m 거리에서 도달하는 조도가 사실상 0이 됩니다. DistantLight는 거리 감쇠가 없어 실제 지구광의 평행광 특성에도 부합합니다.

### 물리 모델

지구광 세기는 세 가지 요인의 곱으로 결정됩니다:

```
intensity = base_intensity × sun_factor × earth_factor × phase_factor
```

#### 1. Sun gating (`sun_factor`)

태양이 수평선 위에 있으면 지구광이 불필요합니다.

| 태양 고도 | `sun_factor` |
|---|---|
| ≥ `sun_threshold` (5°) | 0.0 |
| `sun_threshold` ~ `sun_threshold - sun_fade_range` (5° ~ -5°) | 선형 보간 |
| ≤ `sun_threshold - sun_fade_range` (-5°) | 1.0 |

#### 2. Earth visibility (`earth_factor`)

지구가 수평선 위에 있어야 하며, 고도가 높을수록 밝습니다 (Lambert 코사인 법칙).

```
earth_factor = sin(max(0, earth_altitude))
```

- `earth_altitude < earth_min_altitude` (-5°): 지구광 = 0

#### 3. Earth phase (`phase_factor`)

태양과 지구의 각도 분리로 지구의 조명 비율을 계산합니다 (Lambertian 근사).

```
phase_angle = angular_separation(sun, earth)   # 구면 코사인 법칙
phase_factor = (1 - cos(phase_angle)) / 2
```

| 태양-지구 각도 | 지구 위상 | `phase_factor` |
|---|---|---|
| ~0° (같은 방향) | 삭지구 (New Earth) — 태양이 지구 뒤에 있어 어두운 면이 달을 향함 | 0.0 |
| ~90° | 반달지구 | 0.5 |
| ~180° (반대편) | 보름지구 (Full Earth) — 태양이 달 뒤에 있어 밝은 면이 달을 향함 | 1.0 |

> **참고**: 달 밤(태양 수평선 아래)에는 태양이 달 뒤쪽, 지구가 앞쪽에 위치하므로 phase angle ≈ 180°가 되어 보름지구 조건이 됩니다. 따라서 달 밤에 지구광이 가장 강합니다.

### 파라미터 (`EarthshineConf`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `enable` | `true` | 활성화 여부 |
| `base_intensity` | `3.0` | DistantLight 기본 세기 (보름지구 + 천정 기준) |
| `color` | `(0.85, 0.90, 1.0)` | 광색 — 약간 푸른 백색 (지구 반사광 특성) |
| `temperature` | `5500.0` | 색 온도 (K) |
| `angle` | `2.0` | 광원 시직경 (°). 달에서 본 지구의 실제 시직경 ~2° |
| `sun_threshold` | `5.0` | 지구광이 꺼지는 태양 고도 (°) |
| `sun_fade_range` | `10.0` | 페이드 범위 (°). 5° → -5° 구간에서 점진적 전환 |
| `earth_min_altitude` | `-5.0` | 지구가 이 고도 이하이면 지구광 = 0 |

#### `base_intensity` 튜닝 가이드

태양(DistantLight) intensity가 1750이므로:

```
1.0~2.0  →  은은한 지구광 (보름지구 조건에서도 미묘)
3.0~5.0  →  시각적으로 인지 가능한 수준 (권장)
10.0+    →  강한 지구광 (디버깅/시각화용)
```

실제 달 표면 지구광은 ~0.05~0.1 lux이며 태양광(~135,000 lux) 대비 매우 약합니다. 시뮬레이션에서는 시각적 효과를 위해 상대적으로 높은 값을 사용합니다.

### YAML 설정 예시

```yaml
earthshine_settings:
  enable: true
  base_intensity: 3.0
  color: [0.85, 0.90, 1.0]
  temperature: 5500.0
  angle: 2.0
  sun_threshold: 5.0
  sun_fade_range: 10.0
  earth_min_altitude: -5.0
```

---

## 통합 설정 예시

`config/environment/lunar_yard_40m.yaml`에 추가:

```yaml
stellar_engine_settings:
  start_date:
    year: 2024
    month: 5
    day: 1
    hour: 12
    minute: 50
  time_scale: 36000
  update_interval: 600
  ephemeris_path: assets/Ephemeris

starfield_settings:
  num_stars: 4000000
  magnitude_slope: 0.8
  texture_resolution: 4096

earthshine_settings:
  base_intensity: 3.0
  angle: 2.0
```

`stellar_engine_settings`가 없으면 별과 지구광 모두 업데이트되지 않습니다. 천체 위치 데이터가 매 프레임 태양/지구 고도를 제공해야 동적 밝기 조절이 작동합니다.

---

## 기술 참고

### 파일 구조

| 파일 | 역할 |
|---|---|
| `effects/config.py` | `StarfieldConf`, `EarthshineConf` 데이터클래스 |
| `effects/starfield.py` | 별 카탈로그 생성, HDR 텍스처 베이킹, DomeLight 관리 |
| `effects/earthshine.py` | 지구광 물리 모델, DistantLight 관리 |
| `celestial/stellar_engine.py` | 천체 위치(alt/az) 계산 엔진 |
| `environments/lunar_yard.py` | 환경에서 위 모듈들 조합 및 프레임 업데이트 |

### DistantLight vs SphereLight (지구광)

| | SphereLight | DistantLight |
|---|---|---|
| 감쇠 | 거리² 반비례 | 없음 |
| 물리적 적합성 | 점광원 | 평행광 (지구광에 적합) |
| 방향 설정 | 위치 이동 | Xform 회전 |
| Xform 구조 | 단순 translate | 2단계 (Parent Xform + Child Light) |

### HDR 텍스처 별 밝기 매핑

| 등급 범위 | HDR 값 | 픽셀 크기 | 설명 |
|---|---|---|---|
| mag < 0 | ~30 | 5×5 (Gaussian glow) | 가장 밝은 별 (시리우스급) |
| 0 ≤ mag < 2 | ~3~30 | 3×3 (Gaussian glow) | 밝은 별 |
| mag ≥ 2 | ~0.003~3 | 1×1 (single pixel) | 대부분의 별 |

---

## 태양 렌더링

### 연속 위치 기반 조명

태양은 `SunController`가 관리하는 `DistantLight`로 렌더링됩니다. `StellarEngine`이 매 프레임 계산하는 alt/az를 기반으로 DistantLight의 **방향(orientation)**이 연속적으로 갱신됩니다.

태양 intensity는 고정이며, 수평선 통과 시 별도의 on/off 제어가 없습니다. DistantLight의 방향이 수평선 아래를 향하면 빛이 지면 위쪽에서 오지 않으므로 자연스럽게 지표면 조도가 감소합니다.

### Xform 계층 구조

`SunController`와 `Earthshine` 모두 동일한 2단계 구조를 사용합니다:

```
Parent Xform (alt/az 회전, 매 프레임 갱신)
  └── Child DistantLight (고정 pre-rotation: [0.5, 0.5, -0.5, -0.5])
```

`StellarEngine.convert_alt_az_to_quat(alt, az)`로 변환한 quaternion이 Parent Xform에 적용됩니다.

---

## 지구 시각적 렌더링

### 렌더링 방식

지구는 `EarthController`가 관리하는 USD 메시(`Earth.usd`)로 렌더링됩니다. OmniPBR 재질에 지구 텍스처가 매핑되어 있으며, `StellarEngine`의 alt/az 데이터를 기반으로 위치가 갱신됩니다.

### Emissive 자발광

태양이 수평선 아래로 내려가면 씬에 조명이 없어지므로, OmniPBR 재질만으로는 지구가 보이지 않습니다. 실제로는 태양이 달의 수평선 아래에 있어도 우주 공간에서는 여전히 지구를 비추고 있으므로, OmniPBR의 **emissive 속성**을 활성화하여 야간에도 지구가 자체 발광하도록 합니다.

emissive 설정은 `omni.kit.commands.ChangeProperty`를 통해 런타임에 등록됩니다 (OmniPBR MDL 셰이더는 USD에 직접 input을 생성하는 방식으로는 인식하지 못함).

---

## 좌표계 정합성

### 씬 좌표계 = 달 표면 로컬 접선면

| 좌표 | 의미 |
|---|---|
| 씬 XY 평면 | 관측자 위치의 로컬 수평면 |
| 씬 +Z | 관측자 위치의 법선 벡터 (달 중심 → 관측자 방향) |

`StellarEngine`의 `altaz()`는 관측자의 **로컬 수평면** 기준으로 고도/방위각을 반환합니다. 씬의 좌표계가 이 로컬 수평면과 동일하므로, 위도/경도에 따른 별도의 법선 벡터 회전 변환이 필요 없습니다.

```
Skyfield alt = 0°   ←→  씬 XY 평면 (지면)
Skyfield alt = 90°  ←→  씬 +Z (천정)
```

위도/경도를 변경하면 `StellarEngine`이 해당 위치에서의 천체 alt/az를 다시 계산하고, 씬은 자동으로 그 위치의 로컬 프레임이 됩니다.

### 조석 고정과 지구 가시성

달은 조석 고정(tidal locking)으로 항상 같은 면이 지구를 향합니다:

| 관측 위치 | 지구 가시성 |
|---|---|
| Nearside 중심 (0°, 0°) | 항상 천정 근처, 칭동으로 ±8° 진동 |
| Nearside 가장자리 (limb) | 수평선 근처에서 칭동에 의해 출몰 |
| Farside | 절대 안 보임 |

현재 기본 좌표 (46.8°N, 26.3°W)는 nearside 중심에서 ~52° 거리이며, 지구 고도는 +29°~+45° 범위로 수평선 아래로 내려가지 않습니다.

### Ephemeris 유효 범위

JPL DE421 ephemeris의 유효 범위는 **1900~2050년**입니다. `time_scale: 36000`에서 시뮬레이션 시작(2024-05-01) 기준 약 **390분(6.5시간)** 후 한계에 도달합니다. 장시간 실행 시 시작 시각을 조정하거나 time_scale을 낮춰야 합니다.
