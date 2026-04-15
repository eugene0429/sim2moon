# 지형 재질 시스템

## 개요

달 시뮬레이션은 중앙 절차적 지형과 배경 지형 메시에 커스텀 MDL 재질을 사용합니다. 모든 지형 재질은 **월드 좌표 투영** (`project_uvw: true`)을 사용하여 메시 UV 레이아웃에 관계없이 일관된 텍스처 밀도를 보장합니다.

## 사용 가능한 재질

### LunarRegolith8k (원본)

- **파일**: `assets/Textures/LunarRegolith8k.mdl`
- **타입**: OmniPBR + 월드 좌표 투영
- **방식**: 단순 타일링
- **용도**: 중앙 지형 (40m × 40m), 근거리에서 타일링 아티팩트가 덜 눈에 띄는 경우

주요 파라미터:
```
texture_scale: float2(0.5)   → 2미터당 1회 타일링
project_uvw: true
world_or_object: true
```

### LunarRegolith8k_antiTile (멀티스케일 + 노이즈)

- **파일**: `assets/Textures/LunarRegolith8k_antiTile.mdl`
- **타입**: 커스텀 PBR (GGX specular + fresnel + diffuse)
- **방식**: 멀티스케일 타일링 + FBM 노이즈 밝기 변조
- **용도**: 단순 타일링의 반복 패턴이 눈에 띄는 대형 지형

#### 작동 원리

1. **월드 좌표 UV**: 월드 XY 좌표를 텍스처 좌표로 사용 (OmniPBR의 `project_uvw`와 동일)
2. **3단계 스케일 샘플링**: 같은 텍스처를 near/mid/far 스케일로 샘플링 후 가중 블렌딩
   - near: `base_scale` (2m당 1타일) — 디테일 담당
   - mid: `base_scale × mid_ratio` (~14m당 1타일) — 중거리 변화
   - far: `base_scale × far_ratio` (~125m당 1타일) — 대규모 색조 변화
3. **FBM 노이즈 변조**: 4옥타브 value noise로 밝기를 ±15% 변조하여 타일 경계를 은폐

#### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `base_scale` | 0.5 | 기본 타일링 밀도 (LunarRegolith8k과 동일) |
| `mid_ratio` | 0.14 | 중거리 스케일 비율 (base_scale × mid_ratio) |
| `far_ratio` | 0.016 | 원거리 스케일 비율 (base_scale × far_ratio) |
| `near_weight` | 0.2 | near 레이어 블렌딩 가중치 |
| `mid_weight` | 0.3 | mid 레이어 블렌딩 가중치 |
| `far_weight` | 0.5 | far 레이어 블렌딩 가중치 |
| `noise_frequency` | 0.001 | 노이즈 주파수 (낮을수록 큰 패치) |
| `noise_strength` | 0.15 | 밝기 변조 강도 (0 = 없음, 1 = 최대) |
| `albedo_add` | 1.0 | 밝기 가산값 (OmniPBR LunarRegolith8k과 동일) |

#### 튜닝 가이드

- **타일링이 아직 보임**: `near_weight`를 줄이고 `mid_weight`/`far_weight`를 높임
- **너무 뭉개지거나 밋밋함**: `near_weight`를 높여 디테일 복원
- **타일 경계가 보임**: `noise_strength`를 올림 (0.25~0.4 권장)
- **노이즈 패치가 너무 크거나 작음**: `noise_frequency` 조정

### LunarRegolith8k_stochastic (헥스 그리드 스토캐스틱)

- **파일**: `assets/Textures/LunarRegolith8k_stochastic.mdl`
- **타입**: 커스텀 PBR (GGX specular + fresnel + diffuse)
- **방식**: 헥스 그리드 스토캐스틱 타일링 + 셀별 연속 랜덤 회전
- **용도**: 반복 패턴 제거가 강력한 대안적 방식. 블렌딩 경계에 아티팩트가 생길 수 있음

#### 작동 원리

1. **헥스 그리드 분할**: UV 공간을 오프셋 헥스 셀로 분할
2. **셀별 랜덤 변환**: 각 셀에 해시 함수로 연속 랜덤 회전(0°~360°) + UV 오프셋 적용
3. **3샘플 블렌딩**: 각 픽셀에서 가장 가까운 3개 헥스 셀을 샘플링하고 거리 역수^4 가중치로 블렌딩

#### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `base_scale` | 0.5 | 타일링 밀도 (LunarRegolith8k과 동일) |
| `tile_size` | 1.0 | 헥스 타일 크기 (UV 단위). 클수록 이음새 아티팩트 감소 |

## 설정 방법

### 중앙 지형

환경 YAML의 `texture_path`를 변경:

```yaml
terrain_manager:
  texture_path: /LunarYard/Looks/LunarRegolith8k              # 원본
  # texture_path: /LunarYard/Looks/LunarRegolith8k_antiTile   # 멀티스케일
  # texture_path: /LunarYard/Looks/LunarRegolith8k_stochastic # 스토캐스틱
```

### 배경 지형

static asset 항목에 `material_override` 설정:

```yaml
static_assets_settings:
  parameters:
    - asset_name: background_landscape
      usd_path: Terrains/landscape_cropped/landscape_cropped.usd
      material_override: /LunarYard/Looks/LunarRegolith8k_antiTile
      # material_override: /LunarYard/Looks/LunarRegolith8k_stochastic
```

`material_override`는 static asset의 모든 Mesh 하위 prim에 지정된 재질을 재바인딩하여, 에셋에 내장된 UE4 기반 재질을 대체합니다.

### 재질 등록

모든 재질은 `environments/lunar_yard.py`의 `_load_materials()`에서 로드됩니다. 새로운 재질 변형을 추가하려면 `materials` 리스트에 항목을 추가:

```python
materials = [
    ("LunarRegolith8k", "assets/Textures/LunarRegolith8k.mdl"),
    ("LunarRegolith8k_antiTile", "assets/Textures/LunarRegolith8k_antiTile.mdl"),
    ("LunarRegolith8k_stochastic", "assets/Textures/LunarRegolith8k_stochastic.mdl"),
]
```

## PBR 일관성

세 재질 모두 동일한 PBR 파라미터를 공유하여 빛 반사 특성이 일치합니다:

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| specular_level | 0.26 | Fresnel F0 강도 |
| roughness_constant | 1.0 | 기본 거칠기 |
| roughness_influence | 0.13 | 거칠기 텍스처 기여도 |
| metallic | 0.0 | 비금속 (유전체) |
| bump_factor | 0.8 | 노멀맵 강도 |
| ao_to_diffuse | 0.35 | 앰비언트 오클루전 강도 |

커스텀 재질(antiTile, stochastic)은 OmniPBR의 반사 모델을 다음과 같이 재현합니다:
- `df::fresnel_layer`: specular_level에서 유도한 IOR로 프레넬 반사
- `df::microfacet_ggx_smith_bsdf`: GGX 스페큘러 반사
- `df::diffuse_reflection_bsdf`: 디퓨즈 반사

## 텍스처 에셋

모든 재질이 `assets/Textures/LunarRegolith8k/`의 동일한 텍스처 세트를 참조합니다:

| 파일 | 크기 | 용도 |
|------|------|------|
| `LunarRegolith8k_diff.png` | 321 MB | 디퓨즈 알베도 (8K) |
| `LunarRegolith8k_nor_gl.png` | 366 MB | 노멀맵 (OpenGL 형식) |
| `LunarRegolith8k_rough.png` | 285 MB | 거칠기 |
| `LunarRegolith8k_ao.png` | 120 MB | 앰비언트 오클루전 |

## 기술 참고사항

- **월드 좌표 투영**: 원본 LunarRegolith8k (`project_uvw: true`)과 커스텀 재질 모두 월드 좌표로 UV를 생성합니다. 이로써 서로 다른 UV 레이아웃을 가진 메시 간의 텍스처 밀도 불일치가 해소됩니다.
- **MDL 호환성**: 커스텀 재질은 `for` 루프를 사용하지 않고 (수동 언롤링), 텍스처를 재질 파라미터로 선언하며 (`uniform texture_2d`), `fresnel_layer`의 IOR에 `float` 타입을 사용합니다.
- **성능**: 멀티스케일(텍스처 샘플 9회)과 스토캐스틱(텍스처 샘플 12회)의 GPU 비용은 유사합니다. antiTile의 FBM 노이즈 연산과 stochastic의 헥스 그리드 연산이 서로 상쇄됩니다.
