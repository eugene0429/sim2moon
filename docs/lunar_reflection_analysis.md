# 월면 조명 및 반사 모델(Hapke Model) 적용 분석 보고서

## 1. 현재 프로젝트의 MDL 파일 분석
현재 프로젝트의 월면(Lunar Regolith) 재질은 다음 세 가지 주요 MDL 파일로 구현되어 있습니다:
- [LunarRegolith8k.mdl](file:///home/sim2real1/new_lunar_sim/assets/Textures/LunarRegolith8k.mdl)
- [LunarRegolith8k_antiTile.mdl](file:///home/sim2real1/new_lunar_sim/assets/Textures/LunarRegolith8k_antiTile.mdl)
- [LunarRegolith8k_stochastic.mdl](file:///home/sim2real1/new_lunar_sim/assets/Textures/LunarRegolith8k_stochastic.mdl)

**분석 결과:**
상기 MDL 파일들은 NVIDIA의 표준 `OmniPBR` 기반 또는 `df::microfacet_ggx_smith_bsdf` 및 `df::diffuse_reflection_bsdf`를 활용한 전형적인 **GGX 기반의 물리 기반 렌더링(PBR) 모델**을 사용하고 있습니다. 
*   **Diffuse 한계:** Diffuse 반사는 본질적으로 Lambertian 분포에 가깝게 동작하므로, 시야각이나 조명각에 관계없이 균일한 난반사가 발생합니다.
*   **Specular 한계:** Specular 반사는 GGX 마이크로패싯 모델을 사용하고 있어, 금속이나 젖은 표면, 일반적인 암석의 빛 반사 매커니즘은 잘 표현하지만 다공성(porous)이고 거친 월면 먼지의 독특한 후방 산란(Backscattering) 현상은 모사하지 못합니다.

---

## 2. 실제 월면 조명 및 반사 조건 리서치 (Hapke 모델)
달 표면(월면토, Regolith)의 빛 반사 특성은 지구상의 일반적인 흙이나 암석과는 완전히 다른 거동을 보입니다. 주요 특성은 다음과 같습니다.

1.  **롬멜-젤리거 법칙 (Lommel-Seeliger Law):** 월면과 같은 미립자 다공성 매질은 매우 강한 난반사를 일으키며, 테두리가 어두워지는 현상(Limb darkening)이 거의 발생하지 않아 보름달이 평면적인 원반처럼 보이게 만듭니다.
2.  **층위 효과 및 충돌 흔적 (Shadow-Hiding Opposition Effect):** 다공성 먼지 입자들이 빛이 들어오는 방향에서 자신들의 그림자를 스스로 가리게 되어, **광원(태양)과 관찰자(카메라)의 각도(Phase Angle)가 0에 가까워질 때 밝기가 급격하게 증가**하는 '충(Opposition) 효과'가 일어납니다.
3.  **Hapke BRDF 모델:** 지구 외 행성 표면의 광도 측정에 가장 널리 사용되는 수리적 모델입니다. 입자의 단일 산란 알베도(Single Scattering Albedo), 매크로 러프니스, 충 효과 진폭 및 너비 등을 변수로 삼아 월면의 독특한 반사 특성을 가장 정확히 표현합니다.

---

## 3. 리얼리즘 극대화를 위한 구현 방안 및 복잡도/효과 비교

Isaac Sim 및 MDL(Material Definition Language) 환경에서 Hapke 모델에 기반한 사실적인 반사를 구현하기 위한 방안은 다음과 같습니다.

### 방안 A: MDL SDK를 활용한 C++ Custom BSDF 개발 (가장 현실적이고 정확함)
MDL 언어 자체는 BRDF 공식을 파일 스크립트(.mdl) 내에 직접 코딩하는 것을 허용하지 않습니다. (미리 정의된 노드의 조합만 가능). 따라서 Isaac Sim의 렌더러(RTX)가 해석할 수 있도록 C++ 레벨에서 확장을 해야 합니다.
*   **구현 방법:** NVIDIA MDL SDK를 사용하여 Hapke 공식을 수학적으로 정의한 새로운 Elemental BSDF 노드를 C++로 작성하고 플러그인화 합니다.
*   **구현 복잡도 (매우 높음 ⭐⭐⭐⭐⭐):** 렌더링 엔진 내부 구조에 대한 깊은 이해가 필요하며, Isaac Sim의 RTX 렌더러 파이프라인과 통합하기 위해 고난도의 커스텀 익스텐션 개발이 요구됩니다.
*   **효과 (매우 높음 ⭐⭐⭐⭐⭐):** 논문에 나오는 실제 월면과 파라미터(알베도, 러프니스 등)를 수치적으로 완전히 동일하게 적용 가능하며, 완벽한 충(Opposition) 효과를 구현할 수 있습니다.

### 방안 B: 측정된 BRDF 데이터(MBSDF) 굽기 및 적용
Hapke 수식을 MDL 코드로 직접 넣을 수는 없지만, 파일 포맷을 통해 데이터를 주입할 수는 있습니다.
*   **구현 방법:** Python 등의 외부 툴을 이용해 Hapke 모델 공식 기반의 반사율 데이터를 방사구(Hemisphere) 전체의 입사각/반사각에 대해 샘플링합니다. 이를 NVIDIA의 `.mbsdf`(Measured BSDF) 바이너리 포맷으로 변환한 뒤, MDL에서 `df::measured_bsdf` 노드로 불러와 적용합니다.
*   **구현 복잡도 (높음 ⭐⭐⭐⭐):** 수학적 샘플러 도구와 MBSDF 바이너리 생성 스크립트를 자체 제작해야 합니다.
*   **효과 (높음 ⭐⭐⭐⭐):** 정확한 빛 반사를 보여주며 시뮬레이션 환경에 부하를 적게 줍니다. 단, 파라미터 조절이 필요할 때마다 데이터를 다시 구워야(Baking) 하는 한계가 있습니다.

### 방안 C: 기존 MDL 노드(Oren-Nayar)를 활용한 근사 구현 (가장 현실적인 접근)
완전한 Hapke 모델은 아니지만, 현재의 단순 Lambertian에서 벗어나 훨씬 유사한 느낌을 내는 방법입니다.
*   **구현 방법:** `df::diffuse_reflection_bsdf(roughness)` 노드에 매우 높은 `roughness` 값을 부여하면 MDL 내부적으로 '오렌-나이어(Oren-Nayar)' 표면 난반사 모델로 작동합니다. 오렌-나이어 모델은 거친 표면의 후방 반사를 근사하게 모사하므로 달 표면 질감과 매우 유사한 Flattened look을 제공합니다. (마이크로패싯 모델의 Specular 비중을 최소화하고 Diffuse Roughness를 극대화)
*   **구현 복잡도 (매우 낮음 ⭐):** 기존의 [.mdl](file:///home/sim2real1/new_lunar_sim/assets/Textures/Sand.mdl) 파일 내 요소 배합(Composition) 파라미터와 노드 교체만으로 즉시 적용할 수 있습니다.
*   **효과 (보통 ⭐⭐⭐):** 완벽한 충돌(Opposition) 급상승 효과까지는 없지만, Lambertian보다 훨씬 물리적으로 타당한 거친 먼지의 질감(역반사에 가까운 평면화된 라이팅)을 리얼타임 성능 저하 없이 달성합니다.

---

## 4. 결론 및 권장 사항
현재 프로젝트는 일반적인 그래픽스 에셋 수준의 PBR 쉐이딩을 사용하고 있어 "물리적으로 조명이 완전히 차단된 그림자 공간"과 "직사광이 내리쬐는 극단적 밝기 구역"에서의 로버 카메라 머신비전 시뮬레이션 결과에 오차를 유발할 수 있습니다.

1.  **단기 추천:** [.mdl](file:///home/sim2real1/new_lunar_sim/assets/Textures/Sand.mdl) 파일의 Diffuse를 **Oren-Nayar 모델(거칠기 1.0)로 수정**하고, Specular를 극도로 약화시키는 셰이딩 조정(방안 C)을 즉시 적용하여 기본 Lambertian의 이질감을 줄이는 것을 권장합니다.
2.  **중장기 추천:** 프로젝트의 핵심 목표가 **광학 비전 센서를 위한 초정밀 Ground Truth 시뮬레이션**이라면, 시뮬레이션 팀 내 렌더링 엔지니어를 통한 **MBSDF 데이터 베이킹 파이프라인(방안 B) 구축**을 권장합니다. 수식 파라미터를 동적으로 바꿀 필요가 없다면 C++ 플러그인 구현보다 개발 리스크를 크게 줄이면서 최고 수준의 화질을 확보할 수 있습니다.
