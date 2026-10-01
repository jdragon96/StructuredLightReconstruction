# Structured Light 최신 연구 동향 (2024–2026)

이 저장소가 구현한 두 파이프라인 —
[MultiFrequencyPhaseShift](../MultiFrequencyPhaseShift/README.md) (Gray code + 다중 주파수 PSP) 와
[ColorCoded](../ColorCoded/README.md) (컬러 스트라이프 + RGB 3-step) — 가
현재 학계/산업에서 어디쯤 위치하는지 파악하기 위한 서베이 문서다.

조사 기준일: **2026-08-02**. 2024년 하반기 ~ 2026년 7월 사이 출판된 리뷰 2편과
주요 논문 7편을 대상으로 했다.

---

## 0. 세 줄 요약

1. **정밀도가 목적이면 여전히 "N-step 위상 시프트 + 시간축 언래핑(temporal unwrapping)"이 표준**이다.
   2024~2026년 리뷰 두 편 모두 이 조합을 정확도 기준선(baseline)으로 삼는다.
   이 저장소의 `MultiFrequencyPhaseShift`가 정확히 여기에 해당한다.
2. **속도가 목적이면 "1-bit 바이너리 디포커싱"이 사실상 유일한 답**이다. DLP의
   8-bit 투사는 ~120 Hz인데 1-bit는 4 kHz~20 kHz로, 10~100배 차이가 난다.
3. **딥러닝은 "단일 촬영(single-shot)" 쪽으로 완전히 이동**했지만, 2026년 논문들이
   그 신뢰성에 정면으로 문제를 제기했다 — 네트워크가 프린지 위상이 아니라
   **물체 실루엣을 보고 깊이를 추측하는 지름길(shape-prior shortcut)**을 학습한다는
   것이 실측으로 드러났다.

---

## 1. 조사한 논문 목록

| # | 논문 | 출처 / 날짜 | 분류 |
|---|------|-------------|------|
| P1 | Review of fringe projection profilometry: from geometric triangulation to computational 3D imaging | Light: Advanced Manufacturing 7(2), 2026-06-03 | 리뷰 |
| P2 | Fringe-Based Structured-Light 3D Reconstruction: Principles, Projection Technologies, and Deep Learning Integration | Sensors 25(20):6296, 2025-10 | 리뷰 |
| P3 | Single-shot super-resolved FPP (SSSR-FPP): 100,000 fps 3D imaging with deep learning | Light: Science & Applications, 2025-02-07 | 고속 |
| P4 | Structured light with a million light planes per second | arXiv 2411.18597, rev. 2025-07 | 고속 / 이벤트카메라 |
| P5 | Comprehensive ML Benchmarking for FPP with Photorealistic Synthetic Data | arXiv 2601.08900, 2026-01-13 | 벤치마크 |
| P6 | Diagnosing Shape-Prior Shortcuts in Long-Range Single-Shot FPP | arXiv 2606.17093, 2026-06-13 | 딥러닝 신뢰성 |
| P7 | Deep Learning-based Single-Shot Composite FPP with Pixel-Wise Uncertainty Quantification (HSURE-CFPP) | arXiv 2601.02572, 2026-01-05 | 단일촬영 |
| P8 | GUSLO: General and Unified Structured Light Optimization | arXiv 2501.14659, rev. 2025-11 | 캘리브레이션 |
| P9 | Physics-informed deep learning for fringe pattern analysis (PI-FPA) | Opto-Electronic Advances, 2024 | 딥러닝 일반화 |

---

## 2. 리뷰 논문 분석

### 2.1 P1 — Light: Advanced Manufacturing 리뷰 (2026-06)

Wu, Guo, Chen 외 (Sichuan University + Fraunhofer IOF + TU Ilmenau).
제목이 이 분야의 방향성을 그대로 요약한다: **"기하학적 삼각측량에서 계산적 3D 이미징으로"**.

**패턴 코딩 분류 체계**

| 계열 | 대표 기법 | 촬영 장수 | 성격 |
|------|-----------|-----------|------|
| 시간 코딩 | N-step 위상 시프트 | 3~수십 | 정확도 최고, 정적 물체 |
| 시공간 코딩 | Gray code | 8~12 | 견고, 절대 인덱스 |
| 단일 이미지 | Fourier transform profilometry (FTP) | 1 | 동적 가능, 경계 취약 |
| 컬러 인코딩 | 컬러 프린지 / 스트라이프 | 1 | 표면 색상에 취약 |
| 바이너리 디포커싱 | 1-bit 패턴 + 광학 블러 | 3~4 | 초고속 |
| 복합/하이브리드 | 위 조합 | — | 실무 주류 |

리뷰가 인용하는 성능 지표:
- **micro FTP: 10,000 fps** — 단일 이미지 계열의 속도 상한 사례
- 정확도는 기법에 따라 서브밀리미터 ~ 마이크로미터

**남은 과제로 지목한 것** — 고반사/반투명 물체, 동적 장면의 모션 유발 오차,
복잡한 형상에서의 위상 언래핑 신뢰성, 실시간 처리 효율, 주변광 변화에 대한 견고성.

> 이 저장소 관점: `ColorCoded`가 쓰는 행 단위 Viterbi DP는 "복잡한 형상에서의 언래핑
> 신뢰성" 문제에 대한 고전적 처방이다. 리뷰는 이 문제가 아직 해결되지 않았다고 본다.

### 2.2 P2 — Sensors 리뷰 (2025-10)

Zhang, Wang, Li 외 (Tsinghua SIGS + Pengcheng Laboratory). DOI `10.3390/s25206296`.
P1보다 **하드웨어(투사 방식)** 비교가 훨씬 구체적이다.

**FPP vs PMD 이분법**

- **FPP (Fringe Projection Profilometry)** — 확산 반사면 대상. 위상 → 깊이 직접 복원.
  이 저장소가 하는 일.
- **PMD (Phase Measuring Deflectometry)** — 경면/고반사면 대상. 반사된 프린지의
  위상에서 **표면 기울기(gradient)**를 구하고 적분해서 형상을 얻는다.
  스테레오 PMD는 나노미터급 상대 깊이 정확도를 보고한다.

**위상 언래핑 전략 정리** (이 저장소와 직접 관련)

시간축 언래핑(TPU) 3종:

1. **Gray-code 언래핑** — 바이너리 Gray code로 프린지 차수를 픽셀 단위로 인코딩.
   `MultiFrequencyPhaseShift`의 1단계.
2. **다중 주파수 언래핑** — coarse-to-fine. 프린지 차수는

   ```
   k_h(x,y) = Round( ( (f_h/f_l)·Φ_l(x,y) − φ_h(x,y) ) / 2π )
   ```

   `MultiFrequencyPhaseShift`의 cascade unwrapping이 정확히 이 식이다.
3. **다중 파장(헤테로다인)** — 비트 주파수로 합성 파장을 만든다.

   ```
   λ_eq = λ₁λ₂ / (λ₁ − λ₂)
   ```

공간축 언래핑(SPU) 2종 — 품질 지도 기반 flood-fill, Goldstein branch-cut.
둘 다 오차 전파에 취약해 정밀 측정에서는 보조 수단으로만 쓴다.

**위상 시프트 단계 수(N)** — 리뷰는 보편적 정답을 제시하지 않고,
위상 노이즈가 **N이 늘수록 감소(대략 1/√N)** 한다는 관계만 제시한다.
실무에서 3-step(최소)과 4-step(감마 오차에 유리)이 갈리는 이유가 여기 있다.

**투사 기술 비교 — 이 표가 이 리뷰의 핵심 기여다**

| 항목 | DLP | MEMS 레이저 스캐닝 |
|------|-----|--------------------|
| 정확도 | 10⁻³ mm | 10⁻³ mm |
| 속도 | ~120 fps | **>1000 fps** |
| 해상도 | ~1K px | >4K px |
| 소비전력 | ~50 W | **~5 W** |
| 비용 | ~$2,000 | **~$500** |
| 피사계 심도 | 렌즈 제약 | **초대형 (렌즈 불필요)** |
| 크기 | 큼 | 소형 |

리뷰의 결론은 명확하다 — **"MEMS 기반 구조광 투사가 DLP의 강력한 대안으로 부상 중"**.
다만 MEMS는 스캔 미러 진동, 레이저 선폭에 의한 window smoothing, 광원 세기 변동,
그리고 **기존 핀홀 프로젝터 모델이 안 맞아서 별도 캘리브레이션 모델(unified,
iso-phase surface, phase-angle)이 필요**하다는 실용적 장벽이 있다.

기타 투사 방식: 간섭 기반 <0.1 mm / ~50 fps, 물리 격자 10⁻³ mm / ~100 fps,
LCD 10⁻² mm / ~50 fps.

---

## 3. 개별 논문 분석

### 3.1 P3 — SSSR-FPP: 100,000 fps (Light: Science & Applications, 2025-02)

Wang 외, 난징이공대 SCILab (Qian Chen / Chao Zuo 그룹). 현재 FPP 속도 기록.

**핵심 아이디어** — 초고속 촬영에서는 카메라를 작은 ROI로 잘라야 프레임레이트가
나오는데, 그러면 해상도와 SNR이 무너진다. 이걸 **CNN 초해상도로 되살린다**.

- **CNN1**: 저해상도(160×160) 프린지 → **3배 초해상(480×480)** sin/cos 위상 성분
- **CNN2**: 프린지 차수 모호성 해소 → 절대 위상
- 카메라 2대를 서로 다른 각도로 배치해 절대 위상 정보 확보 (기하 구속)

**성능**

| 지표 | 값 |
|------|-----|
| 프레임레이트 | 100,000 fps (160×160 윈도우) |
| 공간 해상도 | 0.891 lp/mm (3배 향상) |
| 측정 오차 | < 80 μm |
| 평균 절대 위상 오차 | 0.0257 rad |

**하드웨어** — Phantom V611 고속 CMOS 2대(20 μm 픽셀, 최소 노출 9.5 μs),
커스텀 DLP(XGA 1024×768 DMD) + FPGA 동기화, **바이너리 프린지 투사**.

> 주목할 점: 100k fps를 만든 건 딥러닝 자체가 아니라 **1-bit 바이너리 투사 +
> ROI 크롭 + FPGA 동기화**라는 하드웨어 조합이고, 딥러닝은 그 대가로 잃은
> 화질을 복구하는 역할이다. 이 역할 분담이 최근 고속 FPP의 전형적 구조다.

### 3.2 P4 — 초당 100만 광평면 (arXiv 2411.18597)

Sirikonda, Chakravarthula, Gkioulekas, Pediredla.

DLP를 아예 버리고 **음향광학(acousto-optic) 광 스캐닝 소자**로 광평면을
초당 **200만 개** 투사한다. 이벤트 카메라와 결합해 **풀프레임 1000 fps 3D 스캔**,
기존 이벤트 기반 구조광 대비 4배.

기존 이벤트 카메라 구조광의 병목은 카메라가 아니라 **조명을 흔드는 속도**였다는
진단이 핵심이다. 추가로, 이벤트 카메라의 비동기 동작을 활용한 적응형 스캔 전략으로
카메라의 이론적 한계를 한 자릿수 넘어섰다고 주장한다.

관련 흐름: 혼합 반사(확산 + 2-bounce 경면 + 다중 반사) 장면을 에피폴라 구속으로
분해하는 이벤트 기반 연구가 Nature Communications 2026에 실렸고,
이벤트 기반 구조광 캘리브레이션 방법(주파수 정보 이용)도 2026-01에 나왔다.
**이벤트 카메라 + 구조광은 2026년 현재 가장 빠르게 논문이 쌓이는 분야다.**

### 3.3 P5 — ML 벤치마크 + 합성 데이터셋 (arXiv 2601.08900, 2026-01)

Anush Lakshman S, Adam Haroon, Beiwen Li (Iowa State).

FPP 딥러닝의 만성적 문제 — **공개 데이터셋 부재** — 를 정면으로 다룬다.
NVIDIA Isaac Sim으로 만든 **최초의 오픈소스 포토리얼리스틱 FPP 합성 데이터셋**:
50개 물체, 프린지 이미지 15,600장, 깊이 복원 300개. 1.5–2.1 m 장거리, 단일 촬영.

**결과가 정직해서 인용 가치가 높다:**

- UNet 포함 4개 아키텍처 비교 → UNet이 최고지만 **아키텍처 간 성능 차가 작다**.
  저자들의 해석: 문제는 모델 설계가 아니라 **입력 정보량의 한계**다.
- **오차는 고전 FPP의 서브밀리미터 정확도에 한참 못 미친다.**
- 깊이 정규화 방식이 아키텍처보다 영향이 크다 — 개별 정규화가 raw depth 대비 **9.1배** 개선.
- 6개 손실함수 중 hybrid L1 최적.
- **배경 프린지를 제거하면 모든 방법이 심각하게 무너진다** → 배경 패턴이 필수적인
  공간 기준 역할을 하고 있다.

마지막 항목은 다음 논문의 복선이다.

### 3.4 P6 — Shape-Prior Shortcut (arXiv 2606.17093, 2026-06) ★

Haroon, Lakshman, Fleming, Li. **이번 조사에서 가장 중요한 논문.**

질문: 단일 촬영 FPP 네트워크는 정말 **프린지 위상에서** 깊이를 복원하는가,
아니면 **깊이와 상관관계가 있는 물체 형상 단서**를 보고 있는가?

1 m 이상 거리에서는 프린지 차수 정보 없이 깊이를 푸는 문제가 심각하게
ill-posed 해지고, 오차가 작업거리의 **제곱에 비례**해 커진다.

**증거 3종:**

| 실험 | 결과 |
|------|------|
| 선형 프로빙 | 특징맵에서 **엣지가 깊이보다 2.82배 더 잘 디코딩됨** |
| Grad-CAM | 어텐션이 프린지보다 **경계에 1.28배 편중** |
| 평면 테스트 | 특징 없는 평면 → 유효한 프린지가 있는데도 **깊이가 배경값으로 붕괴** |

**결론이 강경하다** — 이 지름길은 가설 공간 자체에 내재하므로
**"데이터를 더 넣거나 모델을 키워도 사라지지 않는다"**. 형상 사전지식 해법을
설계 단계에서 배제하는 아키텍처 재설계가 필요하다.

> 실무 함의: 단일 촬영 딥러닝 FPP의 벤치마크 수치를 액면가로 믿으면 안 된다.
> 평가 시 **평면/무특징 물체 테스트**를 반드시 포함해야 한다.
> 반대로, 이 저장소처럼 위상을 결정론적으로 푸는 방식은 이 실패 모드가 원천적으로 없다.

### 3.5 P7 — HSURE-CFPP: 불확실성 정량화 (arXiv 2601.02572, 2026-01)

Kong, Bao, Yalew, Adesso, Piano (Nottingham).

문제 인식: 래핑 위상의 2π 모호성 때문에 절대 위상 복원은 보통 여러 장의 패턴을
요구하고, 그 대가로 시간 해상도를 잃는다.

접근: **단일 복합(composite) 프린지**에서 래핑 위상 계산용
**분자/분모 비율을 직접 예측**하되, 이를 **이분산(heteroscedastic) 스냅샷 앙상블**로 한다.

- **데이터 불확실성** — 이분산 가능도로 픽셀별 노이즈 분산을 함께 추정
- **모델 불확실성** — 스냅샷 앙상블 간 분산으로 정량화

핵심 성과는 정확도 수치가 아니라 **예측 불확실성이 실제 복원 오차와 잘 상관된다**는
점이다. P6이 제기한 "딥러닝 FPP를 믿을 수 있는가"에 대한 실용적 대응이다 —
믿을 수 없는 픽셀을 스스로 표시하게 만드는 것.

### 3.6 P8 — GUSLO (arXiv 2501.14659)

Wan, Su, Wang.

기존 구조광의 두 가지 실무 고통을 겨냥한다:
(a) 장면마다 수동 파라미터 튜닝이 필요한 캘리브레이션,
(b) 특정 패턴 종류에만 맞춰진 최적화 프레임워크.

- **단일 촬영 캘리브레이션** — 2D 삼각분할 보간으로 희소 대응점을 조밀한 대응장으로 확장
- **아티팩트 인식 광도 적응** — 명시적 전달 함수로 투사-촬영 간 광도 왜곡 보정

**바이너리 / 스페클 / 컬러 코딩 패턴 전반**에 걸쳐 평가했다는 점이 특징이다
(산업 검사 + 문화재 디지털화 시나리오). 패턴 종류에 무관한 통합 최적화라는
방향 자체가 이 저장소처럼 여러 패턴 파이프라인을 갖는 프로젝트에 시사점이 있다.

### 3.7 P9 — PI-FPA: 물리 정보 기반 (Opto-Electronic Advances, 2024)

경량 DNN에 **학습 강화 FTP 모듈**을 결합하고, 물리 사전지식을 네트워크 구조와
손실함수 양쪽에 심는다. 목적은 정확도보다 **일반화** —
대량의 고품질 학습 데이터 없이도 미학습 물체에 대해 신뢰할 만한 위상을 낸다.

같은 흐름의 최근 연구들:
- **Untrained(비학습) 위상 복원** — 사전 학습 없이, 장면 독립적 물리 구속만으로
  손실을 최소화. 장면이 자주 바뀌거나 데이터가 적을 때 기존 딥러닝보다 견고.
- **자기지도 모델 기반 학습** (2025-10) — 간섭계 모델을 이용해 ground truth 위상 없이 학습.
- **CMNet** (J. Optics, 2026-01) — Res-UNet + Mamba(선택적 상태공간 모델) 하이브리드로
  단일 프레임 위상 언래핑. Transformer 이후 Mamba가 이 분야에 진입한 사례.

> P5/P6과 함께 읽으면 그림이 선명하다: **지도학습 단일촬영 FPP는 일반화와
> 신뢰성에서 벽에 부딪혔고, 2026년의 대응은 (a) 물리 구속 내장, (b) 불확실성
> 정량화, (c) 아키텍처 재설계 세 갈래다.**

---

## 4. 최근 가장 많이 쓰이는 패턴

용도별로 뚜렷하게 갈린다. "무엇이 제일 많이 쓰이나"의 답은 **단일 패턴이 아니라
하이브리드 조합**이다.

### 4.1 Tier 1 — 정밀 측정 표준: N-step PSP + 시간축 언래핑

**2024–2026 리뷰 두 편이 공통으로 정확도 기준선으로 삼는 조합.** 산업 검사,
문화재 스캔, 리버스 엔지니어링의 사실상 표준.

두 가지 변종이 경쟁한다:

| | Gray code + PSP | 다중 주파수 / 헤테로다인 PSP |
|---|---|---|
| 촬영 장수 | 8~12(Gray) + 3~4(PSP) ≈ **11~16장** | **9~12장** (3주파수 × 3~4 step) |
| 프린지 차수 | 결정론적, 오차 전파 없음 | 반올림 기반, 노이즈에 민감 |
| 경계 취약점 | Gray code 이진화 경계에서 1-stripe 점프 | 저주파 위상 오차가 고주파로 전파 |
| 최근 선호 | 견고성 우선 시 | **장수 절감 우선 시 — 최근 더 흔함** |

> 이 저장소의 `MultiFrequencyPhaseShift`는 **양쪽을 다 쓴다** —
> Gray code로 최저 주파수(`2^nGrayBits`)의 절대 위상을 결정론적으로 고정하고,
> 그 위에 다중 주파수 cascade를 얹는다. 리뷰 기준으로는 가장 보수적이고
> 견고한 구성이다. 촬영 장수를 줄이고 싶다면 Gray code를 빼고
> 3-주파수 헤테로다인만 남기는 것이 최근 주류에 더 가깝다.

**N의 선택**: 위상 노이즈는 대략 1/√N로 줄어든다.
- **3-step** — 이론적 최소. 장수가 곧 속도인 동적 측정에서 선택.
- **4-step** — 감마 비선형(짝수 고조파) 상쇄에 유리. 정적 측정의 실무 기본값.
- **N≥5** — 노이즈가 지배적이거나 고조파 왜곡이 심한 경우에만.

### 4.2 Tier 2 — 고속 측정 표준: 1-bit 바이너리 디포커싱

**속도가 목적일 때는 대안이 거의 없다.** 8-bit 정현파 대신 1-bit 흑백 패턴을
투사하고 프로젝터를 의도적으로 디포커싱해 광학적으로 정현파를 만든다.

| 방식 | TI DLP4500 기준 투사 속도 |
|------|---------------------------|
| 8-bit 정현파 | ~120 Hz |
| **1-bit 바이너리** | **>4,000 Hz** (일부 하드웨어 최대 20 kHz) |

10~100배 차이라 고속 FPP 논문은 거의 예외 없이 이걸 쓴다 —
P3(SSSR-FPP)의 100k fps도 바이너리 투사가 전제다.

**대가**: 디포커스 양에 따라 정현파 품질이 변해 위상 오차가 생긴다.
그래서 최근 연구는 (a) 디더링/최적화된 바이너리 패턴 생성,
(b) 딥러닝 기반 디포커스 오차 보정, (c) 디지털 상관 기반 최적 디포커스 결정에 집중한다.

### 4.3 Tier 3 — 단일 촬영(single-shot): 용도별로 분화

동적 장면용. 하나로 수렴하지 않고 목적에 따라 갈라져 있다.

| 방식 | 쓰이는 곳 | 상태 |
|------|-----------|------|
| **스페클(pseudo-random) + 액티브 스테레오** | 소비자/로보틱스 뎁스 카메라 계열 | 성숙, 정밀도 낮음 |
| **컬러 코딩 (De Bruijn / 해밍)** | 조밀한 단일촬영 복원 | 표면 색상·색수차에 취약, GAN 기반 스트라이프 엣지 검출로 보완 시도 |
| **복합(composite) 프린지** | 정밀 단일촬영 | P7처럼 딥러닝과 결합 |
| **FTP / 윈도우드 FTP** | 고속·단순 형상 | 고전, 경계에서 취약 |
| **딥러닝 fringe-to-phase** | 연구 주도 | **P5/P6이 신뢰성에 제동** |

> 이 저장소의 `ColorCoded`는 여기 두 번째 줄에 해당한다 —
> 해밍 인접 코드 컬러 스트라이프(27 stripe, 8색 팔레트) + RGB 3-step 위상.
> 컬러 코딩 단일촬영이 De Bruijn 계열 최적화와 딥러닝 스트라이프 세그멘테이션
> 쪽으로 진화 중인 것이 최근 흐름이다 (GUSLO, HEAR-GAN,
> 2025-12 단일촬영 다중선 스트라이프 인식 등). 즉 **컬러 코딩의 약점인
> "스트라이프 라벨링"을 학습 기반으로 대체**하는 방향인데,
> 이 저장소는 그 자리를 행 단위 양방향 Viterbi DP로 채우고 있다.

### 4.4 Tier 4 — 부상 중인 하드웨어

| 기술 | 성숙도 | 요지 |
|------|--------|------|
| **MEMS 레이저 스캐닝** | 상용화 임박 | DLP 대비 8배 속도, 1/4 비용, 1/10 전력, 초대형 심도. 캘리브레이션 모델이 걸림돌 |
| **이벤트 카메라 + 구조광** | 연구 활발 | 1000 fps 풀프레임(P4), 높은 동적 범위, 혼합 반사 분해 |
| **음향광학 스캐닝** | 초기 | 초당 200만 광평면 (P4) |
| **메타표면 프린지 생성** | 초기 | 편광 인코딩 메타표면으로 시스템 소형화 |

---

## 5. 이 저장소에 대한 시사점

조사 결과를 이 코드베이스에 대입하면:

**잘 맞는 부분**
- `MultiFrequencyPhaseShift`는 리뷰들이 정확도 기준선으로 삼는 구성 그 자체다.
  Gray code + cascade unwrapping은 P2가 정리한 TPU 1·2번의 조합이다.
- 결정론적 위상 복원이라 P6이 지적한 shape-prior shortcut 실패 모드가 없다.

**최근 흐름과 격차가 있는 부분**
- 촬영 장수가 많다. 최근 주류는 Gray code를 빼고 **3-주파수 헤테로다인**으로
  9~12장에 수렴하는 쪽이다.
- **1-bit 바이너리 디포커싱 미지원.** 이 저장소는 렌더 기반 데이터셋을 쓰므로
  당장 필요는 없지만, 실물 하드웨어로 확장할 때 8-bit 정현파 투사는
  ~120 Hz에서 막힌다.
- 감마 보정 / 고조파 오차 보상 단계가 파이프라인에 명시적으로 없다.
  4-step PSP를 쓰면 짝수 고조파는 자연 상쇄되지만 홀수 고조파는 남는다.

**`ColorCoded` 쪽에서 검토해볼 만한 것**
- P8(GUSLO)의 **아티팩트 인식 광도 적응**은 컬러 스트라이프 분류
  (`computeColorRatio` → `classifyColorCode`)의 견고성을 직접 겨냥한 기법이다.
  현재의 white 레퍼런스 기반 정규화보다 한 단계 정교하다.
- 해밍 인접 코드 대신 **De Bruijn 시퀀스**를 쓰면 윈도우 유일성이 수열 정의로
  보장되어 `windowToStart` 앵커 매칭이 더 촘촘해진다.

---

## 참고문헌

**리뷰**
- [Review of fringe projection profilometry: from geometric triangulation to computational 3D imaging](https://www.light-am.com/article/doi/10.37188/lam.2026.074) — Light: Advanced Manufacturing 7(2), 2026
- [Fringe-Based Structured-Light 3D Reconstruction: Principles, Projection Technologies, and Deep Learning Integration](https://pmc.ncbi.nlm.nih.gov/articles/PMC12567414/) — Sensors 25(20):6296, 2025

**고속 측정**
- [Single-shot super-resolved FPP (SSSR-FPP): 100,000 fps 3D imaging with deep learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC11802878/) — Light: Sci. Appl., 2025
- [Structured light with a million light planes per second](https://arxiv.org/abs/2411.18597) — arXiv 2411.18597
- [Accurate and fast event-based shape measurement of mixed reflectance scenes](https://www.nature.com/articles/s41467-026-72254-6) — Nature Communications, 2026

**딥러닝 신뢰성 / 벤치마크**
- [Comprehensive Machine Learning Benchmarking for FPP with Photorealistic Synthetic Data](https://arxiv.org/abs/2601.08900) — arXiv 2601.08900
- [Diagnosing Shape-Prior Shortcuts in Long-Range Single-Shot FPP](https://arxiv.org/abs/2606.17093) — arXiv 2606.17093
- [Deep Learning-based Single-Shot Composite FPP with Pixel-Wise Uncertainty Quantification](https://arxiv.org/abs/2601.02572) — arXiv 2601.02572
- [Physics-informed deep learning for fringe pattern analysis](https://www.oejournal.org/oea/article/doi/10.29026/oea.2024.230034) — Opto-Electronic Advances, 2024
- [A Taxonomy of Deep Learning in Single-Frame Fringe Projection Profilometry](https://link.springer.com/article/10.1007/s41871-025-00285-6) — Nanomanufacturing and Metrology, 2025
- [CMNet: phase unwrapping from single structured light image with a CNN-Mamba network](https://iopscience.iop.org/article/10.1088/2040-8986/ae32a5) — J. Optics, 2026

**캘리브레이션 / 최적화**
- [GUSLO: General and Unified Structured Light Optimization](https://arxiv.org/abs/2501.14659) — arXiv 2501.14659
- [Calibration method for event-based structured light systems using frequency information](https://www.sciencedirect.com/science/article/abs/pii/S0263224125018664) — Measurement, 2026

**단일촬영 컬러 코딩**
- [Single-shot multi-line structured light stripe recognition based on deep learning](https://opg.optica.org/ao/abstract.cfm?uri=ao-64-36-10757) — Applied Optics 64(36), 2025
- [Color Structured Light Stripe Edge Detection Method Based on GAN](https://www.mdpi.com/article/10.3390/app13010198) — Applied Sciences, 2023
