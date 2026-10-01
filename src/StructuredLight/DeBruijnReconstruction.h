#pragma once

// DeBruijnReconstruction.h
//
// 위상 채널 없이 컬러 스트라이프 한 장(+ white 레퍼런스)만으로 3D 복원한다.
//
// ColorCoded.h 의 ColorCodedReconstruction 과의 차이:
//
//   ColorCoded          : color + white + RGB 3-step phase (3 장)
//                         위상이 sub-pixel 정밀도를 담당하고 컬러 코드는 fringe
//                         order 를 고정한다. 픽셀마다 3D 점이 나온다(dense).
//
//   DeBruijnReconstruction : color + white (2 장)
//                         고정 패턴 글라스는 정현파를 투사할 수 없으므로 위상이
//                         없다. 정밀도가 전적으로 스트라이프 경계의 sub-pixel
//                         위치에서 나오고, 3D 점은 경계마다 하나씩 나온다(sparse).
//
// 디코딩이 코드가 아니라 run 단위인 이유:
//   인접 동일색은 물리적으로 경계가 없으므로 카메라는 run-length 압축된 수열만
//   본다. B(3,4) 코드 72 스트라이프는 run 52 개로 관측되고, run 의 '색'만으로는
//   어떤 윈도우 길이로도 유일해지지 않는다(최대 중복 2 이상). 따라서 각 run 의
//   '폭'을 스트라이프 단위로 양자화해 (색, 폭) 쌍으로 조회한다.
//   인접 동일색을 금지한 B'(3,6) 패턴이면 모든 run 폭이 1 이라 이 단계가
//   자동으로 단순해진다 — 같은 코드로 두 패턴을 모두 처리한다.
//
// 구현은 DeBruijnReconstruction.cpp 에 분리되어 있다.

#include "StructuredLight/DeBruijnPattern.h"

#include <optional>
#include <vector>

namespace sl
{

    struct DeBruijnReconConfig
    {
        // white 대비 신호가 이보다 약하면 그 픽셀을 버린다. G·B 반사율이 낮은
        // 표면(검정, 순수 빨강)은 여기서 걸러진다 — 틀리는 것보다 버리는 게 낫다.
        float minWhiteLevel = 0.06f;

        // 최근접 팔레트와 차순위 팔레트의 거리 차가 이보다 작으면 애매한 픽셀로 본다.
        float minLabelMargin = 0.10f;

        int medianLabelKsize = 5;  // 라벨 1D median 커널
        int minSegmentPixels = 2;  // 이보다 짧은 세그먼트는 노이즈로 간주
        int minRunsForDecode = 0;  // 0 이면 pattern.runWindow() 를 쓴다

        // run 폭을 스트라이프 단위로 양자화할 때 허용 오차. 반올림 결과와의 상대
        // 오차가 이보다 크면 그 run 은 신뢰하지 않는다.
        float widthTolerance = 0.30f;

        // 이웃 윈도우가 연속된 run 인덱스를 가리켜야 채택한다. De Bruijn 수열은
        // 코드워드 간 Hamming 거리가 1 이라 이 검사가 없으면 오분류 하나가
        // 그대로 틀린 3D 점이 된다 (docs/DeBruijn 6.1절).
        int contextRadius = 2;

        float minTriangulationAngleDeg = 0.25f;
        std::optional<float> maxCameraDistance;
    };

    // 입력 프레임. 각 ImageVec3 의 (x,y,z) = (R,G,B), 0..255 범위.
    struct DeBruijnFrames
    {
        ImageVec3 color; // 패턴 투사 캡처
        ImageVec3 white; // all-white 레퍼런스 (알베도·음영 소거 + 포인트 컬러)
    };

    struct DeBruijnStats
    {
        int rows = 0;                // 처리한 행
        int segments = 0;            // 검출한 세그먼트
        int decodedSegments = 0;     // 절대 인덱스를 얻은 세그먼트
        int boundaries = 0;          // sub-pixel 로 정련한 경계
        int triangulated = 0;        // 3D 점이 된 경계
        int rejectedByContext = 0;   // 문맥 일관성에서 탈락
        int rejectedByWidth = 0;     // 폭 양자화 실패
    };

    struct DeBruijnResult
    {
        PointCloud points;                   // 유효 3D 점 (경계 단위, sparse)
        std::vector<Eigen::Vector3i> colors; // points 와 1:1, white 프레임의 0..255 RGB
        DeBruijnStats stats;
    };

    class DeBruijnReconstruction
    {
    public:
        DeBruijnReconstruction(DeBruijnReconConfig config, CameraCalibration camera,
                               CameraCalibration projector, DeBruijnPattern pattern);

        const DeBruijnReconConfig &config() const { return config_; }
        const DeBruijnPattern &pattern() const { return pattern_; }

        DeBruijnResult run(const DeBruijnFrames &frames) const;

    private:
        // 한 행에서 검출한 세그먼트.
        struct Segment
        {
            int label = -1;      // 색 심볼
            int begin = 0;       // 시작 픽셀 (포함)
            int end = 0;         // 끝 픽셀 (미포함)
            int units = 0;       // 스트라이프 단위 폭 (양자화 결과, 실패 시 0)
            int startStripe = -1;// 절대 스트라이프 인덱스 (미디코딩 시 -1)
        };

        // white 로 정규화한 색 비율. white 가 약한 픽셀은 valid=false.
        void computeRatioRow(const DeBruijnFrames &frames, Eigen::Index row,
                             std::vector<Eigen::Vector3f> &ratio, std::vector<bool> &valid) const;

        // 팔레트 최근접 분류 + 차순위와의 여유 검사.
        std::vector<int> classifyRow(const std::vector<Eigen::Vector3f> &ratio,
                                     const std::vector<bool> &valid) const;

        // 라벨 시퀀스를 run 으로 묶고 짧은 것을 버린다.
        std::vector<Segment> segmentRow(const std::vector<int> &labels) const;

        // 세그먼트 폭에서 단위 스트라이프 폭(픽셀)을 추정하고 각 폭을 양자화한다.
        // 폭 1 인 run 이 다수라는 성질을 이용해 하위 분위수를 단위로 잡는다.
        bool quantizeWidths(std::vector<Segment> &segments, DeBruijnStats &stats) const;

        // (색, 폭) 윈도우를 조회해 절대 스트라이프 인덱스를 채우고 문맥 검사를 건다.
        void decodeSegments(std::vector<Segment> &segments, DeBruijnStats &stats) const;

        // 두 세그먼트 사이 경계의 sub-pixel 카메라 x. 실패하면 음수.
        double refineBoundary(const std::vector<Eigen::Vector3f> &ratio, const std::vector<bool> &valid,
                              const Segment &left, const Segment &right) const;

        // 카메라 레이 ∩ 프로젝터 광평면. ColorCoded::triangulate 와 동일한 기하.
        bool triangulate(double cameraX, double cameraY, double uProjector,
                         Eigen::Vector3d &point) const;

        DeBruijnReconConfig config_;
        CameraCalibration camera_;
        CameraCalibration projector_;
        DeBruijnPattern pattern_;
        std::vector<Eigen::Vector3f> palette_; // 정규화된 [0,1] RGB 캐시
    };

} // namespace sl
