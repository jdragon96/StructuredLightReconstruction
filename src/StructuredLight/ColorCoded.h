#pragma once

// ColorCoded.h
//
// Color stripe + RGB phase 3D reconstruction.
// C++/Eigen port of python/ColorCoded.py.
//
// 입력 데이터:
//  - color: 컬러 스트라이프 코드 이미지 (절대 coarse projector index 제공)
//  - phase: 3-step RGB phase shift 이미지 (sub-pixel 좌표 제공, wrapped phase만 포함)
//  - white: all-white 레퍼런스 (color ratio 정규화 + 포인트 컬러)
//
// wrapped phase 단독으로는 frequency개의 fringe order 후보가 존재하므로,
// 행(row) 단위 Viterbi-style DP로 color code와 phase를 결합해 절대 fringe
// order를 복원한 뒤, 카메라 레이 ∩ 프로젝터 평면 삼각측량으로 3D 점을 얻는다.
//
// MultiFrequencyPhaseShift.h 의 공용 타입(Image, ImageVec3, ValidMask, PointCloud,
// CameraCalibration, kTwoPi)을 재사용한다.
//
// 구현은 ColorCoded.cpp 에 분리되어 있다.

#include "StructuredLight/MultiFrequencyPhaseShift.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sl
{

    // dlp_addon.py 가 내보낸 color_pattern_info.json 을 표현한다.
    // self_equalizing 모드의 물리 시퀀스/팔레트 확장은 create()에서 처리된다.
    struct ColorCodeInfo
    {
        std::string mode; // "hamming" 또는 그 외(팔레트 최근접)
        int decodeWindow = 0;
        int codeLength = 0;

        std::vector<int> sequence;            // 물리 stripe 시퀀스 (palette index)
        std::vector<Eigen::Vector3f> palette; // 정규화된 [0,1] RGB

        // decode_window 길이의 연속 시퀀스 -> 시작 인덱스. 앵커 윈도우 매칭에 사용.
        std::map<std::vector<int>, int> windowToStart;

        // logicalSequence/logicalPalette255 는 color_pattern_info.json 의
        // "sequence"/"palette"(0~255) 그대로.
        static ColorCodeInfo create(const std::vector<int> &logicalSequence,
                                    const std::vector<Eigen::Vector3i> &logicalPalette255,
                                    const std::string &mode, bool selfEqualizing, int decodeWindow);
    };

    // color_phase_info.json
    struct PhaseInfo
    {
        int frequency = 0;
        Eigen::Matrix3d responseMatrix = Eigen::Matrix3d::Identity();
        bool projectorPrecompensation = false;
    };

    struct ColorCodedConfig
    {
        float phaseModulationThreshold = 8.0f; // sqrt(denom^2+numer^2)/3 최소값
        float hammingOnThreshold = 0.80f;      // hamming 모드 채널 on/off 임계값
        int minSegmentPixels = 8;              // 유효 phase 연속 구간 최소 길이
        int minRunPixels = 3;                  // 앵커 판정 시 run-length 최소 길이
        int minAnchorWindows = 1;              // 구간 디코딩에 필요한 최소 앵커 수
        int medianLabelKsize = 5;              // 색상 라벨 1D median 커널

        int localOrderJump = 4;               // DP에서 허용하는 fringe-order 점프 범위
        float expectedProjectorStepPx = 1.0f; // 픽셀당 기대 프로젝터 이동량
        float smoothnessWeight = 0.12f;
        float smoothnessTruncationPx = 5.0f;
        float discontinuityPenalty = 7.0f;
        bool bidirectionalConsensus = true; // 정-역방향 DP 일치만 채택

        float minTriangulationAngleDeg = 0.25f; // 레이-평면 최소 교차각
        std::optional<float> maxCameraDistance; // 카메라로부터 최대 거리(옵션)
    };

    // 입력 프레임. 각 ImageVec3의 (x,y,z) = (R,G,B), 0..255 범위.
    struct ColorCodedFrames
    {
        ImageVec3 color; // 컬러 스트라이프 코드
        ImageVec3 white; // all-white 레퍼런스
        ImageVec3 phase; // 3-step RGB phase
    };

    struct ColorCodedResult
    {
        Image uProjector; // 정규화 projector x 좌표 (무효 픽셀은 NaN)
        ValidMask valid;

        PointCloud points;                   // 유효 3D 점
        std::vector<Eigen::Vector3i> colors; // points와 1:1, 0..255 RGB
    };

    class ColorCodedReconstruction
    {
    public:
        ColorCodedReconstruction(ColorCodedConfig config, CameraCalibration camera, CameraCalibration projector,
                                 ColorCodeInfo colorInfo, PhaseInfo phaseInfo);

        const ColorCodedConfig &config() const { return config_; }
        const CameraCalibration &camera() const { return camera_; }
        const CameraCalibration &projector() const { return projector_; }
        const ColorCodeInfo &colorInfo() const { return colorInfo_; }
        const PhaseInfo &phaseInfo() const { return phaseInfo_; }

        ColorCodedResult run(const ColorCodedFrames &frames) const;

        // run() 또는 computeProjectorCoordinate()가 채운 중간 결과를 디버그 PNG로 저장한다.
        void ExportDebugFile(const std::string &outputDir) const;

    private:
        struct PhaseData
        {
            Image wrapped;
            Image modulation;
            ValidMask valid;
        };

        struct ProjectorCoordinate
        {
            Image uProjector;
            ValidMask valid;
        };

        struct FringeOrderResult
        {
            Eigen::ArrayXXi orders; // -1 = 미디코딩
            ValidMask valid;
        };

        static ImageVec3 computeColorRatio(const ImageVec3 &color, const ImageVec3 &white);

        // R/G/B 3-step phase -> wrapped phase + modulation + 유효성.
        PhaseData computePhase(const ImageVec3 &phaseImage) const;

        // color ratio -> 색상 코드 라벨 (hamming 비트 조합 또는 팔레트 최근접).
        Eigen::ArrayXXi classifyColorCode(const ImageVec3 &colorRatio) const;

        // 1D median-filtered label 시퀀스에서 windowToStart에 매칭되는
        // 고유 윈도우 개수(앵커)를 센다.
        int countAnchorWindows(const std::vector<int> &segmentLabels) const;

        // 주어진 phase fraction/color label에 대한 fringe-order별 emission cost.
        std::vector<float> emissionCost(float phaseFraction, int observedLabel,
                                        const Eigen::Vector3f &observedRatio) const;

        // 한 row 구간에 대한 Viterbi-style DP로 fringe order 시퀀스를 복원한다.
        std::vector<int16_t> decodeSegment(const std::vector<float> &fractions, const std::vector<int> &labels,
                                           const std::vector<Eigen::Vector3f> &ratios, float targetStepPx) const;

        // 행 단위로 유효 phase 구간을 찾아 decodeSegment(정/역방향)을 적용한다.
        FringeOrderResult decodeFringeOrder(const Image &wrapped, const ValidMask &phaseValid,
                                            const Eigen::ArrayXXi &labels, const ImageVec3 &colorRatio) const;

        // Gray code + phase -> 절대 위상 -> 정규화 projector x 좌표.
        ProjectorCoordinate computeProjectorCoordinate(const ColorCodedFrames &frames) const;

        // 카메라 레이 ∩ 프로젝터 평면 삼각측량 (정규화 레이/법선, 교차각/거리 검증).
        ColorCodedResult triangulate(const Image &uProjector, const ValidMask &valid, const ImageVec3 &whiteImage) const;

        // ExportDebugFile()에서 사용하는 파이프라인 중간 결과.
        struct DebugData
        {
            ImageVec3 colorRatio;
            Eigen::ArrayXXi labels;

            Image wrappedPhase;
            Image modulation;
            ValidMask phaseValid;

            Eigen::ArrayXXi fringeOrders;
            ValidMask fringeValid;

            Image uProjector;
            ValidMask valid;

            bool hasTriangulation = false;
            Image depth;
            ValidMask triangulationValid;
        };

        ColorCodedConfig config_;
        CameraCalibration camera_;
        CameraCalibration projector_;
        ColorCodeInfo colorInfo_;
        PhaseInfo phaseInfo_;

        mutable DebugData debug_;
    };

} // namespace sl
