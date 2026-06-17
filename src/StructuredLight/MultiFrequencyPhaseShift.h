#pragma once

// MultiFrequencyPhaseShift.h
//
// Gray Code + Multi-Frequency Phase Shift 3D reconstruction.
// C++/Eigen port of python/MultiFrequencyPhaseShift.py.
//
// 설계 의도:
//  - 모든 픽셀 단위 연산은 Eigen::Array(ArrayXXf)로 표현해 SIMD 친화적이며,
//    각 처리 단계는 인터페이스(Strategy)로 분리되어 있다.
//  - CPU 구현(Cpu*)은 기본 전략이자, 추후 Compute Shader 백엔드(Gpu*)의
//    정합성 비교 기준(oracle) 역할을 한다.
//  - GPU 백엔드는 동일 인터페이스를 구현하면서 Image를 텍스처/SSBO로 치환하면
//    MultiFrequencyPhaseShift 오케스트레이션 코드는 변경 없이 그대로 재사용된다.
//
// 구현은 MultiFrequencyPhaseShift.cpp 에 분리되어 있다.

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <map>
#include <memory>
#include <vector>

namespace sl
{

    // ── 기본 타입 ─────────────────────────────────────────────────────────────

    using Image = Eigen::ArrayXXf;
    using ValidMask = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>;

    constexpr float kTwoPi = 6.28318530717958647692f;

    // 픽셀당 3D 벡터 필드. Compute Shader 환경에서는 3개의 텍스처/SSBO로 대응된다.
    struct ImageVec3
    {
        Image x, y, z;

        static ImageVec3 constant(Eigen::Index rows, Eigen::Index cols, const Eigen::Vector3f &v);

        Image dot(const ImageVec3 &rhs) const;

        ImageVec3 operator+(const ImageVec3 &rhs) const;
        ImageVec3 operator*(const Image &s) const;
    };

    using PointCloud = std::vector<Eigen::Vector3f>;

    // ── 캘리브레이션 ──────────────────────────────────────────────────────────

    struct CameraCalibration
    {
        Eigen::Vector2i imageSize = Eigen::Vector2i::Zero(); // (width, height)
        Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Identity();
        Eigen::VectorXd distortion;                            // planar 투영 모델에서는 미사용
        Eigen::Vector3d rotation = Eigen::Vector3d::Zero();    // Rodrigues, world -> camera
        Eigen::Vector3d translation = Eigen::Vector3d::Zero(); // 광학중심의 world 좌표

        // rotate(world->cam, Rodrigues) -> R_world->cam -> R_cam->world (transpose)
        Eigen::Matrix3d cameraToWorldRotation() const;

        const Eigen::Vector3d &center() const { return translation; }
    };

    // ── 설정 ─────────────────────────────────────────────────────────────────

    struct MultiFreqConfig
    {
        int nPhaseSteps = 4;
        int nGrayBits = 4;

        // 오름차순. frequencies.front() == 2^nGrayBits 이어야 한다.
        std::vector<int> frequencies = {16, 32, 64};

        float grayModulationThreshold = 20.0f; // white-black 최소 명암차
        float phaseModulationThreshold = 8.0f; // I_max-I_min 최소값 (4-step 합)
        int medianKernelSize = 3;              // 고립된 fringe-order 오류 제거
    };

    // ── 단계별 결과 타입 ───────────────────────────────────────────────────────

    struct GrayCodeResult
    {
        Image index; // stripe index k (median filtering을 위해 float로 보관)
        ValidMask valid;
    };

    struct ProjectorCorrespondence
    {
        Image uProj; // 정규화 projector x 좌표, [0, 1)
        ValidMask valid;
    };

    struct TriangulationResult
    {
        ImageVec3 points; // world 좌표, 픽셀별 dense
        ValidMask valid;
    };

    struct GrayCodeFrames
    {
        std::vector<Image> bitPlanes; // MSB first
        Image white;
        Image black;
    };

    // 주파수별 4-step phase shift 캡처
    using PhaseFrequencyFrames = std::map<int, std::array<Image, 4>>;

    struct ReconstructionResult
    {
        ProjectorCorrespondence correspondence;
        TriangulationResult triangulation;

        PointCloud toPointCloud() const;
    };

    // ── 순수 연산 유틸리티 ─────────────────────────────────────────────────────

    namespace detail
    {

        // 3x3 median filter. 테두리 픽셀은 원본 값을 유지한다.
        Image median3(const Image &src);

        // 4-step PSP wrapped phase.
        // I_step(x) = 127.5*(1 + cos(phi - 2*pi*step/4)), phi = 2*pi*freq*x/W
        // a = I0-I2 = 255*cos(phi), b = I1-I3 = 255*sin(phi)
        // => atan2(b, a) = phi  (별도 위상 오프셋 보정 불필요)
        Image wrappedPhase(const std::array<Image, 4> &steps);

        Image modulation(const std::array<Image, 4> &steps);

        // Gray code 비트 평면(MSB first) 디코딩 -> stripe index k = floor(x*2^nBits/W).
        // 임계값은 white/black 레퍼런스의 픽셀별 중간값을 사용한다(셰이딩에 무관).
        GrayCodeResult decodeGrayCode(const std::vector<Image> &bitPlanes, const Image &white,
                                      const Image &black, int nBits, float modulationThreshold);

        // coarseUnwrapped*ratio 를 기준으로 fineWrapped 의 2*pi 배수(fringe order)를 결정한다.
        // 경계 픽셀의 ±1 오차가 라인 단위로 번지는 것을 median filter로 억제한다.
        Image unwrapAgainstCarrier(const Image &fineWrapped, const Image &coarseUnwrapped,
                                   double ratio, int medianKernelSize);

    } // namespace detail

    // ── 파이프라인 단계 인터페이스 (Strategy 패턴) ───────────────────────────────
    //
    // 각 단계는 순수 인터페이스로 분리되어 있어, 추후 Compute Shader 백엔드가
    // 동일 인터페이스를 구현하면 MultiFrequencyPhaseShift 오케스트레이션은
    // 변경 없이 GPU 경로로 교체될 수 있다 (의존성 주입).

    class IGrayCodeDecoder
    {
    public:
        virtual ~IGrayCodeDecoder() = default;
        virtual GrayCodeResult decode(const std::vector<Image> &bitPlanes, const Image &white,
                                      const Image &black, int nBits, float modulationThreshold) const = 0;
    };

    class IPhaseAnalyzer
    {
    public:
        virtual ~IPhaseAnalyzer() = default;
        virtual Image wrappedPhase(const std::array<Image, 4> &steps) const = 0;
        virtual Image modulation(const std::array<Image, 4> &steps) const = 0;
    };

    class IPhaseUnwrapper
    {
    public:
        virtual ~IPhaseUnwrapper() = default;
        virtual Image unwrap(const Image &fineWrapped, const Image &coarseUnwrapped,
                             double ratio, int medianKernelSize) const = 0;
    };

    class ITriangulator
    {
    public:
        virtual ~ITriangulator() = default;
        virtual TriangulationResult triangulate(const ProjectorCorrespondence &correspondence,
                                                const CameraCalibration &camera,
                                                const CameraCalibration &projector) const = 0;
    };

    // ── CPU 기본 구현 ────────────────────────────────────────────────────────

    class CpuGrayCodeDecoder final : public IGrayCodeDecoder
    {
    public:
        GrayCodeResult decode(const std::vector<Image> &bitPlanes, const Image &white, const Image &black,
                              int nBits, float modulationThreshold) const override;
    };

    class CpuPhaseAnalyzer final : public IPhaseAnalyzer
    {
    public:
        Image wrappedPhase(const std::array<Image, 4> &steps) const override;
        Image modulation(const std::array<Image, 4> &steps) const override;
    };

    class CpuCascadeUnwrapper final : public IPhaseUnwrapper
    {
    public:
        Image unwrap(const Image &fineWrapped, const Image &coarseUnwrapped,
                     double ratio, int medianKernelSize) const override;
    };

    // 카메라 레이 ∩ 프로젝터 평면(C_p를 지나며 stripe 방향을 포함) 삼각측량.
    class CpuPlaneTriangulator final : public ITriangulator
    {
    public:
        TriangulationResult triangulate(const ProjectorCorrespondence &corr, const CameraCalibration &camera,
                                        const CameraCalibration &projector) const override;
    };

    // computeProjectorCoordinate()/run() 호출 시 채워지는 중간 연산 결과.
    // ExportDebugFile()이 이 값들을 grayscale PGM 이미지로 저장한다.
    struct DebugData
    {
        Image grayIndex; // Gray code stripe index (median filter 적용 후)
        Image absPhase;  // 누적 unwrap된 절대 위상 [0, 2*pi*topFreq)
        Image uProj;     // 정규화 projector x 좌표 [0, 1)
        ValidMask valid; // computeProjectorCoordinate()의 최종 유효성

        std::map<int, Image> wrappedPhase; // 주파수별 wrapped phase [0, 2*pi)
        std::map<int, Image> modulation;   // 주파수별 modulation (I_max - I_min)

        bool hasTriangulation = false;
        Image depth;                  // world Z 좌표
        ValidMask triangulationValid; // run()의 최종 유효성
    };

    // ── 메인 클래스 ───────────────────────────────────────────────────────────

    class MultiFrequencyPhaseShift
    {
    public:
        MultiFrequencyPhaseShift(MultiFreqConfig config, CameraCalibration camera, CameraCalibration projector,
                                 std::shared_ptr<IGrayCodeDecoder> grayDecoder = std::make_shared<CpuGrayCodeDecoder>(),
                                 std::shared_ptr<IPhaseAnalyzer> phaseAnalyzer = std::make_shared<CpuPhaseAnalyzer>(),
                                 std::shared_ptr<IPhaseUnwrapper> unwrapper = std::make_shared<CpuCascadeUnwrapper>(),
                                 std::shared_ptr<ITriangulator> triangulator = std::make_shared<CpuPlaneTriangulator>());

        const MultiFreqConfig &config() const { return config_; }
        const CameraCalibration &camera() const { return camera_; }
        const CameraCalibration &projector() const { return projector_; }

        // Gray code + 다중 주파수 위상으로부터 절대 위상 -> 정규화 projector x 좌표.
        ProjectorCorrespondence computeProjectorCoordinate(const GrayCodeFrames &gray,
                                                           const PhaseFrequencyFrames &freqFrames) const;

        ReconstructionResult run(const GrayCodeFrames &gray, const PhaseFrequencyFrames &freqFrames) const;

        // computeProjectorCoordinate()/run() 이후 호출하면, 그 과정에서 기록된 중간
        // 결과(Gray code index, 주파수별 wrapped phase/modulation, 절대 위상,
        // projector 좌표, 유효성 마스크, 깊이맵)를 outputDir 아래 *.pgm 파일로 저장한다.
        void ExportDebugFile(const std::string &outputDir) const;

    private:
        MultiFreqConfig config_;
        CameraCalibration camera_;
        CameraCalibration projector_;

        std::shared_ptr<IGrayCodeDecoder> grayDecoder_;
        std::shared_ptr<IPhaseAnalyzer> phaseAnalyzer_;
        std::shared_ptr<IPhaseUnwrapper> unwrapper_;
        std::shared_ptr<ITriangulator> triangulator_;

        mutable DebugData debug_;
    };

} // namespace sl
