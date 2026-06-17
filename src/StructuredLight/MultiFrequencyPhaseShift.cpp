#include "StructuredLight/MultiFrequencyPhaseShift.h"

#include "StructuredLight/DebugImageExport.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <memory>

namespace sl
{

    using detail::writeImagePNG;
    using detail::writeImagePNGAuto;
    using detail::writeMaskPNG;

    // ── CameraCalibration ───────────────────────────────────────────────────

    Eigen::Matrix3d CameraCalibration::cameraToWorldRotation() const
    {
        const double theta = rotation.norm();
        if (theta < 1e-12)
            return Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d worldToCamera =
            Eigen::AngleAxisd(theta, rotation / theta).toRotationMatrix();
        return worldToCamera.transpose();
    }

    // ── ImageVec3 ───────────────────────────────────────────────────────────

    ImageVec3 ImageVec3::constant(Eigen::Index rows, Eigen::Index cols, const Eigen::Vector3f &v)
    {
        return {Image::Constant(rows, cols, v.x()),
                Image::Constant(rows, cols, v.y()),
                Image::Constant(rows, cols, v.z())};
    }

    Image ImageVec3::dot(const ImageVec3 &rhs) const
    {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    ImageVec3 ImageVec3::operator+(const ImageVec3 &rhs) const
    {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }

    ImageVec3 ImageVec3::operator*(const Image &s) const
    {
        return {x * s, y * s, z * s};
    }

    // ── ReconstructionResult ───────────────────────────────────────────────

    PointCloud ReconstructionResult::toPointCloud() const
    {
        PointCloud points;
        const auto &valid = triangulation.valid;
        const auto &pts = triangulation.points;
        points.reserve(static_cast<size_t>(valid.count()));
        for (Eigen::Index r = 0; r < valid.rows(); ++r)
        {
            for (Eigen::Index c = 0; c < valid.cols(); ++c)
            {
                if (valid(r, c))
                {
                    points.emplace_back(pts.x(r, c), pts.y(r, c), pts.z(r, c));
                }
            }
        }
        return points;
    }

    // ── 순수 연산 유틸리티 ─────────────────────────────────────────────────────

    namespace detail
    {

        Image median3(const Image &src)
        {
            const Eigen::Index rows = src.rows(), cols = src.cols();
            if (rows < 3 || cols < 3)
                return src;

            Image out = src;
            std::array<float, 9> window;
            for (Eigen::Index r = 1; r + 1 < rows; ++r)
            {
                for (Eigen::Index c = 1; c + 1 < cols; ++c)
                {
                    int n = 0;
                    for (Eigen::Index dr = -1; dr <= 1; ++dr)
                    {
                        for (Eigen::Index dc = -1; dc <= 1; ++dc)
                        {
                            window[n++] = src(r + dr, c + dc);
                        }
                    }
                    std::nth_element(window.begin(), window.begin() + 4, window.end());
                    out(r, c) = window[4];
                }
            }
            return out;
        }

        Image wrappedPhase(const std::array<Image, 4> &steps)
        {
            const Image a = steps[0] - steps[2]; // a: 0 - 180 (cos)
            const Image b = steps[1] - steps[3]; // b: 90 - 270 (sin)

            Image phase(a.rows(), a.cols());
            for (Eigen::Index r = 0; r < a.rows(); ++r)
            {
                for (Eigen::Index c = 0; c < a.cols(); ++c)
                {
                    float p = std::atan2(b(r, c), a(r, c));
                    if (p < 0.0f)
                        p += kTwoPi;
                    phase(r, c) = p;
                }
            }
            return phase;
        }

        Image modulation(const std::array<Image, 4> &steps)
        {
            Image maxV = steps[0];
            Image minV = steps[0];
            for (int i = 1; i < 4; ++i)
            {
                maxV = maxV.max(steps[i]);
                minV = minV.min(steps[i]);
            }
            return maxV - minV;
        }

        GrayCodeResult decodeGrayCode(const std::vector<Image> &bitPlanes, const Image &white,
                                      const Image &black, int nBits, float modulationThreshold)
        {
            const Eigen::Index rows = white.rows(), cols = white.cols();
            const Image threshold = (white + black) * 0.5f;
            const ValidMask valid = (white - black) > modulationThreshold;

            Image index(rows, cols);
            for (Eigen::Index r = 0; r < rows; ++r)
            {
                for (Eigen::Index c = 0; c < cols; ++c)
                {
                    int prev = bitPlanes[0](r, c) > threshold(r, c) ? 1 : 0;
                    int k = prev;
                    for (int bit = 1; bit < nBits; ++bit)
                    {
                        const int cur = (bitPlanes[bit](r, c) > threshold(r, c) ? 1 : 0) ^ prev;
                        k = (k << 1) | cur;
                        prev = cur;
                    }
                    index(r, c) = static_cast<float>(k);
                }
            }
            return {index, valid};
        }

        Image unwrapAgainstCarrier(const Image &fineWrapped, const Image &coarseUnwrapped,
                                   double ratio, int medianKernelSize)
        {
            const Image expected = coarseUnwrapped * static_cast<float>(ratio);
            Image order = ((expected - fineWrapped) * (1.0f / kTwoPi))
                              .unaryExpr([](float v)
                                         { return std::round(v); });
            if (medianKernelSize > 1)
                order = median3(order);
            return fineWrapped + kTwoPi * order;
        }

    } // namespace detail

    // ── CPU 기본 구현 ────────────────────────────────────────────────────────

    GrayCodeResult CpuGrayCodeDecoder::decode(const std::vector<Image> &bitPlanes, const Image &white,
                                              const Image &black, int nBits, float modulationThreshold) const
    {
        return detail::decodeGrayCode(bitPlanes, white, black, nBits, modulationThreshold);
    }

    Image CpuPhaseAnalyzer::wrappedPhase(const std::array<Image, 4> &steps) const
    {
        return detail::wrappedPhase(steps);
    }

    Image CpuPhaseAnalyzer::modulation(const std::array<Image, 4> &steps) const
    {
        return detail::modulation(steps);
    }

    Image CpuCascadeUnwrapper::unwrap(const Image &fineWrapped, const Image &coarseUnwrapped,
                                      double ratio, int medianKernelSize) const
    {
        return detail::unwrapAgainstCarrier(fineWrapped, coarseUnwrapped, ratio, medianKernelSize);
    }

    TriangulationResult CpuPlaneTriangulator::triangulate(const ProjectorCorrespondence &corr,
                                                          const CameraCalibration &camera,
                                                          const CameraCalibration &projector) const
    {
        const Eigen::Index rows = corr.uProj.rows(), cols = corr.uProj.cols();

        const float fx = static_cast<float>(camera.intrinsic(0, 0));
        const float fy = static_cast<float>(camera.intrinsic(1, 1));
        const float cx = static_cast<float>(camera.intrinsic(0, 2));
        const float cy = static_cast<float>(camera.intrinsic(1, 2));

        Image xs(rows, cols), ys(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r)
        {
            for (Eigen::Index c = 0; c < cols; ++c)
            {
                xs(r, c) = static_cast<float>(c);
                ys(r, c) = static_cast<float>(r);
            }
        }

        const Eigen::Matrix3f Rc2w = camera.cameraToWorldRotation().cast<float>();
        const ImageVec3 rayCam{(xs - cx) / fx, (ys - cy) / fy, Image::Ones(rows, cols)};
        const ImageVec3 rayWorld{
            Rc2w(0, 0) * rayCam.x + Rc2w(0, 1) * rayCam.y + Rc2w(0, 2) * rayCam.z,
            Rc2w(1, 0) * rayCam.x + Rc2w(1, 1) * rayCam.y + Rc2w(1, 2) * rayCam.z,
            Rc2w(2, 0) * rayCam.x + Rc2w(2, 1) * rayCam.y + Rc2w(2, 2) * rayCam.z,
        };

        // 프로젝터 카메라좌표계에서의 column 방향(x/z)
        const float fxp = static_cast<float>(projector.intrinsic(0, 0));
        const float cxp = static_cast<float>(projector.intrinsic(0, 2));
        const float Wp = static_cast<float>(projector.imageSize.x());
        const Image a = (corr.uProj * Wp - cxp) / fxp;

        const Eigen::Matrix3f Rp2w = projector.cameraToWorldRotation().cast<float>();
        const ImageVec3 normalWorld{
            -Rp2w(0, 0) + Rp2w(0, 2) * a,
            -Rp2w(1, 0) + Rp2w(1, 2) * a,
            -Rp2w(2, 0) + Rp2w(2, 2) * a};

        const Eigen::Vector3f camCenter = camera.center().cast<float>();
        const Eigen::Vector3f projCenter = projector.center().cast<float>();
        const Eigen::Vector3f delta = projCenter - camCenter;

        // denom == 0 이면 카메라가 보는 광선과 Projector 평면이 수직이라는 의미
        // 즉, 프로젝터와 카메라가 수직임을 의미한다.
        const Image denom = rayWorld.dot(normalWorld);
        const Image numer = delta.x() * normalWorld.x + delta.y() * normalWorld.y + delta.z() * normalWorld.z;

        constexpr float eps = 1e-6f;
        const ValidMask geometryValid = corr.valid && (denom.abs() > eps);

        Image t = Image::Zero(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r)
        {
            for (Eigen::Index c = 0; c < cols; ++c)
            {
                if (geometryValid(r, c))
                    t(r, c) = numer(r, c) / denom(r, c);
            }
        }

        TriangulationResult result;
        result.points = ImageVec3::constant(rows, cols, camCenter) + rayWorld * t;
        result.valid = geometryValid && (t > 0.0f);
        return result;
    }

    // ── MultiFrequencyPhaseShift ───────────────────────────────────────────

    MultiFrequencyPhaseShift::MultiFrequencyPhaseShift(MultiFreqConfig config, CameraCalibration camera,
                                                       CameraCalibration projector,
                                                       std::shared_ptr<IGrayCodeDecoder> grayDecoder,
                                                       std::shared_ptr<IPhaseAnalyzer> phaseAnalyzer,
                                                       std::shared_ptr<IPhaseUnwrapper> unwrapper,
                                                       std::shared_ptr<ITriangulator> triangulator)
        : config_(std::move(config)),
          camera_(std::move(camera)),
          projector_(std::move(projector)),
          grayDecoder_(std::move(grayDecoder)),
          phaseAnalyzer_(std::move(phaseAnalyzer)),
          unwrapper_(std::move(unwrapper)),
          triangulator_(std::move(triangulator))
    {
        //
    }

    ProjectorCorrespondence MultiFrequencyPhaseShift::computeProjectorCoordinate(
        const GrayCodeFrames &gray, const PhaseFrequencyFrames &freqFrames) const
    {
        std::vector<int> freqs;
        freqs.reserve(freqFrames.size());
        for (const auto &[freq, frames] : freqFrames)
            freqs.push_back(freq);
        std::sort(freqs.begin(), freqs.end());

        if (freqs.empty())
            throw std::invalid_argument("freqFrames가 비어 있습니다.");

        const int nStripes = 1 << config_.nGrayBits;
        if (nStripes != freqs.front())
        {
            throw std::invalid_argument(
                "Gray code stripe 수(2^nGrayBits) != 최저 phase 주파수. "
                "nGrayBits/frequencies 설정을 확인하세요.");
        }

        std::map<int, Image> wrapped;
        std::map<int, Image> modulation;
        for (int f : freqs)
        {
            const auto &steps = freqFrames.at(f);
            wrapped[f] = phaseAnalyzer_->wrappedPhase(steps);
            modulation[f] = phaseAnalyzer_->modulation(steps);
        }

        const GrayCodeResult grayResult = grayDecoder_->decode(
            gray.bitPlanes, gray.white, gray.black, config_.nGrayBits, config_.grayModulationThreshold);
        const Image grayIndex =
            config_.medianKernelSize > 1 ? detail::median3(grayResult.index) : grayResult.index;

        // 최저 주파수: gray_k는 정확한 stripe 인덱스이고, wrapped phase는 그 안에서
        // 0->2pi로 증가하므로 절대 위상은 단순 합으로 결정된다.
        Image absPhase = wrapped[freqs.front()] + kTwoPi * grayIndex;

        for (size_t i = 0; i + 1 < freqs.size(); ++i)
        {
            const int prevFreq = freqs[i];
            const int curFreq = freqs[i + 1];
            const double ratio = static_cast<double>(curFreq) / prevFreq;
            absPhase = unwrapper_->unwrap(wrapped[curFreq], absPhase, ratio, config_.medianKernelSize);
        }

        const int topFreq = freqs.back();

        ProjectorCorrespondence result;
        result.uProj = absPhase / (kTwoPi * static_cast<float>(topFreq));

        const ValidMask phaseValid = modulation[topFreq] > config_.phaseModulationThreshold;
        result.valid = grayResult.valid && phaseValid;

        debug_.grayIndex = grayIndex;
        debug_.absPhase = absPhase;
        // uProj: Projector의 column 하나
        debug_.uProj = result.uProj;
        debug_.valid = result.valid;
        debug_.wrappedPhase = wrapped;
        debug_.modulation = modulation;

        return result;
    }

    ReconstructionResult MultiFrequencyPhaseShift::run(const GrayCodeFrames &gray,
                                                       const PhaseFrequencyFrames &freqFrames) const
    {
        ReconstructionResult result;
        result.correspondence = computeProjectorCoordinate(gray, freqFrames);
        result.triangulation = triangulator_->triangulate(result.correspondence, camera_, projector_);

        debug_.hasTriangulation = true;
        debug_.depth = result.triangulation.points.z;
        debug_.triangulationValid = result.triangulation.valid;

        return result;
    }

    void MultiFrequencyPhaseShift::ExportDebugFile(const std::string &outputDir) const
    {
        std::filesystem::create_directories(outputDir);

        writeImagePNGAuto(outputDir + "/gray_index.png", debug_.grayIndex);
        writeImagePNGAuto(outputDir + "/abs_phase.png", debug_.absPhase);
        writeImagePNG(outputDir + "/uproj.png", debug_.uProj, 0.0f, 1.0f);
        writeMaskPNG(outputDir + "/valid.png", debug_.valid);

        for (const auto &[freq, image] : debug_.wrappedPhase)
        {
            writeImagePNG(outputDir + "/wrapped_phase_" + std::to_string(freq) + ".png", image, 0.0f, kTwoPi);
        }
        for (const auto &[freq, image] : debug_.modulation)
        {
            writeImagePNGAuto(outputDir + "/modulation_" + std::to_string(freq) + ".png", image);
        }

        if (debug_.hasTriangulation)
        {
            writeImagePNGAuto(outputDir + "/depth.png", debug_.depth);
            writeMaskPNG(outputDir + "/triangulation_valid.png", debug_.triangulationValid);
        }
    }

} // namespace sl
