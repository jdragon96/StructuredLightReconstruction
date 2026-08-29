// DeBruijnReconTest.cpp
//
// DeBruijnReconstruction 의 합성 검증. 정답을 아는 장면을 직접 렌더해서
// 복원된 3D 점이 그 장면과 일치하는지 확인한다.
//
// 장면: 정면 평면(깊이 Z0) + 구형 융기. 표면에는 유채색 알베도 텍스처를 입혀
//       white 정규화가 실제로 알베도를 소거하는지 함께 본다.
// 기하: rectified 스테레오 — 카메라 원점, 프로젝터 (B,0,0), 회전 없음.
//
// 사용:
//   ./DeBruijnReconTest                 # B(3,4) 채택안
//   ./DeBruijnReconTest --no-repeat     # B'(3,6) 대안 (인접 동일색 금지)
//   ./DeBruijnReconTest --noise 8 --textured 0

#include "StructuredLight/DeBruijnReconstruction.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace
{

    constexpr int kWidth = 960;
    constexpr int kHeight = 240;
    constexpr double kFocal = 900.0;
    constexpr double kBaseline = 120.0; // world 단위
    constexpr double kDepth = 800.0;

    sl::CameraCalibration makeCamera(double centerX)
    {
        sl::CameraCalibration cal;
        cal.imageSize = Eigen::Vector2i(kWidth, kHeight);
        cal.intrinsic << kFocal, 0.0, kWidth / 2.0,
            0.0, kFocal, kHeight / 2.0,
            0.0, 0.0, 1.0;
        cal.rotation = Eigen::Vector3d::Zero();          // world -> camera 회전 없음
        cal.translation = Eigen::Vector3d(centerX, 0, 0); // 광학중심의 world 좌표
        return cal;
    }

    // 장면 깊이 — 평면 + 가우시안 융기(시차를 만들어 디코딩을 실제로 시험한다).
    double sceneDepth(int x, int y)
    {
        const double dx = (x - kWidth * 0.62) / (kWidth * 0.20);
        const double dy = (y - kHeight * 0.5) / (kHeight * 0.45);
        return kDepth - 90.0 * std::exp(-(dx * dx + dy * dy));
    }

    Eigen::Vector3f albedoAt(int x, int y, bool textured)
    {
        if (!textured)
            return Eigen::Vector3f(0.8f, 0.8f, 0.8f);
        // 저주파 유채색 얼룩 + 채도 높은 패치 몇 개
        const double fx = 2.0 * M_PI * x / (kWidth / 2.3);
        const double fy = 2.0 * M_PI * y / (kHeight / 1.7);
        Eigen::Vector3f a(0.55f + 0.30f * static_cast<float>(std::sin(fx)),
                          0.55f + 0.30f * static_cast<float>(std::sin(fx * 0.7 + fy)),
                          0.55f + 0.30f * static_cast<float>(std::cos(fy * 1.3)));
        if (x > kWidth * 0.15 && x < kWidth * 0.28)
            a = Eigen::Vector3f(0.85f, 0.30f, 0.25f); // 붉은 패치
        if (x > kWidth * 0.70 && x < kWidth * 0.80 && y > kHeight * 0.4)
            a = Eigen::Vector3f(0.30f, 0.75f, 0.70f); // 청록 패치
        return a.cwiseMax(0.05f).cwiseMin(1.0f);
    }

} // namespace

int main(int argc, char **argv)
{
    sl::DeBruijnConfig patternConfig;
    double noiseSigma = 3.0; // 0~255 스케일
    bool textured = true;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--no-repeat")
        {
            patternConfig.noRepeat = true;
            patternConfig.window = 6;
            patternConfig.offset = 50;
        }
        else if (arg == "--noise" && i + 1 < argc)
            noiseSigma = std::stod(argv[++i]);
        else if (arg == "--textured" && i + 1 < argc)
            textured = std::stoi(argv[++i]) != 0;
    }
    patternConfig.imageWidth = kWidth;
    patternConfig.imageHeight = kHeight;

    try
    {
        const sl::DeBruijnPattern pattern(patternConfig);
        std::cout << "패턴: " << (patternConfig.noRepeat ? "B'" : "B") << "(" << patternConfig.alphabet
                  << "," << patternConfig.window << ") offset " << patternConfig.offset << "\n"
                  << "  " << pattern.report().summary() << "\n"
                  << "  run " << pattern.runs().size() << "개, run 윈도우 " << pattern.runWindow()
                  << " (색+폭)\n\n";

        const sl::CameraCalibration camera = makeCamera(0.0);
        const sl::CameraCalibration projector = makeCamera(kBaseline);

        // ── 합성 촬영 ────────────────────────────────────────────────────────
        const int stripePx = kWidth / patternConfig.length;
        const int x0 = (kWidth - stripePx * patternConfig.length) / 2;

        sl::DeBruijnFrames frames;
        for (sl::ImageVec3 *img : {&frames.color, &frames.white})
        {
            img->x = sl::Image::Zero(kHeight, kWidth);
            img->y = sl::Image::Zero(kHeight, kWidth);
            img->z = sl::Image::Zero(kHeight, kWidth);
        }

        std::mt19937 rng(7);
        std::normal_distribution<double> noise(0.0, noiseSigma);
        std::vector<double> truthDepth(static_cast<size_t>(kHeight) * kWidth, 0.0);

        for (int y = 0; y < kHeight; ++y)
        {
            for (int x = 0; x < kWidth; ++x)
            {
                const double Z = sceneDepth(x, y);
                const double X = (x - kWidth / 2.0) / kFocal * Z;
                truthDepth[static_cast<size_t>(y) * kWidth + x] = Z;

                // 프로젝터에 투영해 어느 스트라이프가 비치는지 구한다.
                const Eigen::Vector3f albedo = albedoAt(x, y, textured);
                const float shading = static_cast<float>(0.55 + 0.35 * (1.0 - static_cast<double>(y) / kHeight));

                // 센서는 픽셀 면적에 걸쳐 빛을 적분한다. 계단으로 렌더하면 경계가
                // 항상 픽셀 중앙에 놓여 sub-pixel 추정을 과소평가하게 되므로,
                // 픽셀 폭을 가로질러 초과표본해 평균한다.
                constexpr int kSubSamples = 16;
                Eigen::Vector3f pattern255(0, 0, 0);
                for (int s = 0; s < kSubSamples; ++s)
                {
                    const double sx = x + (s + 0.5) / kSubSamples - 0.5;
                    const double Xs = (sx - kWidth / 2.0) / kFocal * Z;
                    const double xps = kFocal * (Xs - kBaseline) / Z + kWidth / 2.0;
                    const int stripeS = static_cast<int>(std::floor((xps - x0) / stripePx));
                    if (stripeS >= 0 && stripeS < patternConfig.length)
                    {
                        pattern255 += patternConfig.palette[static_cast<size_t>(
                                          pattern.code()[static_cast<size_t>(stripeS)])].cast<float>();
                    }
                }
                pattern255 /= static_cast<float>(kSubSamples);

                const auto shoot = [&](const Eigen::Vector3f &projected) {
                    return Eigen::Vector3f(
                        std::clamp(albedo.x() * shading * projected.x() + static_cast<float>(noise(rng)), 0.0f, 255.0f),
                        std::clamp(albedo.y() * shading * projected.y() + static_cast<float>(noise(rng)), 0.0f, 255.0f),
                        std::clamp(albedo.z() * shading * projected.z() + static_cast<float>(noise(rng)), 0.0f, 255.0f));
                };

                const Eigen::Vector3f c = shoot(pattern255);
                const Eigen::Vector3f w = shoot(Eigen::Vector3f(255, 255, 255));
                frames.color.x(y, x) = c.x(); frames.color.y(y, x) = c.y(); frames.color.z(y, x) = c.z();
                frames.white.x(y, x) = w.x(); frames.white.y(y, x) = w.y(); frames.white.z(y, x) = w.z();
            }
        }

        // ── 복원 ─────────────────────────────────────────────────────────────
        sl::DeBruijnReconConfig reconConfig;
        const sl::DeBruijnReconstruction recon(reconConfig, camera, projector, pattern);
        const sl::DeBruijnResult result = recon.run(frames);
        const sl::DeBruijnStats &st = result.stats;

        std::cout << "촬영: " << kHeight << "x" << kWidth << ", 텍스처 "
                  << (textured ? "있음" : "없음") << ", 노이즈 σ=" << noiseSigma << "\n";
        std::cout << "세그먼트 " << st.segments << " -> 디코딩 " << st.decodedSegments
                  << " (문맥 탈락 " << st.rejectedByContext << ", 폭 실패 " << st.rejectedByWidth << ")\n";
        std::cout << "경계 " << st.boundaries << " -> 3D 점 " << st.triangulated
                  << "  (행당 " << (st.rows ? st.triangulated / st.rows : 0) << "개)\n\n";

        if (result.points.empty())
        {
            std::cout << "3D 점이 하나도 나오지 않았다.\n";
            return 1;
        }

        // ── 정답 대조 ────────────────────────────────────────────────────────
        std::vector<double> errors;
        errors.reserve(result.points.size());
        for (const Eigen::Vector3f &p : result.points)
        {
            // 복원된 점을 카메라에 재투영해 그 위치의 정답 깊이와 비교한다.
            const double px = kFocal * p.x() / p.z() + kWidth / 2.0;
            const double py = kFocal * p.y() / p.z() + kHeight / 2.0;
            const int ix = static_cast<int>(std::lround(px));
            const int iy = static_cast<int>(std::lround(py));
            if (ix < 0 || ix >= kWidth || iy < 0 || iy >= kHeight)
                continue;
            errors.push_back(std::abs(static_cast<double>(p.z()) - sceneDepth(ix, iy)));
        }
        std::sort(errors.begin(), errors.end());
        const double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
        const auto pct = [&](double q) { return errors[static_cast<size_t>(q * (errors.size() - 1))]; };
        const int gross = static_cast<int>(std::count_if(errors.begin(), errors.end(),
                                                         [](double e) { return e > 5.0; }));

        std::cout << "깊이 오차 (world 단위, 장면 깊이 ~" << kDepth << ")\n";
        std::cout << "  평균 " << mean << "   중앙값 " << pct(0.5) << "   p90 " << pct(0.9)
                  << "   최대 " << errors.back() << "\n";
        std::cout << "  총오차(>5) " << gross << " / " << errors.size() << "  ("
                  << (100.0 * gross / static_cast<double>(errors.size())) << "%)\n";

        const bool pass = pct(0.5) < 1.0 && (100.0 * gross / static_cast<double>(errors.size())) < 2.0;
        std::cout << "\n" << (pass ? "PASS" : "FAIL")
                  << " — 중앙 오차 < 1.0 이고 총오차 < 2% 이어야 한다\n";
        return pass ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "오류: " << error.what() << "\n";
        return 1;
    }
}
