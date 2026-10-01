#include "StructuredLight/DeBruijnReconstruction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sl
{

    namespace
    {

        int oddKernel(int value)
        {
            value = std::max(1, value);
            return (value % 2 == 1) ? value : value + 1;
        }

        // 유효하지 않은 라벨(-1)은 그대로 두고, 유효한 값만 median 을 취한다.
        std::vector<int> medianLabels(const std::vector<int> &labels, int ksize)
        {
            const int n = static_cast<int>(labels.size());
            const int half = oddKernel(ksize) / 2;
            std::vector<int> out(labels);
            std::vector<int> window;
            window.reserve(static_cast<size_t>(2 * half + 1));

            for (int i = 0; i < n; ++i)
            {
                if (labels[static_cast<size_t>(i)] < 0)
                    continue;
                window.clear();
                for (int k = -half; k <= half; ++k)
                {
                    const int idx = std::clamp(i + k, 0, n - 1);
                    if (labels[static_cast<size_t>(idx)] >= 0)
                        window.push_back(labels[static_cast<size_t>(idx)]);
                }
                if (window.empty())
                    continue;
                std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
                out[static_cast<size_t>(i)] = window[window.size() / 2];
            }
            return out;
        }

    } // namespace

    DeBruijnReconstruction::DeBruijnReconstruction(DeBruijnReconConfig config, CameraCalibration camera,
                                                   CameraCalibration projector, DeBruijnPattern pattern)
        : config_(std::move(config)), camera_(std::move(camera)), projector_(std::move(projector)),
          pattern_(std::move(pattern))
    {
        if (pattern_.runWindow() <= 0)
        {
            throw std::invalid_argument(
                "DeBruijnReconstruction: run 윈도우가 유일해지지 않는 패턴이다. "
                "위상 없이 컬러 스트라이프만으로는 절대 위치를 얻을 수 없다");
        }
        // 팔레트는 행마다 쓰이므로 정규화해서 캐시한다.
        const auto &raw = pattern_.config().palette;
        palette_.reserve(static_cast<size_t>(pattern_.config().alphabet));
        for (int i = 0; i < pattern_.config().alphabet; ++i)
            palette_.push_back(raw[static_cast<size_t>(i)].cast<float>() / 255.0f);
    }

    // ── 1. white 정규화 ──────────────────────────────────────────────────────

    void DeBruijnReconstruction::computeRatioRow(const DeBruijnFrames &frames, Eigen::Index row,
                                                 std::vector<Eigen::Vector3f> &ratio,
                                                 std::vector<bool> &valid) const
    {
        const Eigen::Index cols = frames.color.x.cols();
        ratio.assign(static_cast<size_t>(cols), Eigen::Vector3f::Zero());
        valid.assign(static_cast<size_t>(cols), false);

        // 관측 = 알베도 x 패턴색 x 음영, white = 알베도 x 음영 이므로
        // 비율을 취하면 알베도와 음영이 정확히 소거되고 패턴색만 남는다.
        const float floorLevel = config_.minWhiteLevel * 255.0f;
        for (Eigen::Index c = 0; c < cols; ++c)
        {
            const Eigen::Vector3f w(frames.white.x(row, c), frames.white.y(row, c), frames.white.z(row, c));
            const Eigen::Vector3f o(frames.color.x(row, c), frames.color.y(row, c), frames.color.z(row, c));

            // R 은 패턴이 쓰지 않으므로 G·B 신호만으로 유효성을 판단한다.
            if (w.y() < floorLevel || w.z() < floorLevel)
                continue;

            ratio[static_cast<size_t>(c)] =
                Eigen::Vector3f(o.x() / std::max(w.x(), 1.0f), o.y() / w.y(), o.z() / w.z());
            valid[static_cast<size_t>(c)] = true;
        }
    }

    // ── 2. 색 분류 ───────────────────────────────────────────────────────────

    std::vector<int> DeBruijnReconstruction::classifyRow(const std::vector<Eigen::Vector3f> &ratio,
                                                         const std::vector<bool> &valid) const
    {
        const std::vector<Eigen::Vector3f> &palette = palette_;
        std::vector<int> labels(ratio.size(), -1);

        for (size_t i = 0; i < ratio.size(); ++i)
        {
            if (!valid[i])
                continue;

            float best = std::numeric_limits<float>::infinity();
            float second = std::numeric_limits<float>::infinity();
            int bestIndex = -1;
            for (size_t p = 0; p < palette.size(); ++p)
            {
                const float d = (palette[p] - ratio[i]).squaredNorm();
                if (d < best)
                {
                    second = best;
                    best = d;
                    bestIndex = static_cast<int>(p);
                }
                else if (d < second)
                {
                    second = d;
                }
            }
            // 1·2순위가 붙어 있으면 판정하지 않는다. 조용한 오분류를 줄이는 게
            // 점 개수를 늘리는 것보다 중요하다.
            if (std::sqrt(second) - std::sqrt(best) < config_.minLabelMargin)
                continue;
            labels[i] = bestIndex;
        }
        return medianLabels(labels, config_.medianLabelKsize);
    }

    // ── 3. 세그먼트화 ────────────────────────────────────────────────────────

    std::vector<DeBruijnReconstruction::Segment>
    DeBruijnReconstruction::segmentRow(const std::vector<int> &labels) const
    {
        std::vector<Segment> segments;
        const int n = static_cast<int>(labels.size());
        int i = 0;
        while (i < n)
        {
            if (labels[static_cast<size_t>(i)] < 0)
            {
                ++i;
                continue;
            }
            int j = i;
            while (j + 1 < n && labels[static_cast<size_t>(j + 1)] == labels[static_cast<size_t>(i)])
                ++j;

            if (j - i + 1 >= config_.minSegmentPixels)
            {
                Segment seg;
                seg.label = labels[static_cast<size_t>(i)];
                seg.begin = i;
                seg.end = j + 1;
                segments.push_back(seg);
            }
            i = j + 1;
        }
        return segments;
    }

    // ── 4. 폭 양자화 ─────────────────────────────────────────────────────────

    bool DeBruijnReconstruction::quantizeWidths(std::vector<Segment> &segments, DeBruijnStats &stats) const
    {
        if (segments.size() < 3)
            return false;

        // 단위 폭 추정. 폭 1 인 run 이 다수(코드상 36/52)라는 성질을 이용해
        // 하위 30% 분위수를 단위로 잡는다. 평균은 넓은 run 에 끌려간다.
        std::vector<double> widths;
        widths.reserve(segments.size());
        for (const Segment &s : segments)
            widths.push_back(static_cast<double>(s.end - s.begin));

        std::vector<double> sorted = widths;
        std::sort(sorted.begin(), sorted.end());
        const double unit = sorted[static_cast<size_t>(0.30 * static_cast<double>(sorted.size() - 1))];
        if (unit <= 0.0)
            return false;

        bool any = false;
        for (Segment &s : segments)
        {
            const double u = static_cast<double>(s.end - s.begin) / unit;
            const int q = static_cast<int>(std::lround(u));
            if (q < 1 || std::abs(u - static_cast<double>(q)) > config_.widthTolerance * static_cast<double>(q))
            {
                s.units = 0; // 신뢰할 수 없는 폭
                ++stats.rejectedByWidth;
                continue;
            }
            s.units = q;
            any = true;
        }
        return any;
    }

    // ── 5. 윈도우 조회 + 문맥 검사 ───────────────────────────────────────────

    void DeBruijnReconstruction::decodeSegments(std::vector<Segment> &segments, DeBruijnStats &stats) const
    {
        const int n = config_.minRunsForDecode > 0 ? config_.minRunsForDecode : pattern_.runWindow();
        const int count = static_cast<int>(segments.size());
        if (count < n)
            return;

        // 각 시작 위치에서 (색, 폭) 윈도우를 조회한다.
        std::vector<int> candidate(static_cast<size_t>(count), -1);
        for (int start = 0; start + n <= count; ++start)
        {
            DeBruijnPattern::RunKey key;
            key.reserve(static_cast<size_t>(n));
            bool usable = true;
            for (int t = 0; t < n; ++t)
            {
                const Segment &s = segments[static_cast<size_t>(start + t)];
                if (s.units <= 0)
                {
                    usable = false;
                    break;
                }
                key.emplace_back(s.label, s.units);
            }
            if (usable)
                candidate[static_cast<size_t>(start)] = pattern_.lookupRuns(key);
        }

        // 문맥 일관성 — 이웃 윈도우가 연속된 run 인덱스를 가리켜야 채택한다.
        const auto &runs = pattern_.runs();
        for (int start = 0; start + n <= count; ++start)
        {
            const int hit = candidate[static_cast<size_t>(start)];
            if (hit < 0)
                continue;

            bool supported = false;
            for (int d = 1; d <= config_.contextRadius && !supported; ++d)
            {
                if (start - d >= 0 && candidate[static_cast<size_t>(start - d)] == hit - d)
                    supported = true;
                if (start + d + n <= count && candidate[static_cast<size_t>(start + d)] == hit + d)
                    supported = true;
            }
            if (!supported)
            {
                ++stats.rejectedByContext;
                continue;
            }

            // 윈도우 안의 각 run 에 절대 스트라이프 인덱스를 붙인다.
            for (int t = 0; t < n; ++t)
            {
                const int runIndex = hit + t;
                if (runIndex < 0 || runIndex >= static_cast<int>(runs.size()))
                    continue;
                Segment &s = segments[static_cast<size_t>(start + t)];
                if (s.startStripe < 0)
                {
                    s.startStripe = runs[static_cast<size_t>(runIndex)].startStripe;
                    ++stats.decodedSegments;
                }
            }
        }
    }

    // ── 6. 경계 sub-pixel 정련 ───────────────────────────────────────────────

    double DeBruijnReconstruction::refineBoundary(const std::vector<Eigen::Vector3f> &ratio,
                                                  const std::vector<bool> &valid, const Segment &left,
                                                  const Segment &right) const
    {
        const std::vector<Eigen::Vector3f> &palette = palette_;
        const Eigen::Vector3f a = palette[static_cast<size_t>(left.label)];
        const Eigen::Vector3f b = palette[static_cast<size_t>(right.label)];

        // 두 팔레트 색을 잇는 방향에 사영하면 경계에서 단조 전이가 된다.
        const Eigen::Vector3f dir = b - a;
        const float denom = dir.squaredNorm();
        if (denom < 1e-6f)
            return -1.0;

        const auto project = [&](int x) { return (ratio[static_cast<size_t>(x)] - a).dot(dir) / denom; };

        // 경계 주변만 본다. 좌 세그먼트 끝 ~ 우 세그먼트 시작.
        const int lo = std::max(left.begin, left.end - 3);
        const int hi = std::min(right.end, right.begin + 3);
        for (int x = lo; x + 1 < hi; ++x)
        {
            if (!valid[static_cast<size_t>(x)] || !valid[static_cast<size_t>(x + 1)])
                continue;
            const float p0 = project(x);
            const float p1 = project(x + 1);
            // 0.5 를 가로지르는 구간에서 선형 보간.
            if ((p0 - 0.5f) * (p1 - 0.5f) <= 0.0f && std::abs(p1 - p0) > 1e-6f)
                return static_cast<double>(x) + static_cast<double>((0.5f - p0) / (p1 - p0));
        }
        return -1.0;
    }

    // ── 7. 삼각측량 ──────────────────────────────────────────────────────────

    bool DeBruijnReconstruction::triangulate(double cameraX, double cameraY, double uProjector,
                                             Eigen::Vector3d &point) const
    {
        const double fx = camera_.intrinsic(0, 0);
        const double fy = camera_.intrinsic(1, 1);
        const double cx = camera_.intrinsic(0, 2);
        const double cy = camera_.intrinsic(1, 2);

        Eigen::Vector3d ray = camera_.cameraToWorldRotation() *
                              Eigen::Vector3d((cameraX - cx) / fx, (cameraY - cy) / fy, 1.0);
        ray.normalize();

        // 컬러 코드는 프로젝터 x 만 알려주므로 대응하는 것은 점이 아니라 평면이다.
        // 법선의 y 성분이 0 인 것이 "y 를 제약하지 않는다"는 뜻이다.
        const double fxp = projector_.intrinsic(0, 0);
        const double cxp = projector_.intrinsic(0, 2);
        const double projectorX = uProjector * static_cast<double>(projector_.imageSize.x());
        const double slope = (projectorX - cxp) / fxp;

        Eigen::Vector3d normal = projector_.cameraToWorldRotation() * Eigen::Vector3d(-1.0, 0.0, slope);
        normal.normalize();

        const Eigen::Vector3d delta = projector_.center() - camera_.center();
        const double denom = ray.dot(normal);
        if (std::abs(denom) < 1e-8)
            return false;

        const double crossingDeg = std::asin(std::clamp(std::abs(denom), 0.0, 1.0)) * (180.0 / EIGEN_PI);
        if (crossingDeg < config_.minTriangulationAngleDeg)
            return false;

        const double distance = delta.dot(normal) / denom;
        if (!(distance > 0.0))
            return false;
        if (config_.maxCameraDistance.has_value() &&
            distance > static_cast<double>(*config_.maxCameraDistance))
            return false;

        point = camera_.center() + distance * ray;
        return point.allFinite();
    }

    // ── 파이프라인 ───────────────────────────────────────────────────────────

    DeBruijnResult DeBruijnReconstruction::run(const DeBruijnFrames &frames) const
    {
        const Eigen::Index rows = frames.color.x.rows();
        const Eigen::Index cols = frames.color.x.cols();
        if (frames.white.x.rows() != rows || frames.white.x.cols() != cols)
            throw std::invalid_argument("DeBruijnReconstruction: color 와 white 의 크기가 다르다");

        DeBruijnResult result;
        std::vector<Eigen::Vector3f> ratio;
        std::vector<bool> valid;

        for (Eigen::Index r = 0; r < rows; ++r)
        {
            computeRatioRow(frames, r, ratio, valid);
            const std::vector<int> labels = classifyRow(ratio, valid);
            std::vector<Segment> segments = segmentRow(labels);
            result.stats.segments += static_cast<int>(segments.size());
            if (segments.size() < 3)
                continue;

            if (!quantizeWidths(segments, result.stats))
                continue;
            decodeSegments(segments, result.stats);
            ++result.stats.rows;

            // 인접한 두 세그먼트가 모두 디코딩되고 절대 인덱스가 연속이어야 경계를 쓴다.
            for (size_t i = 0; i + 1 < segments.size(); ++i)
            {
                const Segment &left = segments[i];
                const Segment &right = segments[i + 1];
                if (left.startStripe < 0 || right.startStripe < 0)
                    continue;
                if (right.startStripe != left.startStripe + left.units)
                    continue; // 사이에 놓친 세그먼트가 있다
                if (right.begin != left.end)
                    continue; // 사이에 무효 픽셀이 있다

                const double x = refineBoundary(ratio, valid, left, right);
                if (x < 0.0)
                    continue;
                ++result.stats.boundaries;

                // 이 경계는 프로젝터에서 스트라이프 (right.startStripe) 가 시작하는 곳이다.
                const double u = pattern_.boundaryU(right.startStripe);

                Eigen::Vector3d point;
                if (!triangulate(x, static_cast<double>(r), u, point))
                    continue;

                const Eigen::Index xi = std::clamp<Eigen::Index>(static_cast<Eigen::Index>(std::lround(x)),
                                                                 0, cols - 1);
                result.points.push_back(point.cast<float>());
                result.colors.emplace_back(
                    std::clamp(static_cast<int>(std::lround(frames.white.x(r, xi))), 0, 255),
                    std::clamp(static_cast<int>(std::lround(frames.white.y(r, xi))), 0, 255),
                    std::clamp(static_cast<int>(std::lround(frames.white.z(r, xi))), 0, 255));
                ++result.stats.triangulated;
            }
        }
        return result;
    }

} // namespace sl
