#include "StructuredLight/ColorCoded.h"
#include "StructuredLight/DebugImageExport.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <Eigen/Core>

namespace sl
{

    namespace
    {

        int oddKernel(int value)
        {
            value = std::max(1, value);
            return (value % 2 == 1) ? value : value + 1;
        }

        // cv2.medianBlur(BORDER_REPLICATE)와 동일한 1D median filter.
        std::vector<int> medianFilter1D(const std::vector<int> &values, int ksize)
        {
            const int half = ksize / 2;
            const int n = static_cast<int>(values.size());

            std::vector<int> out(static_cast<size_t>(n));
            std::vector<int> window(static_cast<size_t>(ksize));
            for (int i = 0; i < n; ++i)
            {
                for (int k = -half; k <= half; ++k)
                {
                    const int idx = std::clamp(i + k, 0, n - 1);
                    window[static_cast<size_t>(k + half)] = values[static_cast<size_t>(idx)];
                }
                std::nth_element(window.begin(), window.begin() + ksize / 2, window.end());
                out[static_cast<size_t>(i)] = window[static_cast<size_t>(ksize / 2)];
            }
            return out;
        }

        int popcount(unsigned value)
        {
            int count = 0;
            while (value)
            {
                count += static_cast<int>(value & 1u);
                value >>= 1u;
            }
            return count;
        }

    } // namespace

    // ── ColorCodeInfo ───────────────────────────────────────────────────────

    ColorCodeInfo ColorCodeInfo::create(const std::vector<int> &logicalSequence,
                                        const std::vector<Eigen::Vector3i> &logicalPalette255,
                                        const std::string &mode, bool selfEqualizing, int decodeWindow)
    {
        ColorCodeInfo info;
        info.mode = mode;

        if (selfEqualizing)
        {
            std::vector<Eigen::Vector3i> physicalColors;
            physicalColors.reserve(logicalSequence.size() * 2);
            for (int symbol : logicalSequence)
            {
                const Eigen::Vector3i &color = logicalPalette255[static_cast<size_t>(symbol)];
                physicalColors.push_back(color);
                physicalColors.push_back(Eigen::Vector3i(255, 255, 255) - color);
            }

            std::vector<Eigen::Vector3i> uniqueColors;
            std::map<std::array<int, 3>, int> colorToIndex;
            std::vector<int> physicalSequence;
            physicalSequence.reserve(physicalColors.size());
            for (const auto &color : physicalColors)
            {
                const std::array<int, 3> key{color.x(), color.y(), color.z()};
                auto it = colorToIndex.find(key);
                int index;
                if (it == colorToIndex.end())
                {
                    index = static_cast<int>(uniqueColors.size());
                    colorToIndex.emplace(key, index);
                    uniqueColors.push_back(color);
                }
                else
                {
                    index = it->second;
                }
                physicalSequence.push_back(index);
            }

            info.sequence = std::move(physicalSequence);
            info.palette.reserve(uniqueColors.size());
            for (const auto &color : uniqueColors)
                info.palette.push_back(color.cast<float>() / 255.0f);
            info.decodeWindow = 2 * decodeWindow;
        }
        else
        {
            info.sequence = logicalSequence;
            info.palette.reserve(logicalPalette255.size());
            for (const auto &color : logicalPalette255)
                info.palette.push_back(color.cast<float>() / 255.0f);
            info.decodeWindow = decodeWindow;
        }

        info.codeLength = static_cast<int>(info.sequence.size());
        if (info.codeLength < info.decodeWindow)
        {
            throw std::invalid_argument("Color code sequence is shorter than its decode window");
        }

        const int windows = info.codeLength - info.decodeWindow + 1;
        for (int start = 0; start < windows; ++start)
        {
            std::vector<int> key(info.sequence.begin() + start, info.sequence.begin() + start + info.decodeWindow);
            info.windowToStart.emplace(std::move(key), start);
        }
        if (static_cast<int>(info.windowToStart.size()) != windows)
        {
            throw std::invalid_argument("Color code windows are not unique; absolute indexing is ambiguous");
        }

        return info;
    }

    // ── ColorCodedReconstruction ──────────────────────────────────────────────

    ColorCodedReconstruction::ColorCodedReconstruction(ColorCodedConfig config, CameraCalibration camera,
                                                       CameraCalibration projector, ColorCodeInfo colorInfo,
                                                       PhaseInfo phaseInfo)
        : config_(std::move(config)),
          camera_(std::move(camera)),
          projector_(std::move(projector)),
          colorInfo_(std::move(colorInfo)),
          phaseInfo_(std::move(phaseInfo))
    {
        if (phaseInfo_.frequency <= 0)
            throw std::invalid_argument("RGB phase frequency must be positive");
    }

    ImageVec3 ColorCodedReconstruction::computeColorRatio(const ImageVec3 &color, const ImageVec3 &white)
    {
        const auto ratioChannel = [](const Image &c, const Image &w) -> Image
        {
            const Image denom = w.max(1.0f);
            return (c / denom).min(1.25f).max(0.0f);
        };
        return {ratioChannel(color.x, white.x), ratioChannel(color.y, white.y), ratioChannel(color.z, white.z)};
    }

    // wrapped = atan2(sqrt(3)*(G-B), 2R-G-B), modulation = sqrt(denom^2+numer^2)/3
    ColorCodedReconstruction::PhaseData ColorCodedReconstruction::computePhase(const ImageVec3 &phaseImage) const
    {
        const Eigen::Index rows = phaseImage.x.rows(), cols = phaseImage.x.cols();
        const float sqrt3 = std::sqrt(3.0f);

        Image wrapped(rows, cols);
        Image modulation(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r)
        {
            for (Eigen::Index c = 0; c < cols; ++c)
            {
                const float red = phaseImage.x(r, c);
                const float green = phaseImage.y(r, c);
                const float blue = phaseImage.z(r, c);

                const float numerator = sqrt3 * (green - blue);
                const float denominator = 2.0f * red - green - blue;

                float p = std::atan2(numerator, denominator);
                if (p < 0.0f)
                    p += kTwoPi;
                wrapped(r, c) = p;

                modulation(r, c) = std::sqrt(denominator * denominator + numerator * numerator) / 3.0f;
            }
        }

        PhaseData result;
        result.wrapped = std::move(wrapped);
        result.modulation = std::move(modulation);
        result.valid = result.modulation > config_.phaseModulationThreshold;
        return result;
    }

    Eigen::ArrayXXi ColorCodedReconstruction::classifyColorCode(const ImageVec3 &colorRatio) const
    {
        const Eigen::Index rows = colorRatio.x.rows(), cols = colorRatio.x.cols();
        Eigen::ArrayXXi labels(rows, cols);

        if (colorInfo_.mode == "hamming")
        {
            for (Eigen::Index r = 0; r < rows; ++r)
            {
                for (Eigen::Index c = 0; c < cols; ++c)
                {
                    const int bitR = colorRatio.x(r, c) > config_.hammingOnThreshold ? 1 : 0;
                    const int bitG = colorRatio.y(r, c) > config_.hammingOnThreshold ? 1 : 0;
                    const int bitB = colorRatio.z(r, c) > config_.hammingOnThreshold ? 1 : 0;
                    labels(r, c) = bitR + 2 * bitG + 4 * bitB;
                }
            }
        }
        else
        {
            for (Eigen::Index r = 0; r < rows; ++r)
            {
                for (Eigen::Index c = 0; c < cols; ++c)
                {
                    const Eigen::Vector3f observed(colorRatio.x(r, c), colorRatio.y(r, c), colorRatio.z(r, c));
                    int best = 0;
                    float bestDistSq = std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < colorInfo_.palette.size(); ++i)
                    {
                        const float distSq = (colorInfo_.palette[i] - observed).squaredNorm();
                        if (distSq < bestDistSq)
                        {
                            bestDistSq = distSq;
                            best = static_cast<int>(i);
                        }
                    }
                    labels(r, c) = best;
                }
            }
        }
        return labels;
    }

    int ColorCodedReconstruction::countAnchorWindows(const std::vector<int> &segmentLabels) const
    {
        const int kernel = oddKernel(config_.medianLabelKsize);
        const std::vector<int> filtered = medianFilter1D(segmentLabels, kernel);

        // run-length encode
        std::vector<int> runSymbols;
        std::vector<int> runWidths;
        size_t i = 0;
        while (i < filtered.size())
        {
            size_t j = i;
            while (j < filtered.size() && filtered[j] == filtered[i])
                ++j;
            runSymbols.push_back(filtered[i]);
            runWidths.push_back(static_cast<int>(j - i));
            i = j;
        }

        int count = 0;
        const int last = static_cast<int>(runSymbols.size()) - colorInfo_.decodeWindow + 1;
        for (int start = 0; start < std::max(0, last); ++start)
        {
            const int stop = start + colorInfo_.decodeWindow;
            bool tooShort = false;
            for (int k = start; k < stop; ++k)
            {
                if (runWidths[static_cast<size_t>(k)] < config_.minRunPixels)
                {
                    tooShort = true;
                    break;
                }
            }
            if (tooShort)
                continue;

            std::vector<int> key(runSymbols.begin() + start, runSymbols.begin() + stop);
            if (colorInfo_.windowToStart.count(key) > 0)
                ++count;
        }
        return count;
    }

    std::vector<float> ColorCodedReconstruction::emissionCost(float phaseFraction, int observedLabel,
                                                              const Eigen::Vector3f &observedRatio) const
    {
        const int nStates = phaseInfo_.frequency;
        std::vector<float> cost(static_cast<size_t>(nStates));

        for (int state = 0; state < nStates; ++state)
        {
            const float u = (static_cast<float>(state) + phaseFraction) / static_cast<float>(nStates);
            const int stripeIndex = std::clamp(static_cast<int>(u * static_cast<float>(colorInfo_.codeLength)), 0,
                                               colorInfo_.codeLength - 1);
            const int expectedLabel = colorInfo_.sequence[static_cast<size_t>(stripeIndex)];
            const Eigen::Vector3f &expectedColor = colorInfo_.palette[static_cast<size_t>(expectedLabel)];

            const float soft = (expectedColor - observedRatio).squaredNorm() / 3.0f;

            if (colorInfo_.mode == "hamming")
            {
                const unsigned xorValue = static_cast<unsigned>(expectedLabel ^ observedLabel);
                cost[static_cast<size_t>(state)] = 1.5f * static_cast<float>(popcount(xorValue)) + soft;
            }
            else
            {
                cost[static_cast<size_t>(state)] = 4.0f * soft;
            }
        }
        return cost;
    }

    std::vector<int16_t> ColorCodedReconstruction::decodeSegment(const std::vector<float> &fractions,
                                                                 const std::vector<int> &labels,
                                                                 const std::vector<Eigen::Vector3f> &ratios,
                                                                 float targetStepPx) const
    {
        const int length = static_cast<int>(fractions.size());
        const int nStates = phaseInfo_.frequency;

        std::vector<std::vector<int16_t>> backtrack(static_cast<size_t>(length),
                                                    std::vector<int16_t>(static_cast<size_t>(nStates), 0));

        std::vector<float> cost = emissionCost(fractions[0], labels[0], ratios[0]);
        for (int state = 0; state < nStates; ++state)
            backtrack[0][static_cast<size_t>(state)] = static_cast<int16_t>(state);

        const int maxJump = config_.localOrderJump;
        const float periodPixels = static_cast<float>(projector_.imageSize.x()) / static_cast<float>(nStates);
        const float truncationSq = config_.smoothnessTruncationPx * config_.smoothnessTruncationPx;

        for (int x = 1; x < length; ++x)
        {
            std::vector<float> best(static_cast<size_t>(nStates), std::numeric_limits<float>::infinity());
            std::vector<int16_t> bestPrevious(static_cast<size_t>(nStates), 0);

            for (int delta = -maxJump; delta <= maxJump; ++delta)
            {
                for (int current = 0; current < nStates; ++current)
                {
                    const int previous = current - delta;
                    if (previous < 0 || previous >= nStates)
                        continue;

                    const float projectedStep =
                        ((static_cast<float>(current) + fractions[static_cast<size_t>(x)]) -
                         (static_cast<float>(previous) + fractions[static_cast<size_t>(x - 1)])) *
                        periodPixels;
                    const float diff = projectedStep - targetStepPx;
                    const float smooth = config_.smoothnessWeight * std::min(diff * diff, truncationSq);

                    const float candidate = cost[static_cast<size_t>(previous)] + smooth;
                    if (candidate < best[static_cast<size_t>(current)])
                    {
                        best[static_cast<size_t>(current)] = candidate;
                        bestPrevious[static_cast<size_t>(current)] = static_cast<int16_t>(previous);
                    }
                }
            }

            const int globalPrevious = static_cast<int>(std::min_element(cost.begin(), cost.end()) - cost.begin());
            const float globalCost = cost[static_cast<size_t>(globalPrevious)] + config_.discontinuityPenalty;
            for (int state = 0; state < nStates; ++state)
            {
                if (globalCost < best[static_cast<size_t>(state)])
                {
                    best[static_cast<size_t>(state)] = globalCost;
                    bestPrevious[static_cast<size_t>(state)] = static_cast<int16_t>(globalPrevious);
                }
            }

            const std::vector<float> emission =
                emissionCost(fractions[static_cast<size_t>(x)], labels[static_cast<size_t>(x)], ratios[static_cast<size_t>(x)]);
            for (int state = 0; state < nStates; ++state)
            {
                cost[static_cast<size_t>(state)] = best[static_cast<size_t>(state)] + emission[static_cast<size_t>(state)];
            }
            backtrack[static_cast<size_t>(x)] = std::move(bestPrevious);
        }

        std::vector<int16_t> orders(static_cast<size_t>(length));
        orders[static_cast<size_t>(length - 1)] =
            static_cast<int16_t>(std::min_element(cost.begin(), cost.end()) - cost.begin());
        for (int x = length - 1; x > 0; --x)
        {
            orders[static_cast<size_t>(x - 1)] =
                backtrack[static_cast<size_t>(x)][static_cast<size_t>(orders[static_cast<size_t>(x)])];
        }
        return orders;
    }

    ColorCodedReconstruction::FringeOrderResult ColorCodedReconstruction::decodeFringeOrder(
        const Image &wrapped, const ValidMask &phaseValid, const Eigen::ArrayXXi &labels,
        const ImageVec3 &colorRatio) const
    {
        const Eigen::Index rows = wrapped.rows(), cols = wrapped.cols();

        Eigen::ArrayXXi orders = Eigen::ArrayXXi::Constant(rows, cols, -1);

        for (Eigen::Index y = 0; y < rows; ++y)
        {
            Eigen::Index x = 0;
            while (x < cols)
            {
                if (!phaseValid(y, x))
                {
                    ++x;
                    continue;
                }

                const Eigen::Index start = x;
                while (x < cols && phaseValid(y, x))
                    ++x;
                const Eigen::Index stop = x;
                const int segmentLength = static_cast<int>(stop - start);
                if (segmentLength < config_.minSegmentPixels)
                    continue;

                std::vector<int> segmentLabels(static_cast<size_t>(segmentLength));
                std::vector<float> fractions(static_cast<size_t>(segmentLength));
                std::vector<Eigen::Vector3f> ratios(static_cast<size_t>(segmentLength));
                for (int k = 0; k < segmentLength; ++k)
                {
                    const Eigen::Index px = start + k;
                    segmentLabels[static_cast<size_t>(k)] = labels(y, px);
                    fractions[static_cast<size_t>(k)] = wrapped(y, px) / kTwoPi;
                    ratios[static_cast<size_t>(k)] =
                        Eigen::Vector3f(colorRatio.x(y, px), colorRatio.y(y, px), colorRatio.z(y, px));
                }

                if (countAnchorWindows(segmentLabels) < config_.minAnchorWindows)
                    continue;

                const std::vector<int16_t> forward =
                    decodeSegment(fractions, segmentLabels, ratios, config_.expectedProjectorStepPx);

                std::vector<int16_t> segmentOrders;
                if (config_.bidirectionalConsensus)
                {
                    std::vector<float> reversedFractions(fractions.rbegin(), fractions.rend());
                    std::vector<int> reversedLabels(segmentLabels.rbegin(), segmentLabels.rend());
                    std::vector<Eigen::Vector3f> reversedRatios(ratios.rbegin(), ratios.rend());

                    const std::vector<int16_t> backwardReversed =
                        decodeSegment(reversedFractions, reversedLabels, reversedRatios, -config_.expectedProjectorStepPx);
                    const std::vector<int16_t> backward(backwardReversed.rbegin(), backwardReversed.rend());

                    segmentOrders.assign(static_cast<size_t>(segmentLength), -1);
                    for (int k = 0; k < segmentLength; ++k)
                    {
                        if (forward[static_cast<size_t>(k)] == backward[static_cast<size_t>(k)])
                        {
                            segmentOrders[static_cast<size_t>(k)] = forward[static_cast<size_t>(k)];
                        }
                    }
                }
                else
                {
                    segmentOrders = forward;
                }

                for (int k = 0; k < segmentLength; ++k)
                    orders(y, start + k) = segmentOrders[static_cast<size_t>(k)];
            }
        }

        FringeOrderResult result;
        result.orders = std::move(orders);
        result.valid = phaseValid && (result.orders >= 0);
        return result;
    }

    ColorCodedReconstruction::ProjectorCoordinate ColorCodedReconstruction::computeProjectorCoordinate(
        const ColorCodedFrames &frames) const
    {
        const ImageVec3 colorRatio = computeColorRatio(frames.color, frames.white);
        const PhaseData phase = computePhase(frames.phase);
        const Eigen::ArrayXXi labels = classifyColorCode(colorRatio);
        const FringeOrderResult fringe = decodeFringeOrder(phase.wrapped, phase.valid, labels, colorRatio);

        const Eigen::Index rows = phase.wrapped.rows(), cols = phase.wrapped.cols();
        Image uProjector(rows, cols);
        ValidMask valid(rows, cols);

        const float frequency = static_cast<float>(phaseInfo_.frequency);
        const float nan = std::numeric_limits<float>::quiet_NaN();

        for (Eigen::Index r = 0; r < rows; ++r)
        {
            for (Eigen::Index c = 0; c < cols; ++c)
            {
                const float fraction = phase.wrapped(r, c) / kTwoPi;
                const float u = (static_cast<float>(fringe.orders(r, c)) + fraction) / frequency;
                const bool ok = fringe.valid(r, c) && u >= 0.0f && u < 1.0f;

                uProjector(r, c) = ok ? u : nan;
                valid(r, c) = ok;
            }
        }

        debug_.colorRatio = colorRatio;
        debug_.labels = labels;
        debug_.wrappedPhase = phase.wrapped;
        debug_.modulation = phase.modulation;
        debug_.phaseValid = phase.valid;
        debug_.fringeOrders = fringe.orders;
        debug_.fringeValid = fringe.valid;
        debug_.uProjector = uProjector;
        debug_.valid = valid;

        return {std::move(uProjector), std::move(valid)};
    }

    ColorCodedResult ColorCodedReconstruction::triangulate(const Image &uProjector, const ValidMask &validIn,
                                                           const ImageVec3 &whiteImage) const
    {
        const Eigen::Index rows = uProjector.rows(), cols = uProjector.cols();

        const double fx = camera_.intrinsic(0, 0);
        const double fy = camera_.intrinsic(1, 1);
        const double cx = camera_.intrinsic(0, 2);
        const double cy = camera_.intrinsic(1, 2);
        const Eigen::Matrix3d Rc2w = camera_.cameraToWorldRotation();
        const Eigen::Vector3d camCenter = camera_.center();

        const double fxp = projector_.intrinsic(0, 0);
        const double cxp = projector_.intrinsic(0, 2);
        const double Wp = static_cast<double>(projector_.imageSize.x());
        const Eigen::Matrix3d Rp2w = projector_.cameraToWorldRotation();
        const Eigen::Vector3d projCenter = projector_.center();
        const Eigen::Vector3d delta = projCenter - camCenter;

        ColorCodedResult result;
        result.uProjector = uProjector;
        result.valid = ValidMask::Constant(rows, cols, false);

        Image depth = Image::Zero(rows, cols);

        for (Eigen::Index r = 0; r < rows; ++r)
        {
            for (Eigen::Index c = 0; c < cols; ++c)
            {
                bool valid = validIn(r, c);

                Eigen::Vector3d rayWorld = Rc2w * Eigen::Vector3d((static_cast<double>(c) - cx) / fx,
                                                                  (static_cast<double>(r) - cy) / fy, 1.0);
                rayWorld.normalize();

                const double projectorX = static_cast<double>(uProjector(r, c)) * Wp; // valid==false면 NaN
                const double slope = (projectorX - cxp) / fxp;
                Eigen::Vector3d normalWorld = Rp2w * Eigen::Vector3d(-1.0, 0.0, slope);
                normalWorld.normalize();

                const double denom = rayWorld.dot(normalWorld);
                const double numer = delta.dot(normalWorld);

                const double crossingAngleDeg = std::asin(std::clamp(std::abs(denom), 0.0, 1.0)) * (180.0 / EIGEN_PI);
                valid = valid && (crossingAngleDeg >= config_.minTriangulationAngleDeg);
                valid = valid && (std::abs(denom) > 1e-8);

                const double distance = valid ? numer / denom : 0.0;
                valid = valid && (distance > 0.0);
                if (valid && config_.maxCameraDistance.has_value())
                {
                    valid = valid && (distance <= static_cast<double>(*config_.maxCameraDistance));
                }

                const Eigen::Vector3d point = camCenter + distance * rayWorld;
                valid = valid && std::isfinite(point.x()) && std::isfinite(point.y()) && std::isfinite(point.z());

                if (valid)
                {
                    result.points.push_back(point.cast<float>());
                    result.colors.emplace_back(
                        std::clamp(static_cast<int>(std::lround(whiteImage.x(r, c))), 0, 255),
                        std::clamp(static_cast<int>(std::lround(whiteImage.y(r, c))), 0, 255),
                        std::clamp(static_cast<int>(std::lround(whiteImage.z(r, c))), 0, 255));
                    result.valid(r, c) = true;
                    depth(r, c) = static_cast<float>(point.z());
                }
            }
        }

        debug_.hasTriangulation = true;
        debug_.depth = std::move(depth);
        debug_.triangulationValid = result.valid;

        return result;
    }

    ColorCodedResult ColorCodedReconstruction::run(const ColorCodedFrames &frames) const
    {
        const ProjectorCoordinate projector = computeProjectorCoordinate(frames);
        return triangulate(projector.uProjector, projector.valid, frames.white);
    }

    void ColorCodedReconstruction::ExportDebugFile(const std::string &outputDir) const
    {
        std::filesystem::create_directories(outputDir);

        detail::writeImagePNGAuto(outputDir + "/color_ratio_r.png", debug_.colorRatio.x);
        detail::writeImagePNGAuto(outputDir + "/color_ratio_g.png", debug_.colorRatio.y);
        detail::writeImagePNGAuto(outputDir + "/color_ratio_b.png", debug_.colorRatio.z);

        detail::writeImagePNGAuto(outputDir + "/labels.png", debug_.labels.cast<float>());

        detail::writeImagePNG(outputDir + "/wrapped_phase.png", debug_.wrappedPhase, 0.0f, kTwoPi);
        detail::writeImagePNGAuto(outputDir + "/modulation.png", debug_.modulation);
        detail::writeMaskPNG(outputDir + "/phase_valid.png", debug_.phaseValid);

        detail::writeImagePNGAuto(outputDir + "/fringe_orders.png", debug_.fringeOrders.cast<float>());
        detail::writeMaskPNG(outputDir + "/fringe_valid.png", debug_.fringeValid);

        detail::writeImagePNG(outputDir + "/uprojector.png", debug_.uProjector, 0.0f, 1.0f);
        detail::writeMaskPNG(outputDir + "/valid.png", debug_.valid);

        if (debug_.hasTriangulation)
        {
            detail::writeImagePNGAuto(outputDir + "/depth.png", debug_.depth);
            detail::writeMaskPNG(outputDir + "/triangulation_valid.png", debug_.triangulationValid);
        }
    }

} // namespace sl
