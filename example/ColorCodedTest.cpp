// ColorCodedTest.cpp
//
// dataset/ (Color-Coded Stripe + RGB Phase Shift, dlp_addon.py 'Export All' 출력)을
// 읽어 src/StructuredLight/Core/ColorCoded.h 의 CPU 파이프라인으로 3D 복원 후 PLY로 저장한다.

#include "StructuredLight/ColorCoded.h"

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    // JSON의 중첩 배열을 row-major로 평탄화한다.
    std::vector<double> flattenNumbers(const cv::FileNode &node)
    {
        std::vector<double> values;
        if (node.isSeq())
        {
            for (const auto &child : node)
            {
                const std::vector<double> nested = flattenNumbers(child);
                values.insert(values.end(), nested.begin(), nested.end());
            }
        }
        else
        {
            values.push_back(static_cast<double>(node));
        }
        return values;
    }

    sl::CameraCalibration loadCalibration(const std::string &path)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("캘리브레이션을 열 수 없습니다: " + path);

        const std::vector<double> imageSize = flattenNumbers(fs["image_size"]);
        const std::vector<double> intrinsic = flattenNumbers(fs["intrinsic"]);
        const std::vector<double> distortion = flattenNumbers(fs["distortion"]);
        const std::vector<double> rotate = flattenNumbers(fs["rotate"]);
        const std::vector<double> translate = flattenNumbers(fs["translate"]);
        fs.release();

        sl::CameraCalibration calib;
        calib.imageSize = Eigen::Vector2i(static_cast<int>(imageSize[0]), static_cast<int>(imageSize[1]));
        calib.intrinsic = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(intrinsic.data());
        calib.distortion = Eigen::Map<const Eigen::VectorXd>(distortion.data(), distortion.size());
        calib.rotation = Eigen::Vector3d(rotate[0], rotate[1], rotate[2]);
        calib.translation = Eigen::Vector3d(translate[0], translate[1], translate[2]);
        return calib;
    }

    sl::ColorCodeInfo loadColorCodeInfo(const std::string &path)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("color_pattern_info.json을 열 수 없습니다: " + path);

        const std::string mode = static_cast<std::string>(fs["mode"]);
        const int decodeWindow = static_cast<int>(fs["decode_window"]);

        const std::vector<double> sequenceRaw = flattenNumbers(fs["sequence"]);
        std::vector<int> sequence(sequenceRaw.size());
        for (size_t i = 0; i < sequenceRaw.size(); ++i)
            sequence[i] = static_cast<int>(sequenceRaw[i]);

        const std::vector<double> paletteRaw = flattenNumbers(fs["palette"]);
        std::vector<Eigen::Vector3i> palette;
        palette.reserve(paletteRaw.size() / 3);
        for (size_t i = 0; i + 2 < paletteRaw.size(); i += 3)
        {
            palette.emplace_back(static_cast<int>(paletteRaw[i]), static_cast<int>(paletteRaw[i + 1]),
                                 static_cast<int>(paletteRaw[i + 2]));
        }

        const int selfEqualizingInt = static_cast<int>(fs["self_equalizing"]);
        const bool selfEqualizing = selfEqualizingInt != 0;
        fs.release();

        return sl::ColorCodeInfo::create(sequence, palette, mode, selfEqualizing, decodeWindow);
    }

    sl::PhaseInfo loadPhaseInfo(const std::string &path)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("color_phase_info.json을 열 수 없습니다: " + path);

        sl::PhaseInfo info;
        info.frequency = static_cast<int>(static_cast<double>(fs["frequency"]));

        const std::vector<double> responseRaw = flattenNumbers(fs["response_matrix"]);
        if (responseRaw.size() == 9)
        {
            info.responseMatrix = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(responseRaw.data());
        }

        const int precompInt = static_cast<int>(fs["projector_precompensation"]);
        info.projectorPrecompensation = precompInt != 0;

        fs.release();
        return info;
    }

    // BGR 8비트 이미지를 (x,y,z)=(R,G,B), 0..255 범위의 ImageVec3로 변환한다.
    sl::ImageVec3 loadImageVec3(const std::string &path)
    {
        const cv::Mat img8 = cv::imread(path, cv::IMREAD_COLOR);
        if (img8.empty())
            throw std::runtime_error("이미지를 읽을 수 없습니다: " + path);

        cv::Mat img32;
        img8.convertTo(img32, CV_32FC3);

        std::vector<cv::Mat> channels(3);
        cv::split(img32, channels); // 0:B, 1:G, 2:R

        const auto toImage = [](const cv::Mat &m) -> sl::Image
        {
            const Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped(
                m.ptr<float>(), m.rows, m.cols);
            return sl::Image(mapped);
        };

        sl::ImageVec3 result;
        result.x = toImage(channels[2]); // R
        result.y = toImage(channels[1]); // G
        result.z = toImage(channels[0]); // B
        return result;
    }

    void savePly(const std::string &path, const sl::PointCloud &points, const std::vector<Eigen::Vector3i> &colors)
    {
        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("PLY 파일을 열 수 없습니다: " + path);

        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "element vertex " << points.size() << "\n";
        out << "property float x\n";
        out << "property float y\n";
        out << "property float z\n";
        out << "property uchar red\n";
        out << "property uchar green\n";
        out << "property uchar blue\n";
        out << "end_header\n";

        out << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < points.size(); ++i)
        {
            const auto &p = points[i];
            const auto &c = colors[i];
            out << p.x() << ' ' << p.y() << ' ' << p.z() << ' ' << c.x() << ' ' << c.y() << ' ' << c.z() << '\n';
        }
    }

} // namespace

int main(int argc, char **argv)
{
    const std::string root = argc > 1 ? argv[1] : "dataset";

    try
    {
        sl::CameraCalibration camera = loadCalibration(root + "/camera_calibration.json");
        sl::CameraCalibration projector = loadCalibration(root + "/projector_calibration.json");

        sl::ColorCodeInfo colorInfo = loadColorCodeInfo(root + "/color_pattern_info.json");
        sl::PhaseInfo phaseInfo = loadPhaseInfo(root + "/color_phase_info.json");

        sl::ColorCodedFrames frames;
        frames.color = loadImageVec3(root + "/color/render_00.png");
        frames.white = loadImageVec3(root + "/color/render_white.png");
        frames.phase = loadImageVec3(root + "/color_phase/render_00.png");

        sl::ColorCodedConfig config;

        const sl::ColorCodedReconstruction reconstructor(config, camera, projector, std::move(colorInfo), phaseInfo);
        const sl::ColorCodedResult result = reconstructor.run(frames);

        std::cout << "[color_coded] valid 3D points: " << result.points.size() << "\n";

        reconstructor.ExportDebugFile("color_coded_debug");
        std::cout << "[color_coded] debug images saved -> color_coded_debug/\n";

        const std::string outputPath = "color_coded_reconstruction.ply";
        savePly(outputPath, result.points, result.colors);
        std::cout << "[color_coded] PLY saved -> " << outputPath << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
