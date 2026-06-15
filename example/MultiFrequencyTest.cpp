// MultiFrequencyTest.cpp
//
// dataset/ (Gray Code + Multi-Frequency Phase Shift, dlp_addon.py 'Export All' 출력)을
// 읽어 src/Core/MultiFrequencyPhaseShift.h 의 CPU 파이프라인으로 3D 복원 후 PLY로 저장한다.

#include "StructuredLight/Core/MultiFrequencyPhaseShift.h"

#include <opencv2/opencv.hpp>

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// JSON의 중첩 배열(예: 3x3 intrinsic)을 row-major로 평탄화한다.
std::vector<double> flattenNumbers(const cv::FileNode& node) {
    std::vector<double> values;
    if (node.isSeq()) {
        for (const auto& child : node) {
            const std::vector<double> nested = flattenNumbers(child);
            values.insert(values.end(), nested.begin(), nested.end());
        }
    } else {
        values.push_back(static_cast<double>(node));
    }
    return values;
}

sl::CameraCalibration loadCalibration(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::runtime_error("캘리브레이션을 열 수 없습니다: " + path);

    const std::vector<double> imageSize  = flattenNumbers(fs["image_size"]);
    const std::vector<double> intrinsic  = flattenNumbers(fs["intrinsic"]);
    const std::vector<double> distortion = flattenNumbers(fs["distortion"]);
    const std::vector<double> rotate     = flattenNumbers(fs["rotate"]);
    const std::vector<double> translate  = flattenNumbers(fs["translate"]);
    fs.release();

    sl::CameraCalibration calib;
    calib.imageSize = Eigen::Vector2i(static_cast<int>(imageSize[0]), static_cast<int>(imageSize[1]));
    calib.intrinsic = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(intrinsic.data());
    calib.distortion = Eigen::Map<const Eigen::VectorXd>(distortion.data(), distortion.size());
    calib.rotation    = Eigen::Vector3d(rotate[0], rotate[1], rotate[2]);
    calib.translation = Eigen::Vector3d(translate[0], translate[1], translate[2]);
    return calib;
}

sl::Image loadImage(const std::string& path) {
    const cv::Mat img8 = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img8.empty()) throw std::runtime_error("이미지를 읽을 수 없습니다: " + path);

    cv::Mat img32;
    img8.convertTo(img32, CV_32F);

    const Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped(
        img32.ptr<float>(), img32.rows, img32.cols);
    return sl::Image(mapped);
}

void savePly(const std::string& path, const sl::PointCloud& points) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("PLY 파일을 열 수 없습니다: " + path);

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << points.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "end_header\n";

    out << std::fixed << std::setprecision(6);
    for (const auto& p : points) {
        out << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : "dataset";

    try {
        sl::CameraCalibration camera    = loadCalibration(root + "/camera_calibration.json");
        sl::CameraCalibration projector = loadCalibration(root + "/projector_calibration.json");

        // Gray code (MSB first) + white/black 레퍼런스
        sl::GrayCodeFrames gray;
        for (int i = 0; i < 4; ++i) {
            gray.bitPlanes.push_back(loadImage(root + "/gray/render_0" + std::to_string(i) + ".png"));
        }
        gray.white = loadImage(root + "/gray/render_white.png");
        gray.black = loadImage(root + "/gray/render_black.png");

        // 주파수별 4-step phase shift
        sl::PhaseFrequencyFrames freqFrames;
        for (int freq : { 16, 32, 64 }) {
            std::array<sl::Image, 4> steps;
            for (int i = 0; i < 4; ++i) {
                steps[i] = loadImage(root + "/" + std::to_string(freq) + "/render_0" + std::to_string(i) + ".png");
            }
            freqFrames[freq] = std::move(steps);
        }

        sl::MultiFreqConfig config;
        config.nGrayBits   = 4;
        config.frequencies = { 16, 32, 64 };

        const sl::MultiFrequencyPhaseShift reconstructor(config, camera, projector);
        const sl::ReconstructionResult result = reconstructor.run(gray, freqFrames);

        const sl::PointCloud points = result.toPointCloud();
        std::cout << "[multi_freq] valid 3D points: " << points.size() << "\n";

        reconstructor.ExportDebugFile("multi_freq_debug");
        std::cout << "[multi_freq] debug images saved -> multi_freq_debug/\n";

        const std::string outputPath = "multi_freq_reconstruction.ply";
        savePly(outputPath, points);
        std::cout << "[multi_freq] PLY saved -> " << outputPath << "\n";
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
