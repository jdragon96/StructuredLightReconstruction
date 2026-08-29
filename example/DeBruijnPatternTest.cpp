// DeBruijnPatternTest.cpp
//
// src/StructuredLight/DeBruijnPattern.h 로 De Bruijn 컬러 스트라이프 패턴을 생성하고
// SVG(마스크 제작용) / PNG(프로젝터 해상도) / JSON 을 저장한다.
// python/DeBruijnPattern.py 와 동일한 결과를 내야 한다.
//
// 사용:
//   ./DeBruijnPatternTest [출력디렉터리]
//   ./DeBruijnPatternTest out --n 6 --no-repeat --offset 50
//   ./DeBruijnPatternTest out --pitch-mm 0.25 --height-mm 12

#include "StructuredLight/DeBruijnPattern.h"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

    std::string sequenceString(const std::vector<int> &values)
    {
        std::string out;
        out.reserve(values.size());
        for (const int v : values)
            out.push_back(static_cast<char>('0' + v));
        return out;
    }

    void writeText(const std::filesystem::path &path, const std::string &text)
    {
        std::ofstream file(path);
        if (!file)
            throw std::runtime_error("파일을 열 수 없다: " + path.string());
        file << text;
    }

    // ImageVec3 (x,y,z)=(R,G,B), 0~255 -> OpenCV BGR 8UC3
    cv::Mat toBgr(const sl::ImageVec3 &image)
    {
        const int rows = static_cast<int>(image.x.rows());
        const int cols = static_cast<int>(image.x.cols());
        cv::Mat bgr(rows, cols, CV_8UC3);
        for (int r = 0; r < rows; ++r)
        {
            auto *row = bgr.ptr<cv::Vec3b>(r);
            for (int c = 0; c < cols; ++c)
            {
                row[c] = cv::Vec3b(static_cast<uchar>(std::lround(image.z(r, c))),
                                   static_cast<uchar>(std::lround(image.y(r, c))),
                                   static_cast<uchar>(std::lround(image.x(r, c))));
            }
        }
        return bgr;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        sl::DeBruijnConfig config; // 기본값이 곧 제작 스펙 (k=3, n=4, offset 6, 72 스트라이프)
        std::filesystem::path outDir = (argc > 1 && argv[1][0] != '-') ? argv[1] : "dataset/debruijn";

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::invalid_argument(arg + " 뒤에 값이 필요하다");
                return argv[++i];
            };
            if (arg == "--k")
                config.alphabet = std::stoi(next());
            else if (arg == "--n")
                config.window = std::stoi(next());
            else if (arg == "--length")
                config.length = std::stoi(next());
            else if (arg == "--offset")
                config.offset = std::stoi(next());
            else if (arg == "--no-repeat")
                config.noRepeat = true;
            else if (arg == "--width")
                config.imageWidth = std::stoi(next());
            else if (arg == "--height")
                config.imageHeight = std::stoi(next());
            else if (arg == "--stripe-px")
                config.stripePx = std::stoi(next());
            else if (arg == "--pitch-mm")
                config.pitchMm = std::stod(next());
            else if (arg == "--height-mm")
                config.heightMm = std::stod(next());
        }

        const sl::DeBruijnPattern pattern(config);
        const sl::DeBruijnReport &report = pattern.report();

        std::filesystem::create_directories(outDir);
        const std::string stem = "debruijn_k" + std::to_string(config.alphabet) + "n" +
                                 std::to_string(config.window) + (config.noRepeat ? "_norep" : "") +
                                 "_" + std::to_string(config.length);

        cv::imwrite((outDir / (stem + ".png")).string(), toBgr(pattern.renderImage()));
        writeText(outDir / (stem + ".svg"), pattern.renderSvg());
        writeText(outDir / "debruijn_pattern_info.json", pattern.toJson());

        const char *kind = config.noRepeat ? "B'" : "B";
        std::cout << "base " << kind << "(" << config.alphabet << "," << config.window
                  << ")  period=" << report.period << "  " << sequenceString(pattern.baseSequence()) << "\n";
        std::cout << "code  offset=" << config.offset << " length=" << config.length << "\n";
        std::cout << "      " << sequenceString(pattern.code()) << "\n\n";

        std::cout << "윈도우 n=" << config.window << ": " << report.windowCount << "개 중 distinct "
                  << report.windowDistinct << ", 최대 중복 " << report.windowMaxMultiplicity
                  << "회 -> 절대 디코딩 " << (report.absolutelyDecodable ? "가능" : "불가") << "\n";
        std::cout << "run: " << report.runCount << "개, 보이는 경계 " << report.visibleEdges << "/"
                  << report.maxEdges << ", run 길이 분포 {";
        bool first = true;
        for (const auto &kv : report.runLengthHistogram)
        {
            std::cout << (first ? "" : ", ") << kv.first << ": " << kv.second;
            first = false;
        }
        std::cout << "}\n심볼 균형: {";
        first = true;
        for (const auto &kv : report.symbolCounts)
        {
            std::cout << (first ? "" : ", ") << kv.first << ": " << kv.second;
            first = false;
        }
        std::cout << "}\n\n";

        // 라운드트립 — 모든 위치에서 윈도우 조회가 자기 자신을 되찾는지.
        const std::vector<int> &code = pattern.code();
        int recovered = 0;
        for (int i = 0; i + config.window <= config.length; ++i)
        {
            const std::vector<int> w(code.begin() + i, code.begin() + i + config.window);
            if (pattern.lookup(w) == i)
                ++recovered;
        }
        std::cout << "윈도우 조회 라운드트립: " << recovered << "/" << report.windowCount
                  << (recovered == report.windowCount ? "  OK" : "  FAIL") << "\n";

        // ColorCoded 파이프라인용 변환이 성립하는지 (윈도우가 유일하지 않으면 예외).
        const sl::ColorCodeInfo info = pattern.toColorCodeInfo();
        std::cout << "ColorCodeInfo 변환: codeLength=" << info.codeLength
                  << ", decodeWindow=" << info.decodeWindow
                  << ", windowToStart=" << info.windowToStart.size() << "개  OK\n\n";

        std::cout << "-> " << (outDir / (stem + ".png")).string() << ", " << stem << ".svg, "
                  << "debruijn_pattern_info.json\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "오류: " << error.what() << "\n";
        return 1;
    }
}
