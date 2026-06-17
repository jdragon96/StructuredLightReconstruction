#pragma once

// DebugImageExport.h
//
// 의존성 없는 최소 PNG(8비트 grayscale) 인코더.
// MultiFrequencyPhaseShift / ColorCoded 의 ExportDebugFile()에서 중간 연산 결과를
// 시각화하기 위해 공용으로 사용한다.

#include "StructuredLight/MultiFrequencyPhaseShift.h"

#include <string>

namespace sl
{
    namespace detail
    {

        // [minV, maxV] -> [0, 255]로 선형 정규화하여 8비트 grayscale PNG로 저장한다.
        void writeImagePNG(const std::string &path, const Image &img, float minV, float maxV);

        // 이미지 자체의 최소/최대값으로 정규화한다. NaN/Inf는 0으로 취급한다.
        void writeImagePNGAuto(const std::string &path, const Image &img);

        // bool 마스크를 0/255 PNG로 저장한다.
        void writeMaskPNG(const std::string &path, const ValidMask &mask);

    } // namespace detail
} // namespace sl
