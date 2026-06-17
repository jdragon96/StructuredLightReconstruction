#include "StructuredLight/DebugImageExport.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace sl
{
    namespace detail
    {

        namespace
        {

            // ── 의존성 없는 최소 PNG 인코더 (8비트 grayscale, 무압축 DEFLATE stored block) ──

            uint32_t crc32(const uint8_t *data, size_t length)
            {
                static uint32_t table[256];
                static bool initialized = false;
                if (!initialized)
                {
                    for (uint32_t i = 0; i < 256; ++i)
                    {
                        uint32_t c = i;
                        for (int k = 0; k < 8; ++k)
                            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                        table[i] = c;
                    }
                    initialized = true;
                }
                uint32_t crc = 0xFFFFFFFFu;
                for (size_t i = 0; i < length; ++i)
                    crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
                return crc ^ 0xFFFFFFFFu;
            }

            uint32_t adler32(const uint8_t *data, size_t length)
            {
                uint32_t a = 1, b = 0;
                for (size_t i = 0; i < length; ++i)
                {
                    a = (a + data[i]) % 65521u;
                    b = (b + a) % 65521u;
                }
                return (b << 16) | a;
            }

            void appendBE32(std::vector<uint8_t> &out, uint32_t v)
            {
                out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
                out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
                out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
                out.push_back(static_cast<uint8_t>(v & 0xFFu));
            }

            void writeChunk(std::ofstream &out, const char type[4], const std::vector<uint8_t> &data)
            {
                std::vector<uint8_t> header;
                appendBE32(header, static_cast<uint32_t>(data.size()));
                out.write(reinterpret_cast<const char *>(header.data()), 4);

                std::vector<uint8_t> crcInput(4 + data.size());
                std::memcpy(crcInput.data(), type, 4);
                if (!data.empty())
                    std::memcpy(crcInput.data() + 4, data.data(), data.size());

                out.write(type, 4);
                if (!data.empty())
                    out.write(reinterpret_cast<const char *>(data.data()), data.size());

                std::vector<uint8_t> trailer;
                appendBE32(trailer, crc32(crcInput.data(), crcInput.size()));
                out.write(reinterpret_cast<const char *>(trailer.data()), 4);
            }

            // 8비트 grayscale PNG로 저장한다. pixels는 row-major, width*height 길이.
            void writePNGGray8(const std::string &path, int width, int height, const std::vector<uint8_t> &pixels)
            {
                std::ofstream out(path, std::ios::binary);
                if (!out)
                    throw std::runtime_error("디버그 이미지를 저장할 수 없습니다: " + path);

                static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
                out.write(reinterpret_cast<const char *>(signature), sizeof(signature));

                std::vector<uint8_t> ihdr;
                appendBE32(ihdr, static_cast<uint32_t>(width));
                appendBE32(ihdr, static_cast<uint32_t>(height));
                ihdr.push_back(8); // bit depth
                ihdr.push_back(0); // color type: grayscale
                ihdr.push_back(0); // compression method
                ihdr.push_back(0); // filter method
                ihdr.push_back(0); // interlace method
                writeChunk(out, "IHDR", ihdr);

                // 각 행 앞에 filter byte(0, None)를 붙인 원본 스캔라인.
                std::vector<uint8_t> raw;
                raw.reserve(static_cast<size_t>(height) * (static_cast<size_t>(width) + 1));
                for (int y = 0; y < height; ++y)
                {
                    raw.push_back(0);
                    const uint8_t *row = pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
                    raw.insert(raw.end(), row, row + width);
                }

                // zlib 스트림: 2바이트 헤더 + 무압축 DEFLATE stored block(들) + Adler-32.
                std::vector<uint8_t> idat;
                idat.push_back(0x78);
                idat.push_back(0x01);

                size_t offset = 0;
                const size_t total = raw.size();
                do
                {
                    const size_t remaining = total - offset;
                    const size_t blockSize = std::min<size_t>(remaining, 65535);
                    const bool isFinal = (offset + blockSize >= total);

                    idat.push_back(isFinal ? 0x01 : 0x00);
                    idat.push_back(static_cast<uint8_t>(blockSize & 0xFFu));
                    idat.push_back(static_cast<uint8_t>((blockSize >> 8) & 0xFFu));
                    const uint16_t nlen = static_cast<uint16_t>(~static_cast<uint16_t>(blockSize));
                    idat.push_back(static_cast<uint8_t>(nlen & 0xFFu));
                    idat.push_back(static_cast<uint8_t>((nlen >> 8) & 0xFFu));

                    idat.insert(idat.end(), raw.begin() + static_cast<long>(offset),
                                raw.begin() + static_cast<long>(offset + blockSize));
                    offset += blockSize;
                } while (offset < total);

                appendBE32(idat, adler32(raw.data(), raw.size()));

                writeChunk(out, "IDAT", idat);
                writeChunk(out, "IEND", {});
            }

        } // namespace

        void writeImagePNG(const std::string &path, const Image &img, float minV, float maxV)
        {
            const float range = (maxV - minV) > 1e-12f ? (maxV - minV) : 1.0f;

            std::vector<uint8_t> pixels(static_cast<size_t>(img.rows()) * static_cast<size_t>(img.cols()));
            size_t idx = 0;
            for (Eigen::Index r = 0; r < img.rows(); ++r)
            {
                for (Eigen::Index c = 0; c < img.cols(); ++c)
                {
                    const float v = std::clamp((img(r, c) - minV) / range, 0.0f, 1.0f);
                    pixels[idx++] = static_cast<uint8_t>(std::round(v * 255.0f));
                }
            }
            writePNGGray8(path, static_cast<int>(img.cols()), static_cast<int>(img.rows()), pixels);
        }

        void writeImagePNGAuto(const std::string &path, const Image &img)
        {
            const Image finite = img.unaryExpr([](float v)
                                               { return std::isfinite(v) ? v : 0.0f; });
            writeImagePNG(path, finite, finite.minCoeff(), finite.maxCoeff());
        }

        void writeMaskPNG(const std::string &path, const ValidMask &mask)
        {
            std::vector<uint8_t> pixels(static_cast<size_t>(mask.rows()) * static_cast<size_t>(mask.cols()));
            size_t idx = 0;
            for (Eigen::Index r = 0; r < mask.rows(); ++r)
            {
                for (Eigen::Index c = 0; c < mask.cols(); ++c)
                    pixels[idx++] = mask(r, c) ? 255 : 0;
            }
            writePNGGray8(path, static_cast<int>(mask.cols()), static_cast<int>(mask.rows()), pixels);
        }

    } // namespace detail
} // namespace sl
