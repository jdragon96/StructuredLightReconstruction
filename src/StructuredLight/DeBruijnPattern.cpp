#include "StructuredLight/DeBruijnPattern.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sl
{

    namespace
    {

        // (n-1) 길이 단어를 정수 키로 접는다. 노드 탐색을 해시로 처리하기 위한 것.
        std::size_t wordKey(const std::vector<int> &word, int k)
        {
            std::size_t key = 0;
            for (const int s : word)
                key = key * static_cast<std::size_t>(k) + static_cast<std::size_t>(s);
            return key;
        }

        std::vector<int> keyToWord(std::size_t key, int k, int len)
        {
            std::vector<int> word(static_cast<size_t>(len));
            for (int i = len - 1; i >= 0; --i)
            {
                word[static_cast<size_t>(i)] = static_cast<int>(key % static_cast<std::size_t>(k));
                key /= static_cast<std::size_t>(k);
            }
            return word;
        }

        std::string formatDouble(double v)
        {
            std::ostringstream os;
            os << std::defaultfloat << std::setprecision(10) << v;
            return os.str();
        }

    } // namespace

    // ── 수열 생성 ────────────────────────────────────────────────────────────

    std::vector<int> DeBruijnPattern::deBruijn(int k, int n)
    {
        if (k < 2 || n < 1)
            throw std::invalid_argument("deBruijn: k >= 2, n >= 1 이어야 한다");

        std::vector<int> a(static_cast<size_t>(k) * static_cast<size_t>(n), 0);
        std::vector<int> sequence;
        sequence.reserve(static_cast<size_t>(std::pow(k, n)));

        // 표준 Lyndon word 재귀. n 의 약수 길이인 Lyndon word 들을 사전순으로 이어붙이면
        // 길이 k^n 의 De Bruijn 수열이 된다.
        std::function<void(int, int)> db = [&](int t, int p) {
            if (t > n)
            {
                if (n % p == 0)
                    sequence.insert(sequence.end(), a.begin() + 1, a.begin() + 1 + p);
                return;
            }
            a[static_cast<size_t>(t)] = a[static_cast<size_t>(t - p)];
            db(t + 1, p);
            for (int j = a[static_cast<size_t>(t - p)] + 1; j < k; ++j)
            {
                a[static_cast<size_t>(t)] = j;
                db(t + 1, t);
            }
        };
        db(1, 1);
        return sequence;
    }

    std::vector<int> DeBruijnPattern::deBruijnNoRepeat(int k, int n)
    {
        if (k < 3)
            throw std::invalid_argument("deBruijnNoRepeat: 인접 상이 제약은 k >= 3 에서만 의미가 있다");
        if (n < 2)
            throw std::invalid_argument("deBruijnNoRepeat: n >= 2 이어야 한다");

        const int wordLen = n - 1;

        // 노드 = 길이 (n-1) 이면서 인접 심볼이 서로 다른 단어.
        std::vector<std::size_t> nodes;
        {
            std::vector<int> word(static_cast<size_t>(wordLen), 0);
            std::function<void(int)> build = [&](int pos) {
                if (pos == wordLen)
                {
                    nodes.push_back(wordKey(word, k));
                    return;
                }
                for (int s = 0; s < k; ++s)
                {
                    if (pos > 0 && s == word[static_cast<size_t>(pos - 1)])
                        continue;
                    word[static_cast<size_t>(pos)] = s;
                    build(pos + 1);
                }
            };
            build(0);
        }

        // 엣지 = (a1..a_{n-1}) -> (a2..a_{n-1}, b), 단 b != a_{n-1}.
        // 모든 노드의 입·출차수가 (k-1) 로 같으므로 오일러 순환이 존재한다.
        std::unordered_map<std::size_t, std::vector<int>> out;
        out.reserve(nodes.size() * 2);
        for (const std::size_t key : nodes)
        {
            const std::vector<int> word = keyToWord(key, k, wordLen);
            std::vector<int> next;
            next.reserve(static_cast<size_t>(k - 1));
            for (int b = 0; b < k; ++b)
            {
                if (b != word.back())
                    next.push_back(b);
            }
            out.emplace(key, std::move(next));
        }

        // Hierholzer. 스택에 노드를 쌓고 나가는 엣지가 없으면 회로에 밀어낸다.
        std::vector<std::size_t> stack{nodes.front()};
        std::vector<std::size_t> circuit;
        circuit.reserve(nodes.size() * static_cast<size_t>(k - 1) + 1);

        while (!stack.empty())
        {
            const std::size_t v = stack.back();
            std::vector<int> &edges = out[v];
            if (!edges.empty())
            {
                const int b = edges.back();
                edges.pop_back();
                std::vector<int> word = keyToWord(v, k, wordLen);
                word.erase(word.begin());
                word.push_back(b);
                stack.push_back(wordKey(word, k));
            }
            else
            {
                circuit.push_back(v);
                stack.pop_back();
            }
        }
        std::reverse(circuit.begin(), circuit.end());

        // 엣지마다 심볼 하나 — 각 노드의 마지막 심볼을 취한다(시작 노드는 제외).
        std::vector<int> sequence;
        sequence.reserve(circuit.size());
        for (size_t i = 1; i < circuit.size(); ++i)
            sequence.push_back(keyToWord(circuit[i], k, wordLen).back());
        return sequence;
    }

    std::vector<int> DeBruijnPattern::cyclicWindow(const std::vector<int> &sequence, int offset, int length)
    {
        if (sequence.empty())
            throw std::invalid_argument("cyclicWindow: 빈 수열");
        const int period = static_cast<int>(sequence.size());
        std::vector<int> out(static_cast<size_t>(length));
        for (int i = 0; i < length; ++i)
        {
            const int idx = ((offset + i) % period + period) % period;
            out[static_cast<size_t>(i)] = sequence[static_cast<size_t>(idx)];
        }
        return out;
    }

    int DeBruijnPattern::minimumWindow(int k, int length, bool noRepeat)
    {
        for (int n = 2; n <= 16; ++n)
        {
            // 제약 없음 k^n, 인접 상이 k(k-1)^(n-1)
            double capacity = noRepeat ? static_cast<double>(k) * std::pow(k - 1, n - 1)
                                       : std::pow(k, n);
            if (static_cast<double>(length - n + 1) <= capacity)
                return n;
        }
        return 0;
    }

    // ── 리포트 ──────────────────────────────────────────────────────────────

    std::string DeBruijnReport::summary() const
    {
        std::ostringstream os;
        os << "길이 " << length << " / 주기 " << period
           << " · 윈도우 " << windowDistinct << "/" << windowCount
           << " (최대 중복 " << windowMaxMultiplicity << ")"
           << " · 절대 디코딩 " << (absolutelyDecodable ? "가능" : "불가")
           << " · 보이는 경계 " << visibleEdges << "/" << maxEdges;
        return os.str();
    }

    // ── 생성자 ──────────────────────────────────────────────────────────────

    DeBruijnPattern::DeBruijnPattern(DeBruijnConfig config) : config_(std::move(config))
    {
        const int k = config_.alphabet;
        const int n = config_.window;
        const int L = config_.length;

        if (k < 2)
            throw std::invalid_argument("DeBruijnPattern: alphabet >= 2 이어야 한다");
        if (n < 2)
            throw std::invalid_argument("DeBruijnPattern: window >= 2 이어야 한다");
        if (L < n)
            throw std::invalid_argument("DeBruijnPattern: length >= window 이어야 한다");
        if (static_cast<int>(config_.palette.size()) < k)
            throw std::invalid_argument("DeBruijnPattern: 팔레트 색이 alphabet 보다 적다");

        sequence_ = config_.noRepeat ? deBruijnNoRepeat(k, n) : deBruijn(k, n);
        const int period = static_cast<int>(sequence_.size());

        // 유일성 조건. 만족하지 못하면 만들어봐야 절대 위치를 못 얻으므로 여기서 막는다.
        if (L - n + 1 > period)
        {
            std::ostringstream os;
            os << "DeBruijnPattern: 윈도우 유일성 불가 — 필요 " << (L - n + 1)
               << "개 > 주기 " << period << "개. window 를 "
               << minimumWindow(k, L, config_.noRepeat) << " 이상으로 올려야 한다";
            throw std::invalid_argument(os.str());
        }

        code_ = cyclicWindow(sequence_, config_.offset, L);

        // 윈도우 -> 시작 인덱스. 중복은 등록하지 않는다.
        std::map<std::vector<int>, int> counts;
        for (int i = 0; i + n <= L; ++i)
        {
            const std::vector<int> w(code_.begin() + i, code_.begin() + i + n);
            auto it = windowToStart_.find(w);
            if (it == windowToStart_.end())
                windowToStart_.emplace(w, i);
            counts[w] += 1;
        }

        report_.length = L;
        report_.period = period;
        report_.windowCount = L - n + 1;
        report_.windowDistinct = static_cast<int>(counts.size());
        report_.windowMaxMultiplicity = 0;
        for (const auto &kv : counts)
            report_.windowMaxMultiplicity = std::max(report_.windowMaxMultiplicity, kv.second);
        report_.absolutelyDecodable = (report_.windowMaxMultiplicity == 1);

        // 중복 윈도우는 조회에서 빼야 조용한 오복원을 막을 수 있다.
        for (const auto &kv : counts)
        {
            if (kv.second > 1)
                windowToStart_.erase(kv.first);
        }

        int runs = 1;
        int runLen = 1;
        for (int i = 1; i < L; ++i)
        {
            if (code_[static_cast<size_t>(i)] == code_[static_cast<size_t>(i - 1)])
            {
                ++runLen;
            }
            else
            {
                ++runs;
                report_.runLengthHistogram[runLen] += 1;
                runLen = 1;
            }
        }
        report_.runLengthHistogram[runLen] += 1;
        report_.runCount = runs;
        report_.visibleEdges = runs - 1;
        report_.maxEdges = L - 1;

        for (const int s : code_)
            report_.symbolCounts[s] += 1;

        buildRuns();
    }

    // 위상 없이 컬러 스트라이프만 쓸 때, 카메라는 인접 동일색이 병합된 run 만 본다.
    // 따라서 디코딩 기준을 코드가 아니라 run 시퀀스로 다시 세워야 한다.
    void DeBruijnPattern::buildRuns()
    {
        const int L = config_.length;
        runs_.clear();
        int i = 0;
        while (i < L)
        {
            int j = i;
            while (j + 1 < L && code_[static_cast<size_t>(j + 1)] == code_[static_cast<size_t>(i)])
                ++j;
            runs_.push_back(Run{code_[static_cast<size_t>(i)], j - i + 1, i});
            i = j + 1;
        }

        // 유일해지는 최소 run 윈도우 길이를 찾는다. (symbol, width) 쌍을 함께 쓴다 —
        // 색만으로는 유일해지지 않는 경우가 많다.
        const int runCount = static_cast<int>(runs_.size());
        runWindow_ = 0;
        runWindowToStart_.clear();
        for (int n = 2; n <= runCount; ++n)
        {
            std::map<RunKey, int> counts;
            for (int start = 0; start + n <= runCount; ++start)
            {
                RunKey key;
                key.reserve(static_cast<size_t>(n));
                for (int t = 0; t < n; ++t)
                    key.emplace_back(runs_[static_cast<size_t>(start + t)].symbol,
                                     runs_[static_cast<size_t>(start + t)].width);
                counts[key] += 1;
            }
            const bool unique = std::all_of(counts.begin(), counts.end(),
                                            [](const auto &kv) { return kv.second == 1; });
            if (unique)
            {
                runWindow_ = n;
                for (int start = 0; start + n <= runCount; ++start)
                {
                    RunKey key;
                    key.reserve(static_cast<size_t>(n));
                    for (int t = 0; t < n; ++t)
                        key.emplace_back(runs_[static_cast<size_t>(start + t)].symbol,
                                         runs_[static_cast<size_t>(start + t)].width);
                    runWindowToStart_.emplace(std::move(key), start);
                }
                break;
            }
        }
    }

    int DeBruijnPattern::lookupRuns(const RunKey &window) const
    {
        const auto it = runWindowToStart_.find(window);
        return it == runWindowToStart_.end() ? -1 : it->second;
    }

    double DeBruijnPattern::boundaryU(int j) const
    {
        const int stripePx = resolvedStripePx();
        const int x0 = (config_.imageWidth - stripePx * config_.length) / 2;
        return static_cast<double>(x0 + j * stripePx) / static_cast<double>(config_.imageWidth);
    }

    int DeBruijnPattern::lookup(const std::vector<int> &window) const
    {
        const auto it = windowToStart_.find(window);
        return it == windowToStart_.end() ? -1 : it->second;
    }

    // ── 렌더링 ──────────────────────────────────────────────────────────────

    int DeBruijnPattern::resolvedStripePx() const
    {
        if (config_.stripePx > 0)
            return config_.stripePx;
        return config_.imageWidth / config_.length;
    }

    ImageVec3 DeBruijnPattern::renderImage() const
    {
        const Eigen::Index rows = config_.imageHeight;
        const Eigen::Index cols = config_.imageWidth;
        const int stripePx = resolvedStripePx();
        if (stripePx <= 0)
            throw std::invalid_argument("renderImage: 스트라이프 폭이 0 이다 (이미지가 너무 좁다)");

        const int used = stripePx * config_.length;
        const int x0 = (static_cast<int>(cols) - used) / 2;

        ImageVec3 image;
        image.x = Image::Zero(rows, cols);
        image.y = Image::Zero(rows, cols);
        image.z = Image::Zero(rows, cols);

        for (int i = 0; i < config_.length; ++i)
        {
            const Eigen::Vector3i rgb = config_.palette[static_cast<size_t>(code_[static_cast<size_t>(i)])];
            const int xa = std::max(0, x0 + i * stripePx);
            const int xb = std::min(static_cast<int>(cols), xa + stripePx);
            if (xb <= xa)
                continue;
            const Eigen::Index w = xb - xa;
            image.x.block(0, xa, rows, w).setConstant(static_cast<float>(rgb.x()));
            image.y.block(0, xa, rows, w).setConstant(static_cast<float>(rgb.y()));
            image.z.block(0, xa, rows, w).setConstant(static_cast<float>(rgb.z()));
        }
        return image;
    }

    std::string DeBruijnPattern::renderSvg() const
    {
        const double pitch = config_.pitchMm;
        const double height = config_.heightMm;
        const double totalWidth = pitch * config_.length;

        std::ostringstream os;
        os << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
           << "width=\"" << formatDouble(totalWidth) << "mm\" "
           << "height=\"" << formatDouble(height) << "mm\" "
           << "viewBox=\"0 0 " << formatDouble(totalWidth) << " " << formatDouble(height) << "\">\n";
        os << "  <rect x=\"0\" y=\"0\" width=\"" << formatDouble(totalWidth)
           << "\" height=\"" << formatDouble(height) << "\" fill=\"#000000\"/>\n";

        // 같은 색 연속 구간은 하나의 rect 로 병합해 벤더 쪽 도형 수를 줄인다.
        int i = 0;
        while (i < config_.length)
        {
            int j = i;
            while (j + 1 < config_.length && code_[static_cast<size_t>(j + 1)] == code_[static_cast<size_t>(i)])
                ++j;

            const Eigen::Vector3i rgb = config_.palette[static_cast<size_t>(code_[static_cast<size_t>(i)])];
            char color[8];
            std::snprintf(color, sizeof(color), "#%02X%02X%02X", rgb.x(), rgb.y(), rgb.z());

            os << "  <rect x=\"" << formatDouble(pitch * i) << "\" y=\"0\" width=\""
               << formatDouble(pitch * (j - i + 1)) << "\" height=\"" << formatDouble(height)
               << "\" fill=\"" << color << "\" shape-rendering=\"crispEdges\"/>\n";
            i = j + 1;
        }
        os << "</svg>";
        return os.str();
    }

    std::string DeBruijnPattern::toJson() const
    {
        const auto array = [](const std::vector<int> &values) {
            std::ostringstream os;
            os << "[";
            for (size_t i = 0; i < values.size(); ++i)
                os << (i ? ", " : "") << values[i];
            os << "]";
            return os.str();
        };

        std::ostringstream os;
        os << "{\n";
        os << "  \"mode\": \"palette\",\n";
        os << "  \"k\": " << config_.alphabet << ",\n";
        os << "  \"n\": " << config_.window << ",\n";
        os << "  \"decode_window\": " << config_.window << ",\n";
        os << "  \"logical_stripes\": " << sequence_.size() << ",\n";
        os << "  \"projected_stripes\": " << config_.length << ",\n";
        os << "  \"offset\": " << config_.offset << ",\n";
        os << "  \"base_sequence\": " << array(sequence_) << ",\n";
        os << "  \"sequence\": " << array(code_) << ",\n";
        os << "  \"palette\": [";
        for (int i = 0; i < config_.alphabet; ++i)
        {
            const Eigen::Vector3i &c = config_.palette[static_cast<size_t>(i)];
            os << (i ? ", " : "") << "[" << c.x() << ", " << c.y() << ", " << c.z() << "]";
        }
        os << "],\n";
        os << "  \"adjacency\": \""
           << (config_.noRepeat ? "no-repeat (인접 동일 심볼 금지, 모든 경계 가시)"
                                : "unconstrained (De Bruijn B(k,n), 인접 동일 심볼 허용)")
           << "\",\n";
        os << "  \"self_equalizing\": false,\n";
        os << "  \"pair_layout\": null,\n";
        os << "  \"image_size\": [" << config_.imageWidth << ", " << config_.imageHeight << "],\n";
        os << "  \"svg_pitch_mm\": " << formatDouble(config_.pitchMm) << ",\n";
        os << "  \"svg_height_mm\": " << formatDouble(config_.heightMm) << ",\n";
        os << "  \"verification\": {\n";
        os << "    \"window_count\": " << report_.windowCount << ",\n";
        os << "    \"window_distinct\": " << report_.windowDistinct << ",\n";
        os << "    \"window_max_multiplicity\": " << report_.windowMaxMultiplicity << ",\n";
        os << "    \"absolutely_decodable\": " << (report_.absolutelyDecodable ? "true" : "false") << ",\n";
        os << "    \"run_count\": " << report_.runCount << ",\n";
        os << "    \"visible_edges\": " << report_.visibleEdges << ",\n";
        os << "    \"max_edges\": " << report_.maxEdges << "\n";
        os << "  }\n";
        os << "}\n";
        return os.str();
    }

    ColorCodeInfo DeBruijnPattern::toColorCodeInfo() const
    {
        std::vector<Eigen::Vector3i> palette(config_.palette.begin(),
                                             config_.palette.begin() + config_.alphabet);
        // mode "palette" 는 ColorCoded 에서 팔레트 최근접 분류로 처리된다.
        return ColorCodeInfo::create(code_, palette, "palette", false, config_.window);
    }

} // namespace sl
