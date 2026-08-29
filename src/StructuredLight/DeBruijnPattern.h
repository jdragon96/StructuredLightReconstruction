#pragma once

// DeBruijnPattern.h
//
// De Bruijn 컬러 스트라이프 패턴 생성기.
// C++/Eigen port of python/DeBruijnPattern.py.
//
// k개 색 알파벳 위의 De Bruijn 수열 B(k, n)을 만들고, 그 순환 수열에서 offset부터
// length개를 잘라 세로 스트라이프 패턴으로 렌더한다. 길이 n인 모든 부분열이 정확히
// 한 번씩 나타나므로, 연속된 n개 스트라이프의 색 나열이 곧 절대 위치의 주소가 된다.
//
// 설계 근거와 파라미터 결정 과정은 docs/DeBruijn/README.md 참고.
//
// 유일성 조건:
//   길이 L 패턴에서 잘라낼 수 있는 길이 n 윈도우는 L - n + 1개이고,
//   이것이 전부 서로 달라야 하므로  L - n + 1 <= (수열 주기).
//
//   제약 없음 : 주기 k^n            -> k=3, L=72 이면 n=4 가 최소 (69 <= 81)
//   인접 상이 : 주기 k(k-1)^(n-1)   -> k=3, L=72 이면 n=6 이 최소 (67 <= 96)
//
// 현재 제작 스펙: k=3 (Green/Blue/Cyan), n=4, 주기 81, offset 6, 길이 72.
//
// ColorCodeInfo 로 변환하면 ColorCoded.h 의 복원 파이프라인에 그대로 들어간다.
//
// 구현은 DeBruijnPattern.cpp 에 분리되어 있다.

#include "StructuredLight/ColorCoded.h"

#include <map>
#include <string>
#include <vector>

namespace sl
{

    struct DeBruijnConfig
    {
        int alphabet = 3;  // k — 색 개수
        int window = 4;    // n — 디코딩 윈도우 길이
        int length = 72;   // L — 실제 제작할 스트라이프 개수
        int offset = 6;    // 순환 수열에서 잘라낼 시작 위치

        // true 면 인접 동일 심볼이 없는 수열을 쓴다(모든 경계가 보이지만 n 이 커진다).
        bool noRepeat = false;

        // 심볼 -> RGB(0~255). 기본값은 Green / Blue / Cyan.
        // R 채널을 쓰지 않는 것은 표면 반사·색수차·베이어 누화를 피하려는 선택이다.
        std::vector<Eigen::Vector3i> palette = {
            Eigen::Vector3i(0, 255, 0),
            Eigen::Vector3i(0, 0, 255),
            Eigen::Vector3i(0, 255, 255),
        };

        // 래스터 렌더 파라미터
        int imageWidth = 1280;
        int imageHeight = 720;
        int stripePx = 0; // 0 이면 imageWidth / length (내림), 나머지는 좌우 여백

        // 벡터(SVG) 렌더 파라미터 — 마스크 제작용 물리 치수
        double pitchMm = 0.5;
        double heightMm = 24.0;
    };

    // 생성된 코드의 검증 결과. 제작 전 확인용이다.
    struct DeBruijnReport
    {
        int length = 0; // 제작 코드 길이
        int period = 0; // 기저 수열 주기

        int windowCount = 0;          // L - n + 1
        int windowDistinct = 0;       // 서로 다른 윈도우 개수
        int windowMaxMultiplicity = 0;// 같은 윈도우가 최대 몇 번 나오는가
        bool absolutelyDecodable = false; // 최대 중복이 1 인가

        int runCount = 0;    // 같은 색이 이어지는 구간 개수
        int visibleEdges = 0;// 물리적으로 보이는 경계 수 (= runCount - 1)
        int maxEdges = 0;    // L - 1

        std::map<int, int> runLengthHistogram; // run 길이 -> 개수
        std::map<int, int> symbolCounts;       // 심볼 -> 사용 횟수

        std::string summary() const; // 한 줄 요약 문자열
    };

    class DeBruijnPattern
    {
    public:
        // config 검증 -> 수열 생성 -> 코드 절단 -> 리포트 계산까지 수행한다.
        // 유일성 조건을 만족하지 못하면 std::invalid_argument 를 던진다.
        explicit DeBruijnPattern(DeBruijnConfig config);

        const DeBruijnConfig &config() const { return config_; }

        // 기저 순환 수열. 길이는 k^n 또는 인접 상이 제약 시 k(k-1)^(n-1).
        const std::vector<int> &baseSequence() const { return sequence_; }

        // 실제 제작할 코드 (config.length 개).
        const std::vector<int> &code() const { return code_; }

        const DeBruijnReport &report() const { return report_; }

        // 길이 n 윈도우 -> 시작 인덱스. 중복되는 윈도우는 등록하지 않는다.
        const std::map<std::vector<int>, int> &windowToStart() const { return windowToStart_; }

        // 관측한 라벨 윈도우로 절대 스트라이프 인덱스를 찾는다. 실패하면 -1.
        // 주의: 완전 De Bruijn 수열은 코드워드 간 Hamming 거리가 1 이므로
        // 오분류 하나가 다른 유효 위치로 조용히 매핑될 수 있다. 반드시 이웃
        // 윈도우와의 문맥 일관성 검사를 함께 써야 한다 (docs/DeBruijn 6.1절).
        int lookup(const std::vector<int> &window) const;

        // ── 렌더링 ──────────────────────────────────────────────────────────
        // 스트라이프 폭을 정수로 고정하고 가운데 정렬. 여백은 검정. (x,y,z)=(R,G,B), 0~255.
        ImageVec3 renderImage() const;

        // 마스크 제작용 벡터. 단위 mm, 같은 색 연속 구간은 하나의 rect 로 병합한다.
        std::string renderSvg() const;

        // color_pattern_info.json 과 호환되는 JSON 문자열.
        std::string toJson() const;

        // ColorCoded 복원 파이프라인에 넘길 수 있는 형태로 변환한다.
        ColorCodeInfo toColorCodeInfo() const;

        // ── 수열 생성 (독립적으로도 쓸 수 있다) ─────────────────────────────
        // 표준 Lyndon word 기반 De Bruijn 수열 B(k, n). 길이 k^n, 순환 수열.
        static std::vector<int> deBruijn(int k, int n);

        // 인접 동일 심볼이 없는 순환 수열. 길이 k(k-1)^(n-1).
        // 제약 De Bruijn 그래프(자기 루프 제거) 위의 오일러 순환을 Hierholzer 로 구한다.
        static std::vector<int> deBruijnNoRepeat(int k, int n);

        // 순환 수열에서 offset 부터 length 개를 잘라낸다.
        static std::vector<int> cyclicWindow(const std::vector<int> &sequence, int offset, int length);

        // 주어진 (k, L, noRepeat) 에서 유일성을 만족하는 최소 윈도우 길이. 없으면 0.
        static int minimumWindow(int k, int length, bool noRepeat);

    private:
        DeBruijnConfig config_;
        std::vector<int> sequence_; // 기저 순환 수열
        std::vector<int> code_;     // 제작 코드
        DeBruijnReport report_;
        std::map<std::vector<int>, int> windowToStart_;

        int resolvedStripePx() const;
    };

} // namespace sl
