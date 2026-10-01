"""
De Bruijn 컬러 스트라이프 패턴 생성기.

k개 색 알파벳 위의 De Bruijn 수열 B(k, n)을 만들고, 그 순환 수열에서
offset부터 length개를 잘라 세로 스트라이프 패턴으로 렌더한다.

기본값은 패턴 글라스용 스펙:
  k=3 (Green / Blue / Cyan), n=4, 주기 81, offset 6, 길이 72
  -> 69개 4-윈도우가 전부 유일해 72 스트라이프 절대 디코딩 가능

출력:
  <out>/debruijn_k{k}n{n}_{length}.svg   벡터 (마스크 제작용)
  <out>/debruijn_k{k}n{n}_{length}.png   프로젝터 해상도 래스터
  <out>/debruijn_pattern_info.json       시퀀스/팔레트/윈도우 매핑

사용:
  python3 python/DeBruijnPattern.py
  python3 python/DeBruijnPattern.py --n 3 --offset 17  # 구 스펙 (위치는 mod 27)
  python3 python/DeBruijnPattern.py --pitch-mm 0.25 --height-mm 12
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from itertools import product

import numpy as np
from PIL import Image

# symbol -> (name, RGB). 인덱스가 곧 De Bruijn 알파벳 심볼.
DEFAULT_PALETTE = [
    ("Green", (0, 255, 0)),
    ("Blue", (0, 0, 255)),
    ("Cyan", (0, 255, 255)),
]


def de_bruijn(k: int, n: int) -> list[int]:
    """표준 Lyndon word 기반 De Bruijn 수열 B(k, n). 길이 k**n, 순환 수열."""
    a = [0] * (k * n)
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq


def de_bruijn_no_repeat(k: int, n: int) -> list[int]:
    """인접 동일 심볼이 없는 순환 De Bruijn 수열. 길이 k(k-1)**(n-1).

    제약 De Bruijn 그래프 — 노드는 길이 (n-1) 이면서 인접 심볼이 서로 다른 단어,
    엣지는 길이 n 윈도우 — 위의 오일러 순환(Hierholzer)으로 구한다. 모든 노드의
    입·출차수가 (k-1) 로 같아 순환이 존재하고, 엣지를 한 번씩 지나므로 유효한
    윈도우 전부가 정확히 한 번씩 나타난다.
    """
    if k < 3:
        raise ValueError("인접 상이 제약은 k >= 3 에서만 의미가 있다")
    nodes = [w for w in product(range(k), repeat=n - 1)
             if all(w[i] != w[i + 1] for i in range(n - 2))]
    out = {w: [b for b in range(k) if b != w[-1]] for w in nodes}

    circuit: list[tuple[int, ...]] = []
    stack = [nodes[0]]
    while stack:
        v = stack[-1]
        if out[v]:
            stack.append(v[1:] + (out[v].pop(),))
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    return [w[-1] for w in circuit[1:]]


def cyclic_window(seq: list[int], offset: int, length: int) -> list[int]:
    """순환 수열 seq에서 offset부터 length개를 잘라낸다."""
    p = len(seq)
    return [seq[(offset + i) % p] for i in range(length)]


def analyze(code: list[int], period: int, n: int) -> dict:
    """윈도우 유일성 / run 구조 / 심볼 균형 검증."""
    N = len(code)
    report: dict = {"length": N, "period": period, "symbol_counts": dict(Counter(code))}

    windows = ["".join(map(str, code[i : i + n])) for i in range(N - n + 1)]
    mult = Counter(windows)
    report["window_n"] = n
    report["window_count"] = len(windows)
    report["window_distinct"] = len(mult)
    report["window_max_multiplicity"] = max(mult.values()) if mult else 0
    report["absolutely_decodable"] = report["window_max_multiplicity"] == 1

    runs: list[tuple[int, int]] = []
    cur, ln = code[0], 1
    for s in code[1:]:
        if s == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur, ln = s, 1
    runs.append((cur, ln))
    report["run_count"] = len(runs)
    report["visible_edges"] = len(runs) - 1
    report["max_edges"] = N - 1
    report["run_length_hist"] = dict(sorted(Counter(l for _, l in runs).items()))
    return report


def render_png(code: list[int], palette: list[tuple[str, tuple[int, int, int]]],
               width: int, height: int, stripe_px: int | None) -> Image.Image:
    """스트라이프 폭을 정수로 고정하고 가운데 정렬. 여백은 검정."""
    n = len(code)
    if stripe_px is None:
        stripe_px = width // n
    used = stripe_px * n
    x0 = (width - used) // 2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i, sym in enumerate(code):
        xa = x0 + i * stripe_px
        img[:, xa : xa + stripe_px] = palette[sym][1]
    return Image.fromarray(img)


def render_svg(code: list[int], palette: list[tuple[str, tuple[int, int, int]]],
               pitch_mm: float, height_mm: float) -> str:
    """마스크 제작용 벡터. 좌표계 단위 = mm, 스트라이프는 인접 무간극."""
    n = len(code)
    total_w = pitch_mm * n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{total_w}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {total_w:g} {height_mm:g}">',
        f'  <rect x="0" y="0" width="{total_w:g}" height="{height_mm:g}" fill="#000000"/>',
    ]
    # 같은 색 연속 구간은 하나의 rect로 병합 (벤더 쪽 도형 수 감소)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and code[j + 1] == code[i]:
            j += 1
        r, g, b = palette[code[i]][1]
        x = pitch_mm * i
        w = pitch_mm * (j - i + 1)
        parts.append(
            f'  <rect x="{x:g}" y="0" width="{w:g}" height="{height_mm:g}" '
            f'fill="#{r:02X}{g:02X}{b:02X}" shape-rendering="crispEdges"/>'
        )
        i = j + 1
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--length", type=int, default=72, help="실제 제작할 스트라이프 개수")
    ap.add_argument("--offset", type=int, default=6, help="순환 수열 시작 위치")
    ap.add_argument("--no-repeat", action="store_true",
                    help="인접 동일색 금지 수열 사용 (주기 k(k-1)^(n-1), 모든 경계가 보임)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--stripe-px", type=int, default=None, help="미지정 시 width//length")
    ap.add_argument("--pitch-mm", type=float, default=0.5, help="SVG 스트라이프 피치")
    ap.add_argument("--height-mm", type=float, default=24.0)
    ap.add_argument("--out", default="dataset/debruijn")
    args = ap.parse_args()

    palette = DEFAULT_PALETTE[: args.k]
    if len(palette) < args.k:
        raise SystemExit(f"팔레트에 색이 {len(palette)}개뿐인데 k={args.k}")

    seq = (de_bruijn_no_repeat(args.k, args.n) if args.no_repeat
           else de_bruijn(args.k, args.n))
    code = cyclic_window(seq, args.offset, args.length)
    report = analyze(code, len(seq), args.n)

    os.makedirs(args.out, exist_ok=True)
    stem = f"debruijn_k{args.k}n{args.n}{'_norep' if args.no_repeat else ''}_{args.length}"

    png = render_png(code, palette, args.width, args.height, args.stripe_px)
    png.save(os.path.join(args.out, f"{stem}.png"))

    svg = render_svg(code, palette, args.pitch_mm, args.height_mm)
    with open(os.path.join(args.out, f"{stem}.svg"), "w") as f:
        f.write(svg)

    # ColorCoded 파이프라인이 읽는 형식 + De Bruijn 메타데이터
    info = {
        "mode": "palette",
        "k": args.k,
        "n": args.n,
        "decode_window": args.n,
        "logical_stripes": len(seq),
        "projected_stripes": args.length,
        "offset": args.offset,
        "base_sequence": seq,
        "sequence": code,
        "palette": [list(c) for _, c in palette],
        "palette_names": [name for name, _ in palette],
        "adjacency": ("no-repeat (인접 동일 심볼 금지, 모든 경계 가시)" if args.no_repeat
                      else "unconstrained (De Bruijn B(k,n), 인접 동일 심볼 허용)"),
        "self_equalizing": False,
        "pair_layout": None,
        "image_size": [args.width, args.height],
        "svg_pitch_mm": args.pitch_mm,
        "svg_height_mm": args.height_mm,
        "verification": report,
    }
    with open(os.path.join(args.out, "debruijn_pattern_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    stripe_px = args.stripe_px if args.stripe_px else args.width // args.length
    kind = "B'" if args.no_repeat else "B"
    print(f"base {kind}({args.k},{args.n})  period={len(seq)}  {''.join(map(str, seq))}")
    print(f"code  offset={args.offset} length={args.length}")
    print(f"      {''.join(map(str, code))}")
    print()
    print(f"윈도우 n={args.n}: {report['window_count']}개 중 distinct {report['window_distinct']}, "
          f"최대 중복 {report['window_max_multiplicity']}회 "
          f"-> 절대 디코딩 {'가능' if report['absolutely_decodable'] else '불가 (위치는 mod %d)' % len(seq)}")
    print(f"run: {report['run_count']}개, 보이는 경계 {report['visible_edges']}/{report['max_edges']}, "
          f"run 길이 분포 {report['run_length_hist']}")
    print(f"심볼 균형: {report['symbol_counts']}")
    print()
    print(f"PNG {args.width}x{args.height}, 스트라이프 {stripe_px}px, "
          f"사용폭 {stripe_px * args.length}px, 좌우 여백 {(args.width - stripe_px * args.length) // 2}px")
    print(f"SVG {args.pitch_mm * args.length:g} x {args.height_mm:g} mm (피치 {args.pitch_mm}mm)")
    print(f"-> {args.out}/{stem}.png, {stem}.svg, debruijn_pattern_info.json")


if __name__ == "__main__":
    main()
