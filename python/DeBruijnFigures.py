"""
docs/DeBruijn/README.md 에 들어가는 설명 그림 생성.

  python3 python/DeBruijnFigures.py
"""

from __future__ import annotations

import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from DeBruijnPattern import cyclic_window, de_bruijn, de_bruijn_no_repeat

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.dpi": 140,
    "font.size": 10,
})

BG = "#1a1a1a"
FG = "#e8e8e8"
DIM = "#8a8a8a"
RGB = [(0.0, 0.9, 0.1), (0.15, 0.35, 1.0), (0.0, 0.85, 0.9)]  # G, B, C
NAMES = ["Green", "Blue", "Cyan"]
OUT = "docs/DeBruijn/image"


def _fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("  ", os.path.join(OUT, name))


def stripe_bar(ax, code, y=0.0, h=1.0, lw=0.0):
    for i, s in enumerate(code):
        ax.add_patch(Rectangle((i, y), 1, h, facecolor=RGB[s], edgecolor="none", linewidth=lw))


# ---------------------------------------------------------------- fig 1
def fig_graph():
    """De Bruijn 그래프 B(3,2): 노드=심볼, 엣지=길이 2 윈도우, 오일러 순환=수열."""
    fig = _fig(9.5, 4.6)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    pos = {0: (0.0, 1.0), 1: (-0.87, -0.5), 2: (0.87, -0.5)}

    # 심볼 쌍 사이 엣지 (양방향)
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            ax.add_patch(FancyArrowPatch(
                pos[a], pos[b], connectionstyle="arc3,rad=0.22",
                arrowstyle="-|>", mutation_scale=13, shrinkA=16, shrinkB=16,
                color=DIM, linewidth=1.2, zorder=2))

    # 자기 자신으로 가는 엣지 (00, 11, 22 -> 인접 동일색 run 의 원인)
    for s_, (x, y) in pos.items():
        d = np.array([x, y]) / np.hypot(x, y)          # 삼각형 바깥 방향
        ax.add_patch(Arc(np.array([x, y]) + 0.36 * d, 0.46, 0.46,
                         theta1=0, theta2=340, color="#d98c3a", linewidth=1.8, zorder=2))

    for s_, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.24, facecolor=RGB[s_], edgecolor=FG,
                            linewidth=1.5, zorder=3))
        ax.text(x, y, str(s_), color="#101010", ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=4)
        d = np.array([x, y]) / np.hypot(x, y)
        lx, ly = np.array([x, y]) - 0.46 * d           # 이름은 삼각형 안쪽
        ax.text(lx, ly, NAMES[s_], color=FG, ha="center", va="center",
                fontsize=10, zorder=4,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))

    ax.set_xlim(-2.0, 3.9)
    ax.set_ylim(-1.5, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")

    txt = (
        "노드 = 길이 (n-1) 단어      엣지 = 길이 n 윈도우\n"
        "엣지를 정확히 한 번씩 지나는 오일러 순환 = De Bruijn 수열\n\n"
        "k=3, n=2  →  엣지 3² = 9개  →  수열 길이 9\n"
        "k=3, n=4  →  엣지 3⁴ = 81개  →  수열 길이 81\n\n"
        "주황색 자기 루프(00·11·22) = 같은 색이 연달아 나오는 경우.\n경계가 사라지는 run 은 여기서 생긴다."
    )
    ax.text(1.55, 0.95, txt, color=FG, fontsize=10.5, va="top", ha="left", linespacing=1.6)
    fig.suptitle("De Bruijn 그래프 — 왜 길이 kⁿ 안에 모든 윈도우가 한 번씩 들어가는가",
                 color=FG, fontsize=13, y=1.0)
    _save(fig, "fig1_debruijn_graph.png")


# ---------------------------------------------------------------- fig 2
def candidates(code, n, period):
    """각 윈도우 위치가 몇 개의 후보 위치로 해석되는지."""
    win = ["".join(map(str, code[i:i + n])) for i in range(len(code) - n + 1)]
    c = Counter(win)
    return [c[w] for w in win]


def fig_uniqueness():
    """n=3(주기 27 반복) vs n=4(주기 81) 의 위치 유일성 비교."""
    s3, s4 = de_bruijn(3, 3), de_bruijn(3, 4)
    c3 = cyclic_window(s3, 17, 72)
    c4 = cyclic_window(s4, 6, 72)

    fig = _fig(11, 5.4)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.5, 1.0, 1.5], hspace=0.75)

    for row, (code, seq, n, off, title) in enumerate([
        (c3, s3, 3, 17, "B(3,3) · 주기 27 · offset 17 — 27칸마다 패턴이 그대로 반복"),
        (c4, s4, 4, 6, "B(3,4) · 주기 81 · offset 6 — 72 < 81 이라 반복 없음"),
    ]):
        ax = fig.add_subplot(gs[row * 2])
        ax.set_facecolor(BG)
        stripe_bar(ax, code)
        ax.set_xlim(0, 72)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, color=FG, fontsize=11, pad=6, loc="left")
        if n == 3:  # 주기 경계 표시
            for x in range(27 - 17, 72, 27):
                ax.axvline(x, color="#ff5555", linewidth=1.6, linestyle="--")

        axc = fig.add_subplot(gs[row * 2 + 1])
        axc.set_facecolor(BG)
        cand = candidates(code, n, len(seq))
        axc.bar(np.arange(len(cand)) + 0.5, cand, width=1.0,
                color=("#ff5555" if n == 3 else "#4ec9a6"), edgecolor="none")
        axc.set_xlim(0, 72)
        axc.set_ylim(0, 4.3)
        axc.set_yticks([1, 2, 3])
        axc.set_ylabel("후보 위치 수", color=FG, fontsize=9)
        axc.axhline(1, color=DIM, linewidth=0.8, linestyle=":")
        for sp in axc.spines.values():
            sp.set_color(DIM)
        axc.tick_params(colors=FG, labelsize=8)
        if row == 1:
            axc.set_xlabel("스트라이프 인덱스", color=FG, fontsize=9)
        msg = (f"n={n} 윈도우 → 후보 3개 · 절대 위치 결정 불가 (mod 27)" if n == 3
               else f"n={n} 윈도우 → 전 구간 후보 1개 · 절대 위치 결정")
        axc.text(0.012, 0.94, msg, transform=axc.transAxes, color=FG,
                 fontsize=9.5, ha="left", va="top")

    fig.suptitle("패턴 길이(72) > 수열 주기 이면 윈도우를 늘려도 모호성이 남는다",
                 color=FG, fontsize=13, y=0.99)
    _save(fig, "fig2_window_uniqueness.png")


# ---------------------------------------------------------------- fig 3
def fig_offset_scan():
    """81개 offset 중 보이는 경계 수가 최대인 지점 선택."""
    seq = de_bruijn(3, 4)
    edges, imbal = [], []
    for off in range(81):
        code = cyclic_window(seq, off, 72)
        e = sum(1 for i in range(71) if code[i] != code[i + 1])
        cnt = Counter(code)
        edges.append(e)
        imbal.append(max(cnt.values()) - min(cnt.values()))

    fig = _fig(11, 3.6)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.bar(range(81), edges, color="#3a6ea5", edgecolor="none", width=0.85)
    best = 6
    ax.bar([best], [edges[best]], color="#4ec9a6", edgecolor="none", width=0.85)
    ax.axhline(np.mean(edges), color="#d98c3a", linewidth=1.2, linestyle="--",
               label=f"평균 {np.mean(edges):.1f}")
    ax.annotate(f"offset {best}\n경계 {edges[best]}/71, 불균형 {imbal[best]}",
                xy=(best, edges[best]), xytext=(best + 5, edges[best] + 5),
                color=FG, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=FG, linewidth=1.2))
    ax.set_ylim(40, max(edges) + 9)
    ax.set_yticks(range(40, max(edges) + 9, 2))
    ax.set_xlim(-1, 81)
    ax.set_xlabel("offset (순환 수열 시작 위치)", color=FG)
    ax.set_ylabel("보이는 경계 수 / 71", color=FG)
    for sp in ax.spines.values():
        sp.set_color(DIM)
    ax.tick_params(colors=FG, labelsize=9)
    leg = ax.legend(facecolor=BG, edgecolor=DIM, labelcolor=FG, fontsize=9, loc="upper right")
    leg.get_frame().set_alpha(0.9)
    ax.set_title("offset 선택 — 81개 전부 절대 디코딩 가능하므로 경계 가시성으로 고른다",
                 color=FG, fontsize=12.5, pad=10, loc="left")
    _save(fig, "fig3_offset_scan.png")


# ---------------------------------------------------------------- fig 4
def fig_decode():
    """최종 패턴 + 4-윈도우 조회로 절대 위치를 얻는 과정."""
    seq = de_bruijn(3, 4)
    code = cyclic_window(seq, 6, 72)

    fig = _fig(11, 3.9)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    stripe_bar(ax, code, y=0.55, h=0.45)

    # 같은 색이 붙어 경계가 사라지는 구간 표시
    for i in range(71):
        if code[i] == code[i + 1]:
            ax.plot([i + 1], [0.52], marker="^", color="#ff5555", markersize=5, clip_on=False)

    i0 = 30
    ax.add_patch(Rectangle((i0, 0.55), 4, 0.45, facecolor="none",
                           edgecolor="#ffffff", linewidth=2.2, zorder=5))
    w = "".join(map(str, code[i0:i0 + 4]))
    ax.annotate("", xy=(i0 + 2, 0.53), xytext=(i0 + 2, 0.34),
                arrowprops=dict(arrowstyle="<-", color=FG, linewidth=1.4))
    ax.text(i0 + 2, 0.28, f"윈도우 [{w}]  →  LUT 조회  →  위치 {i0}  (유일)",
            color=FG, fontsize=11, ha="center", va="top")

    ax.text(0, 1.14, "B(3,4) offset 6 · 72 스트라이프 · Green=0 Blue=1 Cyan=2",
            color=FG, fontsize=11.5, ha="left", va="bottom")
    ax.text(72, 0.46, "▲ 인접 동일색 — 경계 소실 20곳", color="#ff5555",
            fontsize=9, ha="right", va="top")
    ax.set_xlim(0, 72)
    ax.set_ylim(0.0, 1.12)
    ax.axis("off")
    _save(fig, "fig4_decode.png")


# ---------------------------------------------------------------- fig 7
def fig_norepeat():
    """인접 동일색 허용(B) vs 금지(B\') — 경계 가시성과 윈도우 길이의 맞교환."""
    a = cyclic_window(de_bruijn(3, 4), 6, 72)
    b = cyclic_window(de_bruijn_no_repeat(3, 6), 50, 72)

    fig = _fig(11, 4.5)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    for row, (code, n, title, sub) in enumerate([
        (a, 4, "B(3,4) offset 6 — 인접 동일색 허용",
         "보이는 경계 51/71 · 디코딩 윈도우 4 스트라이프 · 심볼 22/26/24"),
        (b, 6, "B'(3,6) offset 50 — 인접 동일색 금지",
         "보이는 경계 71/71 · 디코딩 윈도우 6 스트라이프 · 심볼 24/24/24"),
    ]):
        y = 1.02 - row * 0.62
        stripe_bar(ax, code, y=y, h=0.30)
        ax.text(0, y + 0.36, title, color=FG, fontsize=11.5, va="bottom")
        ax.text(0, y - 0.10, sub, color=DIM, fontsize=9.5, va="top")

        for i in range(71):                       # 경계 소실 표시
            if code[i] == code[i + 1]:
                ax.plot([i + 1], [y - 0.03], marker="^", color="#ff5555",
                        markersize=5, clip_on=False)

        i0 = 30                                   # 디코딩 윈도우 폭을 실제 크기로
        ax.add_patch(Rectangle((i0, y), n, 0.30, facecolor="none",
                               edgecolor="#ffffff", linewidth=2.0, zorder=5))
        ax.text(i0 + n / 2, y + 0.33, f"윈도우 {n}", color=FG, fontsize=9,
                ha="center", va="bottom")

    ax.text(72, 1.38, "▲ 경계 소실 20곳", color="#ff5555",
            fontsize=9, ha="right", va="bottom")
    ax.set_xlim(0, 72)
    ax.set_ylim(0.20, 1.52)
    ax.axis("off")
    fig.suptitle("경계를 전부 살리려면 디코딩 윈도우가 4 -> 6 스트라이프로 길어진다",
                 color=FG, fontsize=13, y=1.0)
    _save(fig, "fig7_norepeat.png")


if __name__ == "__main__":
    print("생성:")
    fig_graph()
    fig_uniqueness()
    fig_offset_scan()
    fig_decode()
    fig_norepeat()
