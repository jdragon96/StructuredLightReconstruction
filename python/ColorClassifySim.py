"""
컬러 스트라이프 분류기 비교 — HSV hue vs 2채널 색도 vs white 레퍼런스.

docs/DeBruijn/README.md 5절의 수치와 그림(fig5, fig6)을 생성한다.

  python3 python/ColorClassifySim.py

모델: 관측 = (음영 x 알베도 x 투사색) @ 누화행렬 + 환경광 + 가우시안 노이즈,
      이후 8비트 양자화. 절대 수치보다 방법 간 순위가 결론이다.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.dpi": 140,
    "font.size": 10,
})

BG, FG, DIM = "#1a1a1a", "#e8e8e8", "#8a8a8a"
OUT = "docs/DeBruijn/image"

PALETTE = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]], float)  # Green, Cyan, Blue
NAMES = ["Green", "Cyan", "Blue"]
DRAW = ["#00E526", "#00D9E5", "#3355FF"]
HUE_TARGET = np.array([120.0, 180.0, 240.0])
G_TARGET = np.array([1.0, 0.5, 0.0])

CROSSTALK = np.eye(3) * 0.88 + 0.06   # 베이어/스펙트럼 누화
NOISE = 0.03
AMBIENT = 0.03
ROWS = 4000


# ------------------------------------------------------------------ 분류기
def hue_deg(c):
    R, G, B = c[..., 0], c[..., 1], c[..., 2]
    return np.degrees(np.arctan2(np.sqrt(3) * (G - B), 2 * R - G - B)) % 360


def cls_hue(c):
    """정통 HSV hue. 분모 2R-G-B 때문에 R 채널이 계산에 들어온다."""
    d = np.abs((hue_deg(c)[..., None] - HUE_TARGET + 180) % 360 - 180)
    return np.argmin(d, -1)


def cls_gchroma(c):
    """2채널 색도 g = G/(G+B). R 을 아예 보지 않는다."""
    G, B = c[..., 1], c[..., 2]
    g = G / np.maximum(G + B, 1e-6)
    return np.argmin(np.abs(g[..., None] - G_TARGET), -1)


def cls_rgb(c):
    return np.argmin(((c[..., None, :] - PALETTE) ** 2).sum(-1), -1)


# ------------------------------------------------------------------ 시뮬레이션
def simulate(code, rows=ROWS, colored=True, red_ambient=0.0, seed=3):
    rng = np.random.default_rng(seed)
    amb = np.array([AMBIENT + red_ambient, AMBIENT, AMBIENT])
    n = len(code)
    acc = {"HSV hue": 0.0, "2채널 색도": 0.0, "RGB + white": 0.0}

    for _ in range(rows):
        x = np.arange(n)
        shading = 0.5 + 0.45 * np.sin(2 * np.pi * x / 97 + rng.uniform(0, 7))
        if colored:
            albedo = np.stack([
                0.35 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * x / rng.uniform(40, 90) + rng.uniform(0, 7)))
                for _ in range(3)], 1)
            edge = rng.integers(15, n - 15)
            albedo[edge:] *= rng.uniform(0.3, 1.0, 3)   # 급격한 표면 색 경계 1곳
        else:
            albedo = np.ones((n, 3)) * rng.uniform(0.3, 1.0)

        def capture(projected):
            v = (shading[:, None] * albedo * projected) @ CROSSTALK.T + amb
            v = v + rng.normal(0, NOISE, v.shape)
            return np.clip(np.round(v * 255) / 255, 0, None)

        obs = capture(PALETTE[code])
        white = capture(np.ones(3))
        ratio = obs / np.maximum(white, 1e-3)

        acc["HSV hue"] += (cls_hue(obs) == code).mean()
        acc["2채널 색도"] += (cls_gchroma(obs) == code).mean()
        acc["RGB + white"] += (cls_rgb(ratio) == code).mean()

    return {k: v / rows for k, v in acc.items()}


CONDITIONS = [
    ("무채색 표면", dict(colored=False)),
    ("유채색 표면", dict(colored=True)),
    ("유채색 + 붉은 환경광 0.10", dict(colored=True, red_ambient=0.10)),
    ("유채색 + 붉은 환경광 0.25", dict(colored=True, red_ambient=0.25)),
]


# ------------------------------------------------------------------ 그림
def fig_chromaticity():
    fig = plt.figure(figsize=(11, 4.4), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.15)

    # (a) hue 원 — 팔레트 위치와 결정 여유
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=DIM, linewidth=1))
    for h, col, nm in zip(HUE_TARGET, DRAW, NAMES):
        for lo, hi in [(h - 30, h + 30)]:
            ax.add_patch(Wedge((0, 0), 1.0, lo, hi, facecolor=col, alpha=0.13, edgecolor="none"))
        t = np.radians(h)
        ax.plot([0, np.cos(t)], [0, np.sin(t)], color=col, linewidth=2.4)
        ax.text(1.16 * np.cos(t), 1.16 * np.sin(t), f"{nm}\n{h:.0f}°",
                color=col, ha="center", va="center", fontsize=9.5)
    for b in (150, 210):
        t = np.radians(b)
        ax.plot([0, 1.02 * np.cos(t)], [0, 1.02 * np.sin(t)],
                color=FG, linewidth=1, linestyle="--", alpha=.55)
    ax.plot([0, 1], [0, 0], color="#7a3a3a", linewidth=2.0, linestyle=":")
    ax.text(1.16, 0, "R · 0°\n(미사용)", color="#b06060", ha="center", va="center", fontsize=9.5)
    ax.set_xlim(-1.55, 1.6)
    ax.set_ylim(-1.5, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("팔레트는 hue 원의 120°~240° 구간만 쓴다\n간격 60°, 결정 여유 ±30°",
                 color=FG, fontsize=11, pad=4)

    # (b) 1D 색도 축
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG)
    ax2.axhline(0, color=DIM, linewidth=1.2)
    for g, col, nm in zip(G_TARGET, DRAW, NAMES):
        ax2.plot([g], [0], marker="o", markersize=13, color=col, zorder=3)
        ax2.text(g, 0.30, nm, color=col, ha="center", fontsize=10)
        ax2.text(g, -0.34, f"g = {g:.1f}", color=DIM, ha="center", fontsize=9)
    for b in (0.25, 0.75):
        ax2.axvline(b, color=FG, linewidth=1, linestyle="--", alpha=.55, ymin=.3, ymax=.7)
        ax2.text(b, -0.62, "결정 경계", color=FG, ha="center", fontsize=8.5, alpha=.75)
    ax2.set_xlim(-0.16, 1.16)
    ax2.set_ylim(-0.85, 0.75)
    ax2.set_yticks([])
    ax2.set_xticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_title("2채널 색도  g = G / (G + B)  —  R 을 보지 않는 같은 판정\n"
                  "등간격 0 / 0.5 / 1, 여유 ±0.25", color=FG, fontsize=11, pad=4)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig5_chromaticity.png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"   {OUT}/fig5_chromaticity.png")


def fig_classifier(results):
    labels = [c[0] for c in CONDITIONS]
    methods = ["HSV hue", "2채널 색도", "RGB + white"]
    colors = ["#d9603a", "#4ec9a6", "#3a6ea5"]

    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(len(labels))
    h = 0.26
    for i, (m, col) in enumerate(zip(methods, colors)):
        vals = [100 * results[l][m] for l in labels]
        pos = y + (i - 1) * h
        ax.barh(pos, vals, height=h, color=col, edgecolor="none", label=m)
        for p, v in zip(pos, vals):
            ax.text(v + 0.7, p, f"{v:.1f}%", color=FG, va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=FG)
    ax.invert_yaxis()
    ax.set_xlim(55, 100)
    ax.set_xlabel("스트라이프당 원시 분류 정확도 (DP 이전)", color=FG)
    ax.tick_params(colors=FG, labelsize=9.5)
    for sp in ax.spines.values():
        sp.set_color(DIM)
    ax.legend(facecolor=BG, edgecolor="none", labelcolor=FG, fontsize=9.5,
              loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
    ax.set_title("붉은 환경광이 들어오면 HSV hue 만 무너진다 — 신호 없는 R 채널을 계산에 넣기 때문",
                 color=FG, fontsize=12, pad=34, loc="left")
    fig.savefig(f"{OUT}/fig6_classifier.png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"   {OUT}/fig6_classifier.png")


if __name__ == "__main__":
    code = np.array(json.load(open("dataset/debruijn/debruijn_pattern_info.json"))["sequence"])

    print("팔레트 색도")
    for nm, c in zip(NAMES, PALETTE):
        print(f"   {nm:6s} RGB{tuple(int(v) for v in c)}  hue {hue_deg(c[None])[0]:6.1f}°"
              f"  g {c[1] / (c[1] + c[2]):.2f}")
    print("   hue 간격 60° (R 을 쓰는 RGB/CMY 팔레트라면 120°)\n")

    results = {}
    print(f"분류 정확도 — {ROWS}행 x 72 스트라이프, 노이즈 σ={NOISE}")
    header = f"   {'조건':<26}" + "".join(f"{m:>14}" for m in ["HSV hue", "2채널 색도", "RGB + white"])
    print(header)
    for label, kw in CONDITIONS:
        r = simulate(code, **kw)
        results[label] = r
        print(f"   {label:<26}" + "".join(f"{100 * r[m]:12.2f}%" for m in
                                          ["HSV hue", "2채널 색도", "RGB + white"]))

    print("\n그림 생성:")
    fig_chromaticity()
    fig_classifier(results)
