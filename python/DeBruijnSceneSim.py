"""
유채색·텍스처 표면에서 De Bruijn 컬러 스트라이프가 얼마나 복원되는지 시뮬레이션.

카메라 캡처를 합성하고 -> 색 분류 -> 행 단위 Viterbi DP -> 스트라이프 인덱스 복원
까지 전 과정을 돌려서, 표면 색이 복원에 어떤 영향을 주는지 정량적으로 본다.

사용 예:
  python3 python/DeBruijnSceneSim.py                        # 기본: 텍스처 씬, white 사용
  python3 python/DeBruijnSceneSim.py --no-white             # white 레퍼런스 없이
  python3 python/DeBruijnSceneSim.py --scene macbeth        # 컬러차트 24색
  python3 python/DeBruijnSceneSim.py --scene macbeth --no-white --noise 0.05
  python3 python/DeBruijnSceneSim.py --margin-map           # 복원 가능성 이론 지도

핵심 결론(5절과 연결):
  white 레퍼런스가 있으면 알베도가 수식상 정확히 소거된다. 남는 문제는 편향이
  아니라 SNR — G 또는 B 반사율이 0 에 가까운 표면은 신호 자체가 없어 못 푼다.
  white 가 없으면 표면의 G:B 반사율 비가 판정 변수를 직접 밀어버린다.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "font.size": 9.5,
})

BG, FG, DIM = "#1a1a1a", "#e8e8e8", "#8a8a8a"
PALETTE = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]], float)   # Green, Cyan, Blue
PAL_NAMES = ["Green", "Cyan", "Blue"]

# 구강 내 대표 표면. 반사율 근사값 (0~1).
# 흥미롭게도 구강 표면 대부분은 G:B 가 균형에 가깝다 — 색도 붕괴보다
# 밝기(혈액·그림자)와 표면하 산란이 실제 위험이다.
DENTAL = [
    ("Enamel", (0.85, 0.85, 0.82)),      # 에나멜 (반투명)
    ("Dentin", (0.82, 0.72, 0.55)),      # 상아질 (노란기)
    ("Gingiva", (0.75, 0.42, 0.42)),     # 치은
    ("Blood", (0.45, 0.10, 0.10)),       # 혈액
    ("Metal", (0.55, 0.55, 0.58)),       # 금속 수복물
    ("Composite", (0.80, 0.76, 0.70)),   # 컴포지트
]

# X-Rite ColorChecker sRGB 대표값
MACBETH = [
    ("Dark skin", (115, 82, 68)), ("Light skin", (194, 150, 130)),
    ("Blue sky", (98, 122, 157)), ("Foliage", (87, 108, 67)),
    ("Blue flower", (133, 128, 177)), ("Bluish green", (103, 189, 170)),
    ("Orange", (214, 126, 44)), ("Purplish blue", (80, 91, 166)),
    ("Moderate red", (193, 90, 99)), ("Purple", (94, 60, 108)),
    ("Yellow green", (157, 188, 64)), ("Orange yellow", (224, 163, 46)),
    ("Blue", (56, 61, 150)), ("Green", (70, 148, 73)),
    ("Red", (175, 54, 60)), ("Yellow", (231, 199, 31)),
    ("Magenta", (187, 86, 149)), ("Cyan", (8, 133, 161)),
    ("White", (243, 243, 242)), ("Neutral 8", (200, 200, 200)),
    ("Neutral 6.5", (160, 160, 160)), ("Neutral 5", (122, 122, 122)),
    ("Neutral 3.5", (85, 85, 85)), ("Black", (52, 52, 52)),
]


# ---------------------------------------------------------------- 이론
def gb_margin(albedo):
    """알베도에서 Cyan 이 Green/Blue 와 떨어져 있는 거리 (2채널 색도 기준).

    관측 색도는 Green -> 1, Blue -> 0, Cyan -> ag/(ag+ab) 로 간다.
    셋이 구분되려면 Cyan 이 0 과 1 양쪽에서 충분히 떨어져야 하고,
    그 값은 오직 알베도의 G:B 비에만 의존한다 (밝기·R 성분과 무관).
    """
    a = np.asarray(albedo, float)
    ag, ab = a[..., 1], a[..., 2]
    gc = ag / np.maximum(ag + ab, 1e-9)
    return np.minimum(gc, 1.0 - gc)          # 이상적일 때 0.5


# ---------------------------------------------------------------- 씬
def make_scene(kind, H, W, rng):
    """(albedo HxWx3, shading HxW, 라벨용 패치 정보) 반환."""
    yy, xx = np.mgrid[0:H, 0:W]
    patches = None

    if kind in ("macbeth", "dental"):
        albedo = np.zeros((H, W, 3))
        table = MACBETH if kind == "macbeth" else [(n, tuple(int(v * 255) for v in c))
                                                   for n, c in DENTAL]
        cols, rowsn = (6, 4) if kind == "macbeth" else (3, 2)
        patches = []
        for i, (name, rgb) in enumerate(table):
            r, c = divmod(i, cols)
            y0, y1 = int(r * H / rowsn), int((r + 1) * H / rowsn)
            x0, x1 = int(c * W / cols), int((c + 1) * W / cols)
            albedo[y0:y1, x0:x1] = np.array(rgb) / 255.0
            patches.append((name, np.array(rgb) / 255.0, (y0, y1, x0, x1)))
    else:
        # 저주파 유채색 얼룩 + 고주파 텍스처 + 채도 높은 색 패치
        albedo = np.stack([
            0.45 + 0.35 * np.sin(2 * np.pi * xx / (W / rng.uniform(1.5, 3.5)) + rng.uniform(0, 7))
                 * np.cos(2 * np.pi * yy / (H / rng.uniform(1.0, 2.5)) + rng.uniform(0, 7))
            for _ in range(3)], -1)
        albedo += 0.06 * rng.standard_normal((H, W, 1))          # 무채색 미세 텍스처
        for rgb, (fy, fx, fh, fw) in [((0.85, 0.15, 0.10), (0.08, 0.06, 0.30, 0.18)),
                                      ((0.95, 0.85, 0.12), (0.55, 0.14, 0.34, 0.16)),
                                      ((0.10, 0.55, 0.95), (0.15, 0.55, 0.40, 0.20)),
                                      ((0.10, 0.10, 0.10), (0.60, 0.62, 0.30, 0.22))]:
            y0, y1 = int(fy * H), int((fy + fh) * H)
            x0, x1 = int(fx * W), int((fx + fw) * W)
            albedo[y0:y1, x0:x1] = rgb
        albedo = np.clip(albedo, 0.02, 1.0)

    # 완만한 음영 + 구형 융기 하나
    bump = np.exp(-(((xx - W * 0.62) / (W * 0.22)) ** 2 + ((yy - H * 0.5) / (H * 0.42)) ** 2))
    shading = np.clip(0.45 + 0.4 * (1 - yy / H) + 0.35 * bump, 0.12, 1.0)
    return albedo, shading, bump, patches


def stripe_truth(H, W, n_stripes, bump):
    """카메라 x -> 프로젝터 스트라이프 인덱스. 융기가 시차를 만든다."""
    xx = np.mgrid[0:H, 0:W][1]
    s = xx / W * n_stripes + 6.0 * bump          # 최대 6 스트라이프 시차
    return s


def subsurface_blur(pattern, sigma_px):
    """표면하 산란 — 채널별 가우시안 확산.

    장파장일수록 조직에 깊이 침투해 옆으로 더 퍼진다(에나멜에서 Green > Blue).
    중요한 성질: white 프레임은 평탄한 장이라 blur 를 먹여도 그대로다.
    따라서 **white 정규화로 이 왜곡은 전혀 제거되지 않는다.**
    """
    out = np.empty_like(pattern)
    for ch in range(3):
        s = sigma_px[ch]
        if s <= 0.05:
            out[..., ch] = pattern[..., ch]
            continue
        r = max(1, int(np.ceil(3 * s)))
        x = np.arange(-r, r + 1)
        k = np.exp(-0.5 * (x / s) ** 2)
        k /= k.sum()
        pad = np.pad(pattern[..., ch], ((0, 0), (r, r)), mode="edge")
        out[..., ch] = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, pad)
    return out


# ---------------------------------------------------------------- 촬영
def capture(albedo, shading, projected, noise, ambient, crosstalk, rng, sigma_px=None):
    M = np.eye(3) * (1 - 2 * crosstalk) + crosstalk
    M = M / M.sum(1, keepdims=True)                       # 에너지 보존
    if sigma_px is not None and np.ndim(projected) == 3:
        projected = subsurface_blur(projected, sigma_px)
    src = shading[..., None] * albedo * projected
    v = np.einsum("...i,ji->...j", src, M) + ambient   # matmul 의 허위 경고 회피
    v = v + rng.normal(0, noise, v.shape)
    return np.clip(np.round(np.clip(v, 0, 1) * 255) / 255, 0, 1)   # 포화 + 8bit


# ---------------------------------------------------------------- 분류
def feature_rgb_white(obs, white):
    return obs / np.maximum(white, 1e-3)


def cls_from_rgb(ratio):
    d = ((ratio[..., None, :] - PALETTE) ** 2).sum(-1)
    return np.argmin(d, -1), -d


def cls_gchroma(obs):
    G, B = obs[..., 1], obs[..., 2]
    g = G / np.maximum(G + B, 1e-6)
    d = np.abs(g[..., None] - np.array([1.0, 0.5, 0.0]))
    return np.argmin(d, -1), -d


def cls_hue(obs):
    R, G, B = obs[..., 0], obs[..., 1], obs[..., 2]
    h = np.degrees(np.arctan2(np.sqrt(3) * (G - B), 2 * R - G - B)) % 360
    d = np.abs((h[..., None] - np.array([120.0, 180.0, 240.0]) + 180) % 360 - 180) / 180.0
    return np.argmin(d, -1), -d


# ---------------------------------------------------------------- 디코딩
def sample_stripe_centers(obs, s_true):
    """각 행에서 스트라이프 중심의 관측색을 뽑는다.

    s_true 는 행마다 x 에 대해 단조증가하므로, 역보간으로 k+0.5 지점의 x 를 찾는다.
    스트라이프 분할(에지 검출)은 성공했다고 가정하고 '색 판정' 문제만 분리한다.
    """
    H, W, _ = obs.shape
    L = int(np.ceil(s_true.max()))
    out = np.full((H, L, 3), np.nan)
    xs = np.arange(W)
    for r in range(H):
        row = s_true[r]
        ks = np.arange(L) + 0.5
        inside = (ks >= row[0]) & (ks <= row[-1])
        xk = np.interp(ks[inside], row, xs)
        for ch in range(3):
            out[r, inside, ch] = np.interp(xk, xs, obs[r, :, ch])
    return out


def build_lut(code, n):
    lut = {}
    for i in range(len(code) - n + 1):
        lut.setdefault(tuple(code[i:i + n]), []).append(i)
    return {k: v[0] for k, v in lut.items() if len(v) == 1}


def decode_windows(labels, code, n=4):
    """4-윈도우를 LUT 에 조회해 절대 스트라이프 인덱스를 얻는다.

    반환 (decoded, hit) — decoded[i] 는 위치 i 에서 시작하는 윈도우가 가리키는 인덱스,
    hit[i] 는 LUT 에 걸렸는지. 걸리지 않으면 오류가 검출된 것이고,
    걸렸는데 틀리면 조용한 오복원이다 (문서 6.1절).
    """
    lut = build_lut(code, n)
    H, L = labels.shape
    M = L - n + 1
    decoded = np.full((H, M), -1)
    for r in range(H):
        for i in range(M):
            decoded[r, i] = lut.get(tuple(labels[r, i:i + n]), -1)
    return decoded, decoded >= 0


def consensus_filter(decoded, radius=2):
    """이웃 윈도우가 연속된 인덱스를 가리킬 때만 채택 — 문맥 제약."""
    H, M = decoded.shape
    keep = np.zeros_like(decoded, bool)
    for d in range(1, radius + 1):
        left = np.full_like(decoded, -99)
        right = np.full_like(decoded, -99)
        left[:, d:] = decoded[:, :-d] + d
        right[:, :-d] = decoded[:, d:] - d
        keep |= (decoded >= 0) & ((left == decoded) | (right == decoded))
    return keep


# ---------------------------------------------------------------- 실행
def evaluate(args):
    rng = np.random.default_rng(args.seed)
    info = json.load(open("dataset/debruijn/debruijn_pattern_info.json"))
    code = np.array(info["sequence"])
    n = info["n"]
    L = len(code)

    H, W = args.height, args.width
    albedo, shading, bump, patches = make_scene(args.scene, H, W, rng)
    s_true = stripe_truth(H, W, L, bump)
    idx_true = np.clip(s_true.astype(int), 0, L - 1)

    # 산란 σ 를 픽셀 단위로. um_per_px = 피치 / (스트라이프당 카메라 픽셀 수)
    um_per_px = args.pitch_um / (W / L)
    if args.sss_um > 0:
        g = args.sss_um / um_per_px
        sigma = np.array([g * args.sss_gb_ratio, g, g / args.sss_gb_ratio])  # R, G, B
    else:
        sigma = None

    projected = PALETTE[code[idx_true]]
    obs = capture(albedo, shading, projected, args.noise, args.ambient, args.crosstalk, rng, sigma)
    white = capture(albedo, shading, np.ones(3), args.noise, args.ambient, args.crosstalk, rng, sigma)

    # 스트라이프 중심 샘플
    c_obs = sample_stripe_centers(obs, s_true)
    c_white = sample_stripe_centers(white, s_true)
    c_alb = sample_stripe_centers(np.broadcast_to(albedo, obs.shape).copy(), s_true)
    c_shade = sample_stripe_centers(np.repeat(shading[..., None], 3, -1), s_true)
    good = np.isfinite(c_obs[..., 0])
    c_obs = np.nan_to_num(c_obs, nan=0.0)
    c_white = np.nan_to_num(c_white, nan=1.0)
    Ls = c_obs.shape[1]
    truth = np.tile(np.arange(Ls), (H, 1))

    feats = {}
    if args.white:
        feats["RGB + white"] = cls_from_rgb(c_obs / np.maximum(c_white, 1e-3))
    feats["2채널 색도"] = cls_gchroma(c_obs)
    feats["HSV hue"] = cls_hue(c_obs)

    # 신호 없는 스트라이프: G,B 반사 * 음영이 바닥
    with np.errstate(invalid="ignore"):
        signal = c_shade[..., 1] * (c_alb[..., 1] + c_alb[..., 2]) / 2
        dead = ~good | (np.nan_to_num(signal, nan=0.0) < args.dead_threshold)

    results = {}
    for name, (labels, score) in feats.items():
        labels = np.where(good, labels, 0)
        true_lab = code[np.clip(truth, 0, L - 1)]
        lab_ok = (labels == true_lab) & good

        decoded, hit = decode_windows(labels, code, n)
        wtruth = np.tile(np.arange(decoded.shape[1]), (H, 1))
        correct = hit & (decoded == wtruth)
        silent = hit & (decoded != wtruth)
        keep = consensus_filter(decoded)
        results[name] = {
            "라벨 정확도": lab_ok[good].mean(),
            "LUT 히트": hit.mean(),
            "위치 정확": correct.mean(),
            "조용한 오복원": silent.mean(),
            "오류 검출(미스)": (~hit).mean(),
            "문맥필터 후 정확": (correct & keep).sum() / max(keep.sum(), 1),
            "문맥필터 후 오복원": (silent & keep).sum() / max(keep.sum(), 1),
            "_labels": labels, "_correct": correct, "_silent": silent, "_hit": hit,
        }
    return dict(albedo=albedo, obs=obs, white=white, idx_true=idx_true, dead=dead,
                code=code, results=results, patches=patches, c_alb=c_alb, signal=signal,
                good=good, s_true=s_true)


# ---------------------------------------------------------------- 그림
def _show(ax, img, title, **kw):
    ax.imshow(img, **kw)
    ax.set_title(title, color=FG, fontsize=10, pad=5)
    ax.set_xticks([]); ax.set_yticks([])


def figure_scene(R, args, path):
    names = list(R["results"].keys())
    ncol = 1 + len(names)
    fig, axes = plt.subplots(2, ncol, figsize=(3.9 * ncol, 5.6), facecolor=BG)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_facecolor(BG)

    _show(axes[0, 0], np.clip(R["albedo"], 0, 1), "표면 알베도 (텍스처)", aspect="auto")
    _show(axes[1, 0], np.clip(R["obs"], 0, 1), "카메라 캡처 (패턴 투사)", aspect="auto")

    for i, name in enumerate(names):
        res = R["results"][name]
        lab_img = PALETTE[res["_labels"]] * R["good"][..., None]
        _show(axes[0, i + 1], lab_img, f"{name} — 분류된 스트라이프 색", aspect="auto")
        st = np.zeros(res["_correct"].shape + (3,))
        st[res["_correct"]] = (0.12, 0.55, 0.35)
        st[res["_silent"]] = (0.85, 0.20, 0.16)
        st[~res["_hit"]] = (0.55, 0.48, 0.15)
        _show(axes[1, i + 1], st, f"정확 {100 * res['위치 정확']:.1f}% · "
                                  f"오복원 {100 * res['조용한 오복원']:.1f}% · "
                                  f"미스 {100 * res['오류 검출(미스)']:.1f}%", aspect="auto")

    fig.suptitle(f"씬 '{args.scene}' · white {'사용' if args.white else '미사용'} · "
                 f"노이즈 σ={args.noise}   |   초록=정확  빨강=조용한 오복원  노랑=LUT 미스",
                 color=FG, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"   {path}")


def figure_margin(path):
    """알베도 색별 이론적 복원 가능성 — G:B 비만이 결정한다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG,
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    names = [n for n, _ in MACBETH]
    cols = [np.array(c) / 255.0 for _, c in MACBETH]
    marg = np.array([gb_margin(c) for c in cols])
    order = np.argsort(marg)
    ax1.barh(range(24), marg[order], color=[cols[i] for i in order],
             edgecolor=DIM, linewidth=0.6)
    ax1.set_yticks(range(24))
    ax1.set_yticklabels([names[i] for i in order], color=FG, fontsize=8.5)
    ax1.axvline(0.5, color=DIM, linestyle=":", linewidth=1)
    ax1.axvline(0.18, color="#ff5555", linestyle="--", linewidth=1.2)
    ax1.text(0.185, 23.4, "이 아래로는 Cyan 이 Green/Blue 로 붕괴", color="#ff5555", fontsize=8.5, va="top")
    ax1.set_xlim(0, 0.55)
    ax1.set_xlabel("색도 여유  min(g, 1-g),   g = a_G/(a_G+a_B)   ← 이상값 0.5", color=FG)
    ax1.tick_params(colors=FG, labelsize=8.5)
    for sp in ax1.spines.values():
        sp.set_color(DIM)
    ax1.set_title("ColorChecker 24색 — white 레퍼런스가 없을 때", color=FG, fontsize=11.5, loc="left")

    g = np.linspace(0.001, 0.999, 400)
    ratio = g / (1 - g)
    ax2.plot(ratio, np.minimum(g, 1 - g), color="#4ec9a6", linewidth=2.2)
    ax2.axhline(0.18, color="#ff5555", linestyle="--", linewidth=1.2)
    ax2.axvspan(1 / 4.5, 4.5, color="#4ec9a6", alpha=0.10)
    ax2.text(0.055, 0.50, "안전 구간\nG:B ≈ 1:4.5 ~ 4.5:1", color="#4ec9a6",
             ha="left", va="top", fontsize=9.5)
    ax2.set_xscale("log")
    ax2.set_xlim(0.04, 25)
    ticks = [0.05, 0.1, 0.25, 1, 4, 10, 20]      # 로그 눈금의 mathtext 마이너스 회피
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(["1:20", "1:10", "1:4", "1:1", "4:1", "10:1", "20:1"])
    ax2.minorticks_off()
    ax2.set_ylim(0, 0.55)
    ax2.set_xlabel("표면의 G : B 반사율 비", color=FG)
    ax2.set_ylabel("색도 여유", color=FG)
    ax2.tick_params(colors=FG, labelsize=9)
    for sp in ax2.spines.values():
        sp.set_color(DIM)
    ax2.set_title("밝기·R 성분과는 무관하다", color=FG, fontsize=11.5, loc="left")

    fig.tight_layout()
    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"   {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["texture", "macbeth", "dental"], default="texture")
    ap.add_argument("--pitch-um", type=float, default=200.0,
                    help="카메라 평면에서의 스트라이프 피치")
    ap.add_argument("--sss-um", type=float, default=0.0,
                    help="표면하 산란 확산 길이 (Green 기준, 0 이면 없음)")
    ap.add_argument("--sss-gb-ratio", type=float, default=1.4,
                    help="Green/Blue 확산 길이 비 — 장파장이 더 퍼진다")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--ambient", type=float, default=0.03)
    ap.add_argument("--crosstalk", type=float, default=0.06)
    ap.add_argument("--dead-threshold", type=float, default=0.05,
                    help="G,B 반사 신호가 이보다 낮으면 '무신호'로 집계")
    ap.add_argument("--no-white", dest="white", action="store_false",
                    help="white 레퍼런스 프레임 없이 (고정 마스크 가정)")
    ap.add_argument("--margin-map", action="store_true", help="이론 지도만 그리고 종료")
    ap.add_argument("--doc", action="store_true",
                    help="docs/DeBruijn/image 에 문서용 그림 8·9 생성")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="dataset/debruijn/sim")
    args = ap.parse_args()

    if args.doc:
        out = "docs/DeBruijn/image"
        os.makedirs(out, exist_ok=True)
        print("문서용 그림 생성:")
        figure_margin(f"{out}/fig8_margin_map.png")
        args.scene, args.white = "macbeth", True
        figure_scene(evaluate(args), args, f"{out}/fig9_surface_color.png")
        return

    os.makedirs(args.out, exist_ok=True)
    if args.margin_map:
        print("그림 생성:")
        figure_margin(f"{args.out}/margin_map.png")
        return

    R = evaluate(args)
    tag = f"{args.scene}_{'white' if args.white else 'nowhite'}"

    sss = (f" · 산란 {args.sss_um:.0f}µm (피치 {args.pitch_um:.0f}µm 의 "
           f"{args.sss_um / args.pitch_um:.2f}배)" if args.sss_um > 0 else " · 산란 없음")
    print(f"씬 '{args.scene}' · white {'사용' if args.white else '미사용'} · "
          f"노이즈 σ={args.noise}{sss} · {args.height}x{args.width} · 72 스트라이프/행\n")
    print(f"   {'방법':<13}{'라벨':>8}{'LUT히트':>9}{'위치정확':>9}{'조용한오복원':>13}"
          f"{'문맥필터후 정확':>16}{'오복원':>9}")
    for name, r in R["results"].items():
        print(f"   {name:<13}{100*r['라벨 정확도']:7.1f}%{100*r['LUT 히트']:8.1f}%"
              f"{100*r['위치 정확']:8.1f}%{100*r['조용한 오복원']:12.1f}%"
              f"{100*r['문맥필터 후 정확']:15.1f}%{100*r['문맥필터 후 오복원']:8.1f}%")

    if R["patches"]:
        first = list(R["results"].values())[0]
        print("\n   패치별 위치 정확도 (색도 여유 오름차순)")
        rows = []
        for name, rgb, (y0, y1, x0, x1) in R["patches"]:
            m = float(gb_margin(rgb))
            sel = R["idx_true"][y0:y1, x0:x1]
            lo, hi = int(sel.min()), min(int(sel.max()), first["_correct"].shape[1] - 1)
            if hi > lo:
                acc = first["_correct"][y0:y1, lo:hi].mean()
                rows.append((m, name, rgb, acc))
        for m, name, rgb, acc in sorted(rows):
            flag = "  ←위험" if m < 0.18 else ""
            print(f"      {name:<14} G:B = {rgb[1]:.2f}:{rgb[2]:.2f}   여유 {m:.3f}"
                  f"   정확 {100*acc:5.1f}%{flag}")

    print("\n그림 생성:")
    figure_scene(R, args, f"{args.out}/scene_{tag}.png")


if __name__ == "__main__":
    main()
