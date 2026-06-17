"""
dlp_addon.py — Blender Addon: DLP Structured Light Simulator

설치:
    Edit > Preferences > Add-ons > Install > dlp_addon.py 선택
    "3D View: DLP Structured Light Simulator" 활성화

사용법:
    1. 3D Viewport에서 N 키 → 사이드바 → 'DLP' 탭
    2. Blender 기본 도구로 메시 임포트 후 선택
    3. [Setup Scene] → 프로젝터/카메라 자동 배치
    4. Ctrl+Shift+Z (Rendered Preview) → 뷰포트에서 프린지 실시간 확인
    5. ◀ ▶ 버튼으로 패턴 스텝 전환
    6. [Render All] → PNG 파일 일괄 출력
"""

bl_info = {
    "name":        "DLP Structured Light Simulator",
    "author":      "VulkanProject / StructedLight",
    "version":     (1, 1),
    "blender":     (4, 0, 0),
    "location":    "View3D > Sidebar > DLP",
    "description": "Simulate fringe projection profilometry with a virtual DLP projector",
    "category":    "3D View",
}

import bpy
import math
import numpy as np
from functools import lru_cache
from pathlib import Path
from bpy.types  import Operator, Panel, PropertyGroup
from bpy.props  import (StringProperty, IntProperty, FloatProperty,
                         EnumProperty, BoolProperty, PointerProperty)

# ─── Named scene objects ──────────────────────────────────────────────────────
PROJ_CAM   = "DLP_ProjectorCamera"
MAIN_CAM   = "DLP_MainCamera"
PROJ_LIGHT = "DLP_ProjectorLight"
FRINGE_IMG = "DLP_FringeImage"
MAT_NAME   = "DLP_Projection"   # 더 이상 사용 안 함 (하위 호환성 유지)
WHITE_MAT  = "DLP_White"        # 흰색 Lambertian 재질 (구조광 측정 대상)


# ─── Pattern generators (pure NumPy) ─────────────────────────────────────────

def _sinusoidal(step: int, N: int, freq: float, W: int, H: int) -> np.ndarray:
    xs    = np.arange(W, dtype=np.float32)
    phase = 2 * math.pi * freq * xs / W - 2 * math.pi * step / N
    row   = np.clip(127.5 * (1.0 + np.cos(phase)), 0, 255).astype(np.uint8)
    return np.tile(row, (H, 1))

def _gray_code(bit: int, N_bits: int, W: int, H: int) -> np.ndarray:
    xs   = np.arange(W)
    gray = (xs * (2 ** N_bits) // W).astype(np.int32)
    gray ^= gray >> 1
    row  = (((gray >> (N_bits - 1 - bit)) & 1) * 255).astype(np.uint8)
    return np.tile(row, (H, 1))

def _binary(step: int, W: int, H: int) -> np.ndarray:
    xs  = np.arange(W)
    row = (((xs * 2 ** (step + 1) // W) % 2) * 255).astype(np.uint8)
    return np.tile(row, (H, 1))

def _flat(value: int, W: int, H: int) -> np.ndarray:
    return np.full((H, W), value, dtype=np.uint8)


def _solid_color(rgb: tuple, W: int, H: int) -> np.ndarray:
    row = np.tile(np.asarray(rgb, dtype=np.uint8), (W, 1))
    return np.tile(row[np.newaxis, :, :], (H, 1, 1))


# ─── Single-shot color-coded stripe patterns ─────────────────────────────────

_COLOR_PALETTE = [
    (224, 31,  31),   # red
    (31,  224, 31),   # green
    (31,  31,  224),  # blue
    (31,  224, 224),  # cyan
    (224, 31,  224),  # magenta
    (224, 224, 31),   # yellow
]

_HAMMING_PALETTE = [
    (0,   0,   0),    # 000
    (255, 0,   0),    # 001  (R)
    (0,   255, 0),    # 010  (G)
    (255, 255, 0),    # 011  (RG)
    (0,   0,   255),  # 100  (B)
    (255, 0,   255),  # 101  (RB)
    (0,   255, 255),  # 110  (GB)
    (255, 255, 255),  # 111  (RGB)
]

@lru_cache(maxsize=None)
def _de_bruijn_sequence(k: int, n: int) -> tuple:
    """B(k, n): 길이 k^n, 모든 길이-n 윈도우(순환)가 정확히 1회 등장하는 시퀀스."""
    a   = [0] * (k * n + 1)
    seq = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return tuple(seq)


@lru_cache(maxsize=None)
def _constrained_de_bruijn_sequence(k: int, n: int) -> tuple:
    """
    모든 길이-n 윈도우가 유일하면서 같은 색이 연속하지 않는 순환 시퀀스.

    일반 De Bruijn 그래프에서 인접 심볼이 같은 edge를 제거한 뒤 Euler
    circuit을 구한다. 결과 길이는 k * (k - 1) ** (n - 1)이다.
    """
    from itertools import product

    def valid(word):
        return all(a != b for a, b in zip(word, word[1:]))

    vertices = [
        tuple(word)
        for word in product(range(k), repeat=n - 1)
        if valid(word)
    ]
    adjacency = {
        vertex: [
            vertex[1:] + (symbol,)
            for symbol in reversed(range(k))
            if symbol != vertex[-1]
        ]
        for vertex in vertices
    }

    stack = [vertices[0]]
    circuit = []
    while stack:
        vertex = stack[-1]
        if adjacency[vertex]:
            stack.append(adjacency[vertex].pop())
        else:
            circuit.append(stack.pop())
    circuit.reverse()

    edge_count = k * (k - 1) ** (n - 1)
    linear = list(circuit[0]) + [vertex[-1] for vertex in circuit[1:]]
    return tuple(linear[:edge_count])


@lru_cache(maxsize=None)
def _hamming_color_sequence(n: int) -> tuple:
    """
    내부 인접 코드워드의 Hamming 거리가 항상 1이고 길이-n 윈도우가 유일한
    길이 3^(n-1) 코드워드(0..7) 선형 시퀀스.

    diffs = de_bruijn(3, n-1)   (알파벳 {0,1,2} = 토글할 비트 위치, 길이 3^(n-1))
    seq[0] = 111
    seq[i] = seq[i-1] XOR (1 << diffs[i-1])

    화면의 좌우 끝을 연결하지 않는 선형 패턴이므로 마지막→첫 번째 색의
    Hamming 거리는 디코딩 조건에 포함하지 않는다.
    """
    diffs = _de_bruijn_sequence(3, n - 1)
    seq   = [0b111]
    for d in diffs[:-1]:
        seq.append(seq[-1] ^ (1 << d))
    return tuple(seq)


def _stripe_image(sequence: list, palette: list, W: int, H: int) -> np.ndarray:
    """색상 인덱스 시퀀스를 수직 스트라이프 RGB 이미지로 변환."""
    palette_arr = np.asarray(palette, dtype=np.uint8)
    sequence_arr = np.asarray(sequence, dtype=np.int64)
    stripe = np.arange(W, dtype=np.int64) * len(sequence) // W
    row = palette_arr[sequence_arr[stripe]]
    return np.tile(row[np.newaxis, :, :], (H, 1, 1))


def _self_equalizing_image(sequence: list, palette: list,
                           W: int, H: int) -> np.ndarray:
    """
    각 논리 stripe를 [색, 보색] 두 sub-stripe로 투영한다.

    두 sub-stripe의 채널별 합이 항상 255이므로 수신 영상에서 pair sum을
    local illumination/albedo 기준값으로, pair difference를 색 코드로 사용할
    수 있다. 공간 해상도는 일반 De Bruijn의 절반이다.
    """
    palette_arr = np.asarray(palette, dtype=np.uint8)
    sequence_arr = np.asarray(sequence, dtype=np.int64)
    physical_count = 2 * len(sequence)
    physical_index = np.arange(W, dtype=np.int64) * physical_count // W
    logical_index = physical_index // 2
    row = palette_arr[sequence_arr[logical_index]]
    complement = (physical_index % 2) == 1
    row = np.where(complement[:, np.newaxis], 255 - row, row).astype(np.uint8)
    return np.tile(row[np.newaxis, :, :], (H, 1, 1))


def _color_code_data(props) -> dict:
    """현재 color stripe 설정의 시퀀스, 팔레트, 디코딩 정보를 반환."""
    mode = props.color_code_mode
    if mode == 'hamming':
        sequence = _hamming_color_sequence(props.color_n)
        return {
            "mode": mode,
            "sequence": sequence,
            "palette": _HAMMING_PALETTE,
            "logical_stripes": len(sequence),
            "projected_stripes": len(sequence),
            "decode_window": props.color_n,
            "adjacency": "Hamming distance 1 (linear neighbors)",
            "self_equalizing": False,
        }

    sequence = _constrained_de_bruijn_sequence(props.color_k, props.color_n)
    self_equalizing = mode == 'self_equalizing'
    return {
        "mode": mode,
        "sequence": sequence,
        "palette": _COLOR_PALETTE[:props.color_k],
        "logical_stripes": len(sequence),
        "projected_stripes": len(sequence) * (2 if self_equalizing else 1),
        "decode_window": props.color_n,
        "adjacency": "different color for every cyclic neighbor",
        "self_equalizing": self_equalizing,
    }


def _color_coded(props, W: int, H: int) -> np.ndarray:
    data = _color_code_data(props)
    if data["self_equalizing"]:
        return _self_equalizing_image(
            data["sequence"], data["palette"], W, H
        )
    return _stripe_image(data["sequence"], data["palette"], W, H)


# ─── Color-Multiplexed Phase Shift 패턴 ───────────────────────────────────────
# 단일 촬영용 색 다중화 phase-shifting. 3-step PSP의 I0,I1,I2를 R/G/B
# 채널에 각각 실어 한 장으로 투영 — single-shot이면서 phase-shift 수준의
# per-pixel dense 정밀도를 얻는다 (대가: 실제 카메라에서는 채널 간 crosstalk
# 보정이 필요하나, 본 렌더러는 채널별로 독립 렌더되므로 crosstalk 없음).

def _color_phase(freq: float, W: int, H: int) -> np.ndarray:
    """(H,W,3) uint8 — 3-step PSP를 R(=I0)/G(=I1)/B(=I2) 채널에 다중화."""
    channels = [_sinusoidal(step, 3, freq, W, H) for step in range(3)]
    return np.stack(channels, axis=-1)


def _parse_color_matrix(value: str) -> np.ndarray:
    """'a,b,c;d,e,f;g,h,i' 문자열을 nonsingular 3x3 행렬로 파싱."""
    try:
        rows = [
            [float(component.strip()) for component in row.split(',')]
            for row in value.split(';')
        ]
        matrix = np.asarray(rows, dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Color Matrix는 '1,0,0;0,1,0;0,0,1' 형식이어야 합니다") from exc

    if matrix.shape != (3, 3):
        raise ValueError("Color Matrix는 행 3개, 열 3개여야 합니다")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Color Matrix에는 유한한 숫자만 사용할 수 있습니다")
    if abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("Color Matrix가 singular라 역행렬을 계산할 수 없습니다")
    if np.linalg.cond(matrix) > 1e6:
        raise ValueError("Color Matrix의 condition number가 너무 커서 보정이 불안정합니다")
    return matrix


def _precompensate_color(arr: np.ndarray, response_matrix: np.ndarray) -> np.ndarray:
    """
    camera_rgb = response_matrix @ projector_rgb 모델의 projector-side 역보정.
    clipping된 픽셀은 실제 시스템에서도 완전한 보상이 불가능하다.
    """
    inverse = np.linalg.inv(response_matrix)
    corrected = (arr.astype(np.float64) / 255.0) @ inverse.T
    return np.clip(np.rint(corrected * 255.0), 0, 255).astype(np.uint8)


def make_pattern(props, step: int) -> np.ndarray:
    W, H = props.width, props.height
    t    = props.pattern_type
    if t == 'phase_shift': return _sinusoidal(step, props.N, props.freq, W, H)
    if t == 'gray_code':   return _gray_code(step, props.N_bits, W, H)
    if t == 'color_coded':
        arr = _color_coded(props, W, H)
    elif t == 'color_phase':
        arr = _color_phase(props.color_phase_freq, W, H)
    else:
        return _binary(step, W, H)

    if props.color_precompensate:
        arr = _precompensate_color(
            arr, _parse_color_matrix(props.color_response_matrix)
        )
    return arr

def pattern_count(props) -> int:
    t = props.pattern_type
    if t == 'phase_shift': return props.N
    if t == 'gray_code':   return props.N_bits
    if t == 'color_coded': return 1
    if t == 'color_phase': return 1
    return props.N_binary


# ─── Blender image helpers ────────────────────────────────────────────────────

def _arr_to_image(arr: np.ndarray, name: str) -> bpy.types.Image:
    """(H,W) 또는 (H,W,3) uint8 numpy → Blender Image (reuses existing if same name)."""
    if arr.ndim == 3:
        h, w, _ = arr.shape
    else:
        h, w = arr.shape
    img = bpy.data.images.get(name)
    if img is None or img.size[0] != w or img.size[1] != h:
        if img:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name, width=w, height=h, alpha=False)
    img.colorspace_settings.name = 'Non-Color'
    flipped = np.flipud(arr).astype(np.float32) / 255.0
    if arr.ndim == 3:
        rgba = np.concatenate([flipped, np.ones((h, w, 1), dtype=np.float32)], axis=-1)
    else:
        rgba = np.stack([flipped, flipped, flipped, np.ones_like(flipped)], axis=-1)
    img.pixels[:] = rgba.ravel()
    return img


# ─── Scene setup ─────────────────────────────────────────────────────────────

def _get_or_new(name: str, data) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
    return obj

def _point_at(obj: bpy.types.Object, loc: tuple, target: tuple = (0, 0, 0)):
    import mathutils
    obj.location      = mathutils.Vector(loc)
    direction         = mathutils.Vector(target) - mathutils.Vector(loc)
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def _apply_world_render(props):
    """월드 ambient 및 Cycles 렌더 설정 적용 (카메라 위치는 건드리지 않음)."""
    # World ambient
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value    = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = props.ambient

    s = bpy.context.scene

    # EEVEE 사용 — 뷰포트 Rendered Preview와 동일한 엔진이므로 밝기가 자연스럽게 일치.
    # (Cycles는 물리 기반 광감쇄로 어둡게 나오는 반면, EEVEE는 뷰포트와 동일한 출력)
    s.render.engine       = 'BLENDER_EEVEE_NEXT'   # Blender 4.2+
    s.render.resolution_x = props.width
    s.render.resolution_y = props.height
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode  = (
        'RGB' if props.pattern_type in ('color_coded', 'color_phase') else 'BW'
    )
    s.render.image_settings.color_depth = '8'

    # 컬러 매니지먼트를 건드리지 않음 → 씬 기본값(= 뷰포트와 동일) 그대로 사용.
    # 뷰포트에서 보이는 밝기와 렌더 결과가 자동으로 일치.


def _configure_scene(proj_pos, main_pos, target, props) -> str:
    """
    지정한 위치에 DLP 프로젝터 카메라 · 메인 카메라 · 스팟 라이트를 배치.
    proj_pos / main_pos / target 은 mathutils.Vector 또는 3-tuple.
    """
    import mathutils
    fov = props.fov_deg

    # Projector camera (투영 좌표계 기준점 — 실제 렌더는 하지 않음)
    cam_p           = bpy.data.cameras.get(PROJ_CAM) or bpy.data.cameras.new(PROJ_CAM)
    cam_p.lens_unit = 'FOV'
    cam_p.angle     = math.radians(fov)
    proj_obj        = _get_or_new(PROJ_CAM, cam_p)
    _point_at(proj_obj, proj_pos, target)

    # Main camera (렌더 카메라)
    cam_m           = bpy.data.cameras.get(MAIN_CAM) or bpy.data.cameras.new(MAIN_CAM)
    cam_m.lens_unit = 'FOV'
    cam_m.angle     = math.radians(fov)
    main_obj        = _get_or_new(MAIN_CAM, cam_m)
    _point_at(main_obj, main_pos, target)
    bpy.context.scene.camera = main_obj

    # ── Fill Light: 메인 카메라(뷰포트) 방향에서 백색광 조사 ──────
    # "내가 바라보는 방향에서 빛 쏴줘" → 카메라와 동일한 위치에 배치
    ld               = bpy.data.lights.get(PROJ_LIGHT) or bpy.data.lights.new(PROJ_LIGHT, type='SPOT')
    ld.energy        = props.light_energy
    ld.spot_size     = math.radians(fov * 1.8)   # 넓게 — 전체 장면 조명
    ld.spot_blend    = 0.3
    ld.shadow_soft_size = 0.05
    ld.use_nodes     = False
    ld.color         = (1.0, 1.0, 1.0)
    light_obj        = _get_or_new(PROJ_LIGHT, ld)
    # 카메라와 같은 위치·방향 → 정면 백색광
    light_obj.location       = main_obj.location.copy()
    light_obj.rotation_euler = main_obj.rotation_euler.copy()

    _apply_world_render(props)

    p  = mathutils.Vector(proj_pos)
    m  = mathutils.Vector(main_pos)
    dp = (p - m).length
    return (f"Cam ({m.x:.2f},{m.y:.2f},{m.z:.2f})  "
            f"Proj ({p.x:.2f},{p.y:.2f},{p.z:.2f})  "
            f"baseline={dp:.3f}m")


# ─── Camera Calibration export ────────────────────────────────────────────────

def _opencv_cam2world_matrix(cam_obj: bpy.types.Object):
    """Blender camera pose -> OpenCV camera-to-world rotation."""
    import mathutils

    # Blender(−Z forward, Y up) → OpenCV(+Z forward, Y down): 로컬 Y·Z 반전
    flip = mathutils.Matrix.Diagonal(mathutils.Vector((1.0, -1.0, -1.0)))
    return cam_obj.matrix_world.to_3x3() @ flip


def _build_calibration_dict(props, cam_name: str = MAIN_CAM,
                            origin_cam_name: str = MAIN_CAM) -> dict:
    """
    지정한 카메라(기본 DLP_MainCamera)의 intrinsic/extrinsic을 PhaseShift.py의
    CameraInformation(image_size, intrinsic, distortion, rotate, translate)
    필드에 맞춰 dict로 구성.

    DLP_ProjectorCamera에도 동일하게 적용 가능
    (projector_calibration.json — 구조광 평면 투영 모델용).

    - intrinsic: 핀홀 모델. fx == fy == (W/2) / tan(fov/2)  (정사각 픽셀, 중앙 주점)
    - distortion: 렌더 결과는 왜곡이 없으므로 0
    - rotate/translate: origin_cam_name의 OpenCV 카메라 좌표계를 기준 world로 삼는다.
      기본값은 DLP_MainCamera이므로 camera_calibration.json은 원점/무회전이고,
      projector_calibration.json은 메인 카메라 기준 상대 pose가 된다.
      rotate는 기준 world→해당 카메라의 Rodrigues, translate는 해당 광학중심의
      기준 world 좌표다.
    """
    cam_obj = bpy.data.objects.get(cam_name)
    if cam_obj is None:
        raise RuntimeError(f"{cam_name} 없음. 'Setup from View' 먼저 실행.")
    origin_obj = bpy.data.objects.get(origin_cam_name)
    if origin_obj is None:
        raise RuntimeError(f"{origin_cam_name} 없음. 'Setup from View' 먼저 실행.")

    W, H = props.width, props.height
    f    = (W / 2.0) / math.tan(math.radians(props.fov_deg) / 2.0)
    cx, cy = W / 2.0, H / 2.0

    intrinsic = [[f,   0.0, cx],
                 [0.0, f,   cy],
                 [0.0, 0.0, 1.0]]

    R_origin_cam2world = _opencv_cam2world_matrix(origin_obj)
    R_origin_world2cam = R_origin_cam2world.transposed()
    R_cam2origin       = R_origin_world2cam @ _opencv_cam2world_matrix(cam_obj)
    R_origin2cam       = R_cam2origin.transposed()

    quat  = R_origin2cam.to_quaternion()
    axis  = quat.axis
    angle = quat.angle
    rvec  = [axis.x * angle, axis.y * angle, axis.z * angle]

    loc = R_origin_world2cam @ (
        cam_obj.matrix_world.translation - origin_obj.matrix_world.translation
    )

    return {
        "image_size": [W, H],
        "intrinsic":  intrinsic,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rotate":     rvec,
        "translate":  [loc.x, loc.y, loc.z],
    }


# ─── Projection material ──────────────────────────────────────────────────────

def _build_material(bpy_img: bpy.types.Image, fov_deg: float,
                    width: int, height: int) -> bpy.types.Material:
    """
    깊이 반영 구조광 재질 — Emission + 프로젝터 원근 UV.

    왜 이 방식만 깊이를 반영하는가:
      Spot Light 텍스처 방식은 light shader 안에서 각 픽셀이 어느 표면점에
      닿는지 알 수 없어 원근 투영이 불가능 → 시차(parallax) 0.

      이 방식은 표면 셰이더 안에서 Geometry.Position(월드 좌표)을 직접 읽어
      프로젝터 역행렬로 변환하므로 깊이별로 u_proj가 달라짐 → 시차 발생.

    Node graph:
      Geometry.Position (월드)
        → dot(P, M.row[0..2]) + M.row[3]    ← Python에서 행렬 상수 하드코딩
        → [X_local, Y_local, Z_local]
        → X / (−Z) / tan(fov/2)        × 0.5 + 0.5  ─┐
        → Y / (−Z) / (tan/aspect)      × 0.5 + 0.5  ─┤ → ImageTex
                                                        │   ↓ Color
      Geometry.Normal · proj_forward → Lambert → Strength │
                                                        → Emission → Output
    """
    import mathutils

    proj_cam_obj = bpy.data.objects.get(PROJ_CAM)
    if proj_cam_obj is None:
        raise RuntimeError("DLP_ProjectorCamera 없음. 'Setup from View' 먼저 실행.")

    # 프로젝터 world→local 행렬 (4×4 역행렬)
    M          = proj_cam_obj.matrix_world.inverted()
    r0, r1, r2 = M.row[0], M.row[1], M.row[2]

    # 표면 → 프로젝터 방향 (Lambert용): local +Z 축 = world에서 프로젝터 쪽을 향함
    proj_fwd = proj_cam_obj.matrix_world.col[2].xyz.normalized()

    half_tan = math.tan(math.radians(fov_deg) / 2)
    aspect   = width / height

    old = bpy.data.materials.get(MAT_NAME)
    if old:
        bpy.data.materials.remove(old)
    mat = bpy.data.materials.new(MAT_NAME)
    mat.use_nodes = True
    nt    = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    # ── Output + Emission ────────────────────────────────────────
    out  = nodes.new('ShaderNodeOutputMaterial')
    emit = nodes.new('ShaderNodeEmission')
    links.new(emit.outputs['Emission'], out.inputs['Surface'])

    # ── 프린지 이미지 텍스처 ─────────────────────────────────────
    img_tex               = nodes.new('ShaderNodeTexImage')
    img_tex.image         = bpy_img
    img_tex.extension     = 'CLIP'
    img_tex.interpolation = 'Linear'
    links.new(img_tex.outputs['Color'], emit.inputs['Color'])

    # ── 표면점 월드 좌표 → 프로젝터 로컬 좌표 ────────────────────
    # 각 픽셀의 3D 위치(깊이)가 다르면 u_proj가 달라짐 = 시차
    geom = nodes.new('ShaderNodeNewGeometry')
    pos  = geom.outputs['Position']

    def row_dot(pos_socket, row):
        """P_local[i] = dot(P_world_xyz, M.row[i][:3]) + M.row[i][3]"""
        dot = nodes.new('ShaderNodeVectorMath')
        dot.operation = 'DOT_PRODUCT'
        dot.inputs[1].default_value = (float(row[0]), float(row[1]), float(row[2]))
        links.new(pos_socket, dot.inputs[0])
        add = nodes.new('ShaderNodeMath')
        add.operation = 'ADD'
        add.inputs[1].default_value = float(row[3])
        links.new(dot.outputs['Value'], add.inputs[0])
        return add.outputs['Value']

    x_l = row_dot(pos, r0)
    y_l = row_dot(pos, r1)
    z_l = row_dot(pos, r2)

    # depth = −Z_local (프로젝터 앞쪽이 음수이므로 부호 반전)
    neg_z               = nodes.new('ShaderNodeMath')
    neg_z.operation     = 'MULTIPLY'
    neg_z.inputs[1].default_value = -1.0
    links.new(z_l, neg_z.inputs[0])

    def proj_uv(coord_val, half_t):
        """coord / depth / half_t  ×0.5 + 0.5  →  UV [0,1]"""
        d = nodes.new('ShaderNodeMath'); d.operation = 'DIVIDE'
        links.new(coord_val,              d.inputs[0])
        links.new(neg_z.outputs['Value'], d.inputs[1])
        n = nodes.new('ShaderNodeMath'); n.operation = 'DIVIDE'
        n.inputs[1].default_value = half_t
        links.new(d.outputs['Value'], n.inputs[0])
        r = nodes.new('ShaderNodeMath'); r.operation = 'MULTIPLY_ADD'
        r.inputs[1].default_value = 0.5; r.inputs[2].default_value = 0.5
        links.new(n.outputs['Value'], r.inputs[0])
        return r.outputs['Value']

    u = proj_uv(x_l, half_tan)
    v = proj_uv(y_l, half_tan / aspect)

    comb = nodes.new('ShaderNodeCombineXYZ')
    comb.inputs['Z'].default_value = 0.0
    links.new(u, comb.inputs['X'])
    links.new(v, comb.inputs['Y'])
    links.new(comb.outputs['Vector'], img_tex.inputs['Vector'])

    # ── Emission 강도: Lambert 인자 → 3D 입체감 ─────────────────
    # strength = max(0.08, dot(N_world, proj_forward))
    # - 프로젝터를 마주보는 면: ~1.0  (밝음)
    # - 측면: ~0.5
    # - 등지는 면: 0.08 (최소값, 완전 검정 방지)
    dot_nl = nodes.new('ShaderNodeVectorMath')
    dot_nl.operation = 'DOT_PRODUCT'
    dot_nl.inputs[1].default_value = (float(proj_fwd.x),
                                      float(proj_fwd.y),
                                      float(proj_fwd.z))
    links.new(geom.outputs['Normal'], dot_nl.inputs[0])

    clamp = nodes.new('ShaderNodeMath')
    clamp.operation = 'MAXIMUM'
    clamp.inputs[1].default_value = 0.08
    links.new(dot_nl.outputs['Value'], clamp.inputs[0])

    links.new(clamp.outputs['Value'], emit.inputs['Strength'])

    return mat


# ─── 흰색 Lambertian 재질 ────────────────────────────────────────────────────

def _build_white_material() -> bpy.types.Material:
    """
    구조광 측정 대상의 표준 재질: 흰색 Lambertian (완전 확산 반사).

    실제 DLP 구조광 시스템에서 측정 물체는 반사율이 균일한 흰색 표면으로 간주.
    프린지 패턴은 Spot Light로 투영되므로 재질 자체는 순수 흰색으로 유지.
    """
    mat = bpy.data.materials.get(WHITE_MAT)
    if mat:
        return mat  # 이미 생성된 경우 재사용

    mat = bpy.data.materials.new(WHITE_MAT)
    mat.use_nodes = True
    nt    = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out  = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value          = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs['Roughness'].default_value           = 0.9   # 무광 흰색
    bsdf.inputs['Specular IOR Level'].default_value  = 0.0   # 반사 없음
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


# ─── 프로젝터 Spot Light에 프린지 패턴 부착 ──────────────────────────────────

def _setup_projector_light_fringe(bpy_img: bpy.types.Image,
                                   fov_deg: float, width: int, height: int,
                                   light_energy: float):
    """
    Spot Light에 프린지 이미지를 텍스처로 연결하여 실제 DLP 프로젝터를 시뮬레이션.

    작동 원리 (물리적으로 정확):
      - Blender가 각 광선의 방향을 계산할 때 해당 방향의 프린지 강도를 사용
      - 깊이가 다른 표면에 도달하는 광선은 서로 다른 방향에서 오므로 시차가 자동 발생
      - 별도 UV/행렬 계산 없이 Cycles/EEVEE의 광선 추적이 올바른 구조광 왜곡을 생성

    Light shader 노드 그래프:
      TexCoord.Normal (world-space 방출 방향)
        → VectorTransform (World → Object = light-local space)
        → SeparateXYZ → [X_local, Y_local, Z_local]
        → X / (−Z) / tan(fov/2)       * 0.5 + 0.5  ─┐
        → Y / (−Z) / (tan/aspect)     * 0.5 + 0.5  ─┤ → ImageTex → Emission → LightOutput
    """
    light_obj = bpy.data.objects.get(PROJ_LIGHT)
    if light_obj is None:
        print("[DLP] 경고: DLP_ProjectorLight 없음. Setup from View 먼저 실행.")
        return

    half_tan = math.tan(math.radians(fov_deg) / 2)
    aspect   = width / height

    ld            = light_obj.data
    ld.energy     = light_energy
    ld.spot_size  = math.radians(fov_deg * 1.15)   # 프린지 FOV보다 약간 넓게
    ld.spot_blend = 0.04                             # 경계 부드럽게
    ld.use_nodes  = True

    nodes = ld.node_tree.nodes
    links = ld.node_tree.links
    nodes.clear()

    # ── Terminal nodes ──────────────────────────────────────────
    out      = nodes.new('ShaderNodeOutputLight')
    emission = nodes.new('ShaderNodeEmission')
    links.new(emission.outputs['Emission'], out.inputs['Surface'])

    # ── Fringe image texture ────────────────────────────────────
    img_tex               = nodes.new('ShaderNodeTexImage')
    img_tex.image         = bpy_img
    img_tex.extension     = 'CLIP'       # 투영 범위 밖은 검정
    img_tex.interpolation = 'Linear'
    links.new(img_tex.outputs['Color'], emission.inputs['Color'])

    # ── 방출 방향 → light-local space ──────────────────────────
    # TexCoord.Normal: 이 광선이 향하는 방향 (world space)
    tex_coord = nodes.new('ShaderNodeTexCoord')

    # World → light 오브젝트의 local space 로 변환
    vec_xfm              = nodes.new('ShaderNodeVectorTransform')
    vec_xfm.vector_type  = 'VECTOR'
    vec_xfm.convert_from = 'WORLD'
    vec_xfm.convert_to   = 'OBJECT'   # light 자신의 좌표계
    links.new(tex_coord.outputs['Normal'], vec_xfm.inputs['Vector'])

    sep = nodes.new('ShaderNodeSeparateXYZ')
    links.new(vec_xfm.outputs['Vector'], sep.inputs['Vector'])

    # depth = −Z_local  (light의 forward 방향 = local −Z)
    neg_z                       = nodes.new('ShaderNodeMath')
    neg_z.operation             = 'MULTIPLY'
    neg_z.inputs[1].default_value = -1.0
    links.new(sep.outputs['Z'], neg_z.inputs[0])

    def proj(coord_socket, half_t):
        """coord / depth / half_t * 0.5 + 0.5  →  UV [0,1]"""
        d = nodes.new('ShaderNodeMath'); d.operation = 'DIVIDE'
        links.new(coord_socket,              d.inputs[0])
        links.new(neg_z.outputs['Value'],    d.inputs[1])
        n = nodes.new('ShaderNodeMath'); n.operation = 'DIVIDE'
        n.inputs[1].default_value = half_t
        links.new(d.outputs['Value'], n.inputs[0])
        r = nodes.new('ShaderNodeMath'); r.operation = 'MULTIPLY_ADD'
        r.inputs[1].default_value = 0.5; r.inputs[2].default_value = 0.5
        links.new(n.outputs['Value'], r.inputs[0])
        return r.outputs['Value']

    u = proj(sep.outputs['X'], half_tan)
    v = proj(sep.outputs['Y'], half_tan / aspect)

    comb = nodes.new('ShaderNodeCombineXYZ')
    comb.inputs['Z'].default_value = 0.0
    links.new(u, comb.inputs['X'])
    links.new(v, comb.inputs['Y'])
    links.new(comb.outputs['Vector'], img_tex.inputs['Vector'])


def _set_rendered_preview(context):
    """뷰포트를 Rendered Preview 모드로 전환."""
    sd = getattr(context, 'space_data', None)
    if sd and getattr(sd, 'type', '') == 'VIEW_3D':
        sd.shading.type = 'RENDERED'


def apply_to_selection(arr: np.ndarray, props) -> str:
    """
    구조광 패턴을 선택된 메시에 적용.

    ① Spot Light → 순수 백색광 (프린지 없음, 보조 조명 역할)
    ② 표면 재질 → Emission + 프로젝터 원근 UV (깊이 반영)

    결과: 각 픽셀의 3D 위치에서 프로젝터 시점의 UV를 계산하므로
          깊이가 다른 면에서 프린지가 서로 다르게 이동 = 시차 = 구조광 효과.
    """
    bpy_img = _arr_to_image(arr, FRINGE_IMG)

    # Spot Light를 순수 백색광으로 리셋 (프린지는 재질이 담당)
    light_obj = bpy.data.objects.get(PROJ_LIGHT)
    if light_obj:
        ld           = light_obj.data
        ld.use_nodes = False          # 노드 제거 → 기본 흰색
        ld.energy    = props.light_energy * 0.15   # 보조 조명 (15%)
        ld.color     = (1.0, 1.0, 1.0)

    # 표면 재질: Emission + 프로젝터 원근 UV (핵심 깊이 반영 로직)
    mat = _build_material(bpy_img, props.fov_deg, props.width, props.height)

    targets = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not targets and bpy.context.active_object and bpy.context.active_object.type == 'MESH':
        targets = [bpy.context.active_object]

    for obj in targets:
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    return (f"구조광 투영 — {len(targets)} 메시  "
            f"[step {props.current_step + 1}/{pattern_count(props)}]")


# ─── Properties ──────────────────────────────────────────────────────────────

class DLPProperties(PropertyGroup):

    # Pattern ─────────────────────────────────────────────────
    pattern_type: EnumProperty(
        name="Type", default='phase_shift',
        items=[
            ('phase_shift', "Phase Shift",   "Sinusoidal N-step phase shifting fringe"),
            ('gray_code',   "Gray Code",     "Binary Gray code (2^N_bits stripes)"),
            ('binary',      "Binary Stripe", "Sequential binary stripe patterns"),
            ('color_coded', "Color Stripes", "Single-shot De Bruijn/Hamming/self-equalizing color stripe pattern"),
            ('color_phase', "RGB Phase",     "Single-shot 3-step phase shift multiplexed into R/G/B channels"),
        ],
    )
    N:            IntProperty(name="Steps",    default=3,    min=3, max=16,
                              description="Number of phase-shift steps")
    N_bits:       IntProperty(name="Bits",     default=4,    min=1, max=8,
                              description="Gray code bits (= number of patterns)")
    N_binary:     IntProperty(name="Patterns", default=5,    min=1, max=10,
                              description="Number of binary stripe patterns")
    color_code_mode: EnumProperty(
        name="Color Code", default='hamming',
        items=[
            (
                'debruijn',
                "De Bruijn",
                "Constrained De Bruijn sequence: unique windows and no equal neighboring colors",
            ),
            (
                'hamming',
                "Hamming",
                "RGB bit code with Hamming distance 1 between neighboring stripes",
            ),
            (
                'self_equalizing',
                "Self-equalizing",
                "De Bruijn code rendered as color/complement pairs to normalize local albedo and illumination",
            ),
        ],
    )
    color_k:      IntProperty(
        name="Colors (k)", default=5, min=3, max=6,
        description="De Bruijn/self-equalizing 패턴에 사용할 색상 수",
    )
    color_n:      IntProperty(
        name="Window (n)", default=4, min=3, max=6,
        description="절대 stripe 위치를 식별하는 연속 색상 window 길이",
    )
    color_precompensate: BoolProperty(
        name="Crosstalk Precompensation",
        default=False,
        description=(
            "Color Matrix를 camera_rgb = M * projector_rgb 응답으로 보고 "
            "M의 역행렬을 RGB 투영 패턴에 적용"
        ),
    )
    color_response_matrix: StringProperty(
        name="Color Matrix",
        default="1,0,0;0,1,0;0,0,1",
        description=(
            "3x3 projector-to-camera RGB response/crosstalk matrix. "
            "Rows are separated by ';' and columns by ','"
        ),
    )
    color_phase_freq: FloatProperty(
        name="RGB Phase Freq",
        default=64.0,
        min=1.0,
        max=256.0,
        description=(
            "Single-shot RGB phase fringe cycles across image width. "
            "At 1280px, 64 cycles gives a 20px period"
        ),
    )
    freq:         FloatProperty(name="Freq",   default=16.0, min=1.0, max=64.0,
                                description="Fringe cycles across image width")
    current_step: IntProperty(name="Step",     default=0,    min=0)
    combo_frequencies: StringProperty(
        name="Phase Freqs", default="16,32,64",
        description=(
            "Render Gray+Phase Set에서 사용할 phase-shift 주파수 목록 (쉼표 구분).\n"
            "각 주파수는 output/<freq>/ 폴더에 N-step 패턴으로 렌더됨.\n"
            "Gray Code는 output/gray/ 에 렌더됨."
        )
    )

    # Geometry ────────────────────────────────────────────────
    fov_deg:    FloatProperty(name="FOV (°)",          default=45.0, min=10.0, max=120.0)
    proj_angle: FloatProperty(
        name="Projector Angle (°)", default=15.0, min=-90.0, max=90.0,
        description=(
            "프로젝터가 카메라로부터 오빗 중심을 기준으로 회전하는 각도.\n"
            "양수(+) = 카메라 기준 왼쪽,  음수(−) = 오른쪽.\n"
            "각도가 클수록 깊이 민감도(시차) 증가."
        )
    )

    # Rotation / Orbit ────────────────────────────────────────
    rotate_x:  FloatProperty(name="X (°)", default=0.0, min=-360.0, max=360.0)
    rotate_y:  FloatProperty(name="Y (°)", default=0.0, min=-360.0, max=360.0)
    rotate_z:  FloatProperty(name="Z (°)", default=0.0, min=-360.0, max=360.0)
    orbit_n:   IntProperty(name="Orbit Steps", default=1, min=1, max=36,
                           description="Number of evenly-spaced Y-axis rotation steps for batch render")

    # Lighting ────────────────────────────────────────────────
    light_energy: FloatProperty(
        name="Light (W)", default=5000.0, min=1.0, max=100000.0,
        description="Projector spot light energy in Watts. Increase if scene is too dark."
    )
    ambient: FloatProperty(
        name="Ambient", default=0.1, min=0.0, max=1.0,
        description="World background strength (fill light for shadowed areas)"
    )
    exposure: FloatProperty(
        name="Exposure (EV)", default=1.0, min=-6.0, max=6.0,
        description="Render exposure in EV. +1 EV = 2× brighter output image."
    )

    # Render ──────────────────────────────────────────────────
    samples:    IntProperty(name="Samples",  default=64,   min=1,  max=4096)
    width:      IntProperty(name="Width",    default=1280, min=64, max=4096)
    height:     IntProperty(name="Height",   default=720,  min=64, max=4096)
    output_dir: StringProperty(name="Output", default="//dlp_output", subtype='DIR_PATH')

    # Status ──────────────────────────────────────────────────
    status: StringProperty(name="Status", default="Ready.")


# ─── Operators ───────────────────────────────────────────────────────────────

class DLP_OT_setup_scene(Operator):
    bl_idname  = "dlp.setup_scene"
    bl_label   = "Setup from View"
    bl_description = (
        "현재 뷰포트 시점 기준으로 DLP 씬 자동 배치\n"
        "  MainCamera   = 뷰포트 정확히 동일 위치 (렌더 카메라)\n"
        "  ProjectorCam = Orbit Center 기준 N도 회전된 위치 (패턴 투영)\n"
        "  Fill Light   = MainCamera 방향에서 백색광 조사"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import mathutils
        props = context.scene.dlp_props
        r3d   = context.space_data.region_3d

        # ── 뷰포트 상태 읽기 ────────────────────────────────────
        quat  = r3d.view_rotation                        # 뷰 회전 쿼터니언
        orbit = mathutils.Vector(r3d.view_location)      # 오빗 중심 (look-at)
        dist  = r3d.view_distance                        # eye ↔ orbit 거리

        # ── 메인 카메라: 뷰포트와 완전히 동일한 위치 ───────────
        # "내가 바라보고 있는 방향에서 빛 쏴줘" → 여기에 카메라+백색광 배치
        eye      = orbit + quat @ mathutils.Vector((0.0, 0.0, dist))
        main_pos = eye   # ← 카메라는 뷰포트 eye와 동일

        # ── 프로젝터: 오빗 중심 기준으로 N도 회전 ──────────────
        # 카메라 UP 벡터를 축으로 eye를 orbit 기준 회전
        # proj_angle > 0: 카메라에서 봤을 때 왼쪽
        # proj_angle < 0: 오른쪽
        cam_up = quat @ mathutils.Vector((0.0, 1.0, 0.0))   # 카메라 상향 벡터(world)
        rot_mat = mathutils.Matrix.Rotation(
            math.radians(props.proj_angle), 4, cam_up
        )
        eye_from_orbit = eye - orbit          # 오빗 중심 기준 eye 벡터
        proj_pos = orbit + rot_mat @ eye_from_orbit   # 회전 후 다시 world 좌표

        msg = _configure_scene(proj_pos, main_pos, orbit, props)
        props.status = msg
        return {'FINISHED'}


class DLP_OT_apply_pattern(Operator):
    bl_idname  = "dlp.apply_pattern"
    bl_label   = "Apply to Selection"
    bl_description = "Project current pattern onto selected mesh(es)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props  = context.scene.dlp_props
        try:
            arr = make_pattern(props, props.current_step)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            props.status = str(exc)
            return {'CANCELLED'}
        msg    = apply_to_selection(arr, props)
        props.status = msg
        _set_rendered_preview(context)   # 자동으로 Rendered Preview 전환
        return {'FINISHED'}


class DLP_OT_step_next(Operator):
    bl_idname  = "dlp.step_next"
    bl_label   = ""
    bl_description = "Next pattern step (auto-applies to selection)"

    def execute(self, context):
        props = context.scene.dlp_props
        total = pattern_count(props)
        props.current_step = (props.current_step + 1) % total
        try:
            arr = make_pattern(props, props.current_step)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            props.status = str(exc)
            return {'CANCELLED'}
        props.status = apply_to_selection(arr, props)
        _set_rendered_preview(context)
        return {'FINISHED'}


class DLP_OT_step_prev(Operator):
    bl_idname  = "dlp.step_prev"
    bl_label   = ""
    bl_description = "Previous pattern step (auto-applies to selection)"

    def execute(self, context):
        props = context.scene.dlp_props
        total = pattern_count(props)
        props.current_step = (props.current_step - 1) % total
        try:
            arr = make_pattern(props, props.current_step)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            props.status = str(exc)
            return {'CANCELLED'}
        props.status = apply_to_selection(arr, props)
        _set_rendered_preview(context)
        return {'FINISHED'}


class DLP_OT_apply_rotation(Operator):
    bl_idname  = "dlp.apply_rotation"
    bl_label   = "Apply Rotation"
    bl_description = "Rotate active mesh by Rotation (X,Y,Z) values"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.dlp_props
        obj   = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a mesh object first")
            return {'CANCELLED'}
        obj.rotation_euler = (
            math.radians(props.rotate_x),
            math.radians(props.rotate_y),
            math.radians(props.rotate_z),
        )
        props.status = (f"Rotation applied: ({props.rotate_x:.1f}°, "
                        f"{props.rotate_y:.1f}°, {props.rotate_z:.1f}°)")
        return {'FINISHED'}


class DLP_OT_hide_overlays(Operator):
    bl_idname  = "dlp.hide_overlays"
    bl_label   = "Toggle Grid"
    bl_description = "뷰포트 그리드·축·커서 오버레이를 표시/숨김 전환"
    bl_options = {'REGISTER'}

    def execute(self, context):
        overlay = context.space_data.overlay
        # 현재 상태의 반전 (floor가 켜져 있으면 전부 끄기, 꺼져 있으면 전부 켜기)
        hide = overlay.show_floor
        overlay.show_floor            = not hide
        overlay.show_axis_x           = not hide
        overlay.show_axis_y           = not hide
        overlay.show_axis_z           = not hide
        overlay.show_cursor           = not hide
        overlay.show_object_origins   = not hide
        state = "숨김" if hide else "표시"
        context.scene.dlp_props.status = f"그리드/오버레이 {state}"
        return {'FINISHED'}


class DLP_OT_viewport_capture(Operator):
    bl_idname  = "dlp.viewport_capture"
    bl_label   = "Viewport Capture"
    bl_description = (
        "뷰포트를 그대로 PNG로 저장 (그리드 자동 숨김, EEVEE Rendered Preview 사용).\n"
        "Blender 화면과 동일한 밝기로 캡처됩니다."
    )
    bl_options = {'REGISTER'}

    def execute(self, context):
        props   = context.scene.dlp_props
        overlay = context.space_data.overlay

        # ── 뷰포트를 Rendered Preview로 전환 ──────────────────────
        context.space_data.shading.type = 'RENDERED'

        # ── 오버레이 저장 후 숨김 ─────────────────────────────────
        saved = {
            'show_floor':          overlay.show_floor,
            'show_axis_x':         overlay.show_axis_x,
            'show_axis_y':         overlay.show_axis_y,
            'show_axis_z':         overlay.show_axis_z,
            'show_cursor':         overlay.show_cursor,
            'show_object_origins': overlay.show_object_origins,
        }
        for key in saved:
            setattr(overlay, key, False)

        # ── EEVEE 렌더 설정 적용 ──────────────────────────────────
        _apply_world_render(props)

        # ── 출력 경로 설정 ────────────────────────────────────────
        out_dir = Path(bpy.path.abspath(props.output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        step    = props.current_step
        out_path = str(out_dir / f"viewport_step{step:02d}.png")
        context.scene.render.filepath = out_path

        # ── OpenGL 캡처 (뷰포트 그대로, 활성 카메라 기준) ─────────
        bpy.ops.render.opengl(write_still=True, view_context=False)

        # ── 오버레이 복원 ─────────────────────────────────────────
        for key, val in saved.items():
            setattr(overlay, key, val)

        props.status = f"캡처 완료 → {out_path}"
        self.report({'INFO'}, props.status)
        return {'FINISHED'}


class DLP_OT_render_combined(Operator):
    bl_idname  = "dlp.render_combined"
    bl_label   = "Export All"
    bl_description = (
        "Gray Code(+White/Black 레퍼런스), 여러 주파수의 Phase Shift 패턴, "
        "선택한 single-shot Color Stripe, RGB 다중화 PSP 패턴, "
        "카메라/프로젝터 Calibration 정보를 한 번에 export.\n"
        "  output/gray/                  ← Gray Code (N_bits 패턴 + white/black)\n"
        "  output/<freq>/                ← 'Phase Freqs'에 입력한 각 주파수별 N-step Phase Shift\n"
        "  output/color/                 ← 선택한 컬러 스트라이프 (단일 촬영 + white)\n"
        "  output/color_pattern_info.json    ← 컬러 코딩 디코딩용 팔레트/시퀀스 정보\n"
        "  output/color_phase/           ← RGB 다중화 PSP (단일 촬영)\n"
        "  output/color_calibration/     ← RGB crosstalk 측정용 원색/white/black 패턴\n"
        "  output/camera_calibration.json    ← DLP_MainCamera intrinsic/extrinsic\n"
        "  output/projector_calibration.json ← DLP_ProjectorCamera intrinsic/extrinsic\n"
        "Blender will be unresponsive during rendering."
    )
    bl_options = {'REGISTER'}

    @staticmethod
    def _render_set(props, scene, out_dir: Path, orbit_step: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        n_pat = pattern_count(props)
        scene.render.image_settings.color_mode = (
            'RGB' if props.pattern_type in ('color_coded', 'color_phase') else 'BW'
        )

        for pat_step in range(n_pat):
            arr = make_pattern(props, pat_step)
            if orbit_step == 0:
                tmp = _arr_to_image(arr, "_dlp_tmp")
                tmp.filepath_raw = str(out_dir / f"pattern_{pat_step:02d}.png")
                tmp.file_format  = 'PNG'
                tmp.save()

            apply_to_selection(arr, props)
            scene.render.filepath = str(out_dir / f"render_{pat_step:02d}.png")
            bpy.ops.render.render(write_still=True)

            props.status = f"{out_dir.name}  pattern {pat_step + 1}/{n_pat}"
            print(f"[DLP] {props.status}")

        if props.pattern_type in ('gray_code', 'color_coded'):
            refs = (("white", 255), ("black", 0)) if props.pattern_type == 'gray_code' else (("white", 255),)
            for ref_name, ref_val in refs:
                arr = _flat(ref_val, props.width, props.height)
                if orbit_step == 0:
                    tmp = _arr_to_image(arr, "_dlp_tmp")
                    tmp.filepath_raw = str(out_dir / f"pattern_{ref_name}.png")
                    tmp.file_format  = 'PNG'
                    tmp.save()

                apply_to_selection(arr, props)
                scene.render.filepath = str(out_dir / f"render_{ref_name}.png")
                bpy.ops.render.render(write_still=True)

                props.status = f"{out_dir.name}  reference {ref_name}"
                print(f"[DLP] {props.status}")

    @staticmethod
    def _save_color_calibration_patterns(props, out_dir: Path):
        """실제 projector-camera RGB 응답행렬을 측정할 때 사용할 원색 패턴."""
        out_dir.mkdir(parents=True, exist_ok=True)
        references = {
            "red":   (255, 0, 0),
            "green": (0, 255, 0),
            "blue":  (0, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        for name, rgb in references.items():
            arr = _solid_color(rgb, props.width, props.height)
            tmp = _arr_to_image(arr, "_dlp_tmp")
            tmp.filepath_raw = str(out_dir / f"pattern_{name}.png")
            tmp.file_format = 'PNG'
            tmp.save()

    def execute(self, context):
        props    = context.scene.dlp_props
        scene    = context.scene
        out_base = Path(bpy.path.abspath(props.output_dir))

        if bpy.data.objects.get(PROJ_CAM) is None or bpy.data.objects.get(MAIN_CAM) is None:
            self.report({'ERROR'}, "먼저 'Setup from View'를 실행하세요")
            return {'CANCELLED'}

        mesh_obj = context.active_object
        if mesh_obj is None or mesh_obj.type != 'MESH':
            self.report({'WARNING'}, "Select a target mesh first")
            return {'CANCELLED'}

        try:
            freqs = [float(f.strip()) for f in props.combo_frequencies.split(',') if f.strip()]
        except ValueError:
            self.report({'ERROR'}, "Phase Freqs 형식 오류 (예: 16,32,64)")
            return {'CANCELLED'}

        if not freqs:
            self.report({'ERROR'}, "Phase Freqs가 비어있습니다")
            return {'CANCELLED'}

        try:
            response_matrix = _parse_color_matrix(props.color_response_matrix)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            props.status = str(exc)
            return {'CANCELLED'}

        _apply_world_render(props)

        orbit    = props.orbit_n
        step_deg = 360.0 / orbit

        orig_type = props.pattern_type
        orig_freq = props.freq

        try:
            for orbit_step in range(orbit):
                ry = props.rotate_y + orbit_step * step_deg
                rot_label = f"rot_{int(ry) % 360:03d}deg"

                mesh_obj.rotation_euler = (
                    math.radians(props.rotate_x),
                    math.radians(ry),
                    math.radians(props.rotate_z),
                )

                # ── Gray Code ────────────────────────────────────────
                props.pattern_type = 'gray_code'
                gray_dir = out_base / "gray" / rot_label if orbit > 1 else out_base / "gray"
                self._render_set(props, scene, gray_dir, orbit_step)

                # ── Phase Shift (주파수별) ─────────────────────────────
                props.pattern_type = 'phase_shift'
                for freq in freqs:
                    props.freq = freq
                    freq_label = f"{freq:g}"
                    freq_dir = out_base / freq_label / rot_label if orbit > 1 else out_base / freq_label
                    self._render_set(props, scene, freq_dir, orbit_step)

                # ── Color stripe code (single-shot) ────────────────────
                props.pattern_type = 'color_coded'
                color_dir = out_base / "color" / rot_label if orbit > 1 else out_base / "color"
                self._render_set(props, scene, color_dir, orbit_step)

                # ── RGB-multiplexed 3-step PSP (single-shot) ────────────
                props.pattern_type = 'color_phase'
                color_phase_dir = (
                    out_base / "color_phase" / rot_label
                    if orbit > 1 else out_base / "color_phase"
                )
                self._render_set(props, scene, color_phase_dir, orbit_step)
        finally:
            props.pattern_type = orig_type
            props.freq         = orig_freq

        # ── 카메라 / 프로젝터 Calibration 정보 export ───────────────
        import json
        out_base.mkdir(parents=True, exist_ok=True)

        cam_calib = _build_calibration_dict(props, MAIN_CAM, MAIN_CAM)
        with open(out_base / "camera_calibration.json", 'w', encoding='utf-8') as f:
            json.dump(cam_calib, f, indent=2)

        proj_calib = _build_calibration_dict(props, PROJ_CAM, MAIN_CAM)
        with open(out_base / "projector_calibration.json", 'w', encoding='utf-8') as f:
            json.dump(proj_calib, f, indent=2)

        # ── 컬러 코딩 디코딩용 패턴 정보 export ───────────────────────
        color_data = _color_code_data(props)
        color_info = {
            "mode":              color_data["mode"],
            "k":                 3 if color_data["mode"] == "hamming" else props.color_k,
            "n":                 props.color_n,
            "decode_window":     color_data["decode_window"],
            "logical_stripes":   color_data["logical_stripes"],
            "projected_stripes": color_data["projected_stripes"],
            "sequence":          color_data["sequence"],
            "palette":           [list(c) for c in color_data["palette"]],
            "adjacency":         color_data["adjacency"],
            "self_equalizing":   color_data["self_equalizing"],
            "pair_layout": (
                "[palette[symbol], 255 - palette[symbol]]"
                if color_data["self_equalizing"] else None
            ),
            "image_size": [props.width, props.height],
        }
        with open(out_base / "color_pattern_info.json", 'w', encoding='utf-8') as f:
            json.dump(color_info, f, indent=2)

        color_phase_info = {
            "frequency": props.color_phase_freq,
            "period_pixels": props.width / props.color_phase_freq,
            "channel_order": ["R:I0", "G:I1", "B:I2"],
            "phase_shifts_radians": [
                0.0,
                -2.0 * math.pi / 3.0,
                -4.0 * math.pi / 3.0,
            ],
            "wrapped_phase_formula": (
                "atan2(sqrt(3) * (G - B), 2 * R - G - B)"
            ),
            "response_model": "camera_rgb = response_matrix @ projector_rgb",
            "response_matrix": response_matrix.tolist(),
            "projector_precompensation": props.color_precompensate,
            "inverse_matrix": (
                np.linalg.inv(response_matrix).tolist()
                if props.color_precompensate else None
            ),
            "image_size": [props.width, props.height],
        }
        with open(out_base / "color_phase_info.json", 'w', encoding='utf-8') as f:
            json.dump(color_phase_info, f, indent=2)

        self._save_color_calibration_patterns(
            props, out_base / "color_calibration"
        )

        props.status = (
            f"Done! gray + {len(freqs)} phase sets + "
            f"{color_info['mode']}({color_info['projected_stripes']} stripes) + "
            f"RGB phase + calibration → {out_base}"
        )
        self.report({'INFO'}, props.status)
        return {'FINISHED'}


# ─── Panel ───────────────────────────────────────────────────────────────────

class DLP_PT_main(Panel):
    bl_label       = "DLP Simulator"
    bl_idname      = "DLP_PT_main"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'DLP'

    def draw(self, context):
        layout = self.layout
        props  = context.scene.dlp_props

        # ── Scene setup ──────────────────────────────────────────
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Projector / Camera", icon='CAMERA_DATA')
        col.prop(props, "fov_deg")

        # 프로젝터 각도 슬라이더
        row = col.row(align=True)
        row.label(text="Proj Angle:")
        row.prop(props, "proj_angle", text="")
        col.label(text="  + = 왼쪽 오프셋 / − = 오른쪽 오프셋", icon='INFO')

        col.separator()
        col.prop(props, "light_energy")
        col.prop(props, "ambient")
        box.operator("dlp.setup_scene", icon='SCENE_DATA', text="Setup from View  👁")

        # ── Pattern ──────────────────────────────────────────────
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Pattern", icon='TEXTURE')
        col.prop(props, "pattern_type", text="")

        if props.pattern_type == 'phase_shift':
            row = col.row(align=True)
            row.prop(props, "N")
            row.prop(props, "freq")
        elif props.pattern_type == 'gray_code':
            col.prop(props, "N_bits")
        elif props.pattern_type == 'color_coded':
            col.prop(props, "color_code_mode", text="")
            if props.color_code_mode != 'hamming':
                col.prop(props, "color_k")
            col.prop(props, "color_n")
            data = _color_code_data(props)
            logical = data["logical_stripes"]
            projected = data["projected_stripes"]
            col.label(
                text=f"Logical: {logical} / projected: {projected}",
                icon='INFO',
            )
            col.label(
                text=f"Width: ~{props.width / projected:.1f}px/stripe"
            )
        elif props.pattern_type == 'color_phase':
            col.prop(props, "color_phase_freq")
            period = props.width / props.color_phase_freq
            col.label(text=f"Period: {period:.1f}px/cycle", icon='INFO')
            if period < 8.0:
                col.label(text="Too fine: blur/crosstalk risk", icon='ERROR')
            elif period > 32.0:
                col.label(text="Coarse fringe: lower depth precision", icon='ERROR')
            col.label(text="Wrapped phase: order/prior required", icon='INFO')
        else:
            col.prop(props, "N_binary")

        if props.pattern_type in ('color_coded', 'color_phase'):
            col.separator()
            col.prop(props, "color_precompensate")
            col.prop(props, "color_response_matrix")

        # Step navigator
        total = pattern_count(props)
        col.separator()
        nav = col.row(align=True)
        nav.operator("dlp.step_prev", text="", icon='TRIA_LEFT')
        nav.label(text=f"  Step {props.current_step + 1} / {total}  ")
        nav.operator("dlp.step_next", text="", icon='TRIA_RIGHT')
        box.operator("dlp.apply_pattern", icon='MATERIAL')

        # ── Rotation ─────────────────────────────────────────────
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Rotation", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        row = col.row(align=True)
        row.prop(props, "rotate_x")
        row.prop(props, "rotate_y")
        row.prop(props, "rotate_z")
        col.prop(props, "orbit_n")
        box.operator("dlp.apply_rotation", icon='OBJECT_ORIGIN')

        # ── Viewport / Grid ──────────────────────────────────────
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Viewport", icon='SHADING_RENDERED')
        row = col.row(align=True)
        row.operator("dlp.hide_overlays",     icon='GRID',          text="Toggle Grid")
        row.operator("dlp.viewport_capture",  icon='IMAGE_DATA',    text="Capture")

        # ── Render (EEVEE full render) ────────────────────────────
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Batch Render (EEVEE)", icon='RENDER_STILL')
        row = col.row(align=True)
        row.prop(props, "width")
        row.prop(props, "height")
        col.prop(props, "output_dir", text="")
        col.prop(props, "combo_frequencies")
        box.operator("dlp.render_combined", icon='RENDER_RESULT')

        # ── Status ───────────────────────────────────────────────
        if props.status:
            layout.label(text=props.status, icon='CHECKMARK')


# ─── Registration ─────────────────────────────────────────────────────────────

_classes = [
    DLPProperties,
    DLP_OT_setup_scene,
    DLP_OT_apply_pattern,
    DLP_OT_step_next,
    DLP_OT_step_prev,
    DLP_OT_apply_rotation,
    DLP_OT_hide_overlays,
    DLP_OT_viewport_capture,
    DLP_OT_render_combined,
    DLP_PT_main,
]

def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dlp_props = PointerProperty(type=DLPProperties)

def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.dlp_props

if __name__ == "__main__":
    register()
