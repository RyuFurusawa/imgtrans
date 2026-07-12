"""HDR EOTF/OETF ヘルパと HDR→SDR 変換。

PQ/HLG の信号値とリニア光量の相互変換、トーンマップ、
HDR PQ PNG → sRGB PNG への変換 (hdr_pq_to_srgb) を提供する。
"""
import os
import subprocess
import json
import cv2
import numpy as np


def _pq_eotf(N):
    """PQ (SMPTE ST 2084) EOTF: 信号値 [0,1] → リニアライト [0, 10000] cd/m²"""
    m1 = 0.1593017578125       # = 2610 / 16384
    m2 = 78.84375              # = 2523 / 32 * 128
    c1 = 0.8359375             # = 3424 / 4096
    c2 = 18.8515625            # = 2413 / 128
    c3 = 18.6875               # = 2392 / 128
    Np = np.maximum(N, 0.0) ** (1.0 / m2)
    Y = np.maximum(Np - c1, 0.0) / (c2 - c3 * Np)
    return Y ** (1.0 / m1) * 10000.0  # cd/m²

def _srgb_oetf(linear):
    """リニア [0,1] → sRGB ガンマ [0,1]"""
    out = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055
    )
    return np.clip(out, 0.0, 1.0)

# BT.2020 → BT.709 色域変換行列 (3×3, Bradford適応)
_MAT_2020_TO_709 = np.array([
    [ 1.6605, -0.5877, -0.0728],
    [-0.1246,  1.1329, -0.0083],
    [-0.0182, -0.1006,  1.1187],
], dtype=np.float64)

def _hlg_eotf(E, peak_nits=1000.0, gamma=1.2):
    """HLG (ARIB STD-B67 / BT.2100) inverse-OETF + OOTF → ディスプレイ光 cd/m²"""
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * np.log(4.0 * a)
    E = np.clip(E, 0.0, 1.0)
    L = np.where(
        E <= 0.5,
        (E ** 2) / 3.0,
        (np.exp((E - c) / a) + b) / 12.0,
    )
    return peak_nits * np.maximum(L, 0.0) ** gamma

def _tonemap_hdr_rgb01_to_srgb(rgb01_hdr, transfer, max_nits=1000.0, tonemap='aces'):
    """HDR信号 (PQ / HLG) の RGB[0,1] を sRGB RGB[0,1] にトーンマップする。
    animationout 系で PQ映像を matplotlib に貼るときに褪せて見える問題を補正するための簡易版。
    transfer が PQ/HLG でなければ入力をそのまま返す。"""
    rgb01_hdr = np.asarray(rgb01_hdr, dtype=np.float64)
    if transfer == "smpte2084":
        linear_nits = _pq_eotf(rgb01_hdr)
    elif transfer == "arib-std-b67":
        linear_nits = _hlg_eotf(rgb01_hdr, peak_nits=max_nits)
    else:
        return np.clip(rgb01_hdr, 0.0, 1.0)

    h, w, _ = linear_nits.shape
    pixels_709 = linear_nits.reshape(-1, 3) @ _MAT_2020_TO_709.T
    linear_709 = np.maximum(pixels_709, 0.0).reshape(h, w, 3)
    linear_01 = linear_709 / max_nits

    if tonemap == 'aces':
        a_, b_, c_, d_, e_ = 2.51, 0.03, 2.43, 0.59, 0.14
        x = linear_01
        mapped = (x * (a_ * x + b_)) / (x * (c_ * x + d_) + e_)
    elif tonemap == 'reinhard':
        mapped = linear_01 / (1.0 + linear_01)
    else:
        mapped = linear_01
    mapped = np.clip(mapped, 0.0, 1.0)
    return _srgb_oetf(mapped)

def _probe_video_transfer(path):
    """ffprobe で動画の color_transfer を取得。失敗時は None。"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=color_transfer",
            "-of", "json", path,
        ]
        out = subprocess.check_output(cmd).decode("utf-8", errors="ignore")
        info = json.loads(out)
        st = (info.get("streams") or [{}])[0]
        return st.get("color_transfer")
    except Exception:
        return None

def hdr_pq_to_srgb(input_path, output_path=None,
                    exposure=1.0, max_nits=1000.0,
                    tonemap='aces',
                    output_bit_depth=16,
                    embed_profile=True):
    """HDR PQ (Rec.2100) PNG を sRGB にトーンマッピング変換する。

    Photoshop での編集を前提に、sRGB IEC61966-2.1 プロファイル埋め込み対応。

    処理パイプライン:
        1. PQ EOTF → リニアライト (cd/m²)
        2. 露出調整 (exposure)
        3. BT.2020 → BT.709 色域変換
        4. トーンマッピング (HDR→SDR)
        5. sRGB OETF (ガンマ)
        6. ICC プロファイル埋め込み保存

    Args:
        input_path (str): 入力 HDR PNG パス (16bit RGB, PQ)
        output_path (str): 出力パス。None → 入力名 + '_srgb.png'
        exposure (float): 露出補正倍率 (デフォルト 1.0)。
            2.0 で +1EV (明るく)、0.5 で -1EV (暗く)。
        max_nits (float): トーンマップの基準最大輝度 (cd/m²)。
            デフォルト 1000 nit。ソース映像のピーク輝度に合わせて調整。
        tonemap (str): トーンマップ方式。
            'aces'    — ACES Filmic (映画的、S字カーブ、推奨)
            'reinhard'— Reinhard (自然なロールオフ)
            'hable'   — Hable/Uncharted2 (ゲーム系、コントラスト強め)
        output_bit_depth (int): 出力ビット深度 (8 or 16)。
            Photoshop 編集用は 16 推奨。
        embed_profile (bool): sRGB ICC プロファイルを埋め込むか。

    Returns:
        str: 出力ファイルパス

    Example:
        >>> hdr_pq_to_srgb('render_hdr.png')
        >>> hdr_pq_to_srgb('render_hdr.png', exposure=1.5, tonemap='aces')
        >>> # バッチ処理
        >>> import glob
        >>> for p in glob.glob('img/*.png'):
        ...     hdr_pq_to_srgb(p, tonemap='aces', max_nits=1000)
    """
    import subprocess

    # --- 入出力パス ---
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_srgb{ext}"

    # --- 読み込み (16bit BGR → RGB float64 [0,1]) ---
    img_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open: {input_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bit_depth = 16 if img_rgb.dtype == np.uint16 else 8
    max_val = 65535.0 if bit_depth == 16 else 255.0
    img_norm = img_rgb.astype(np.float64) / max_val  # [0, 1] PQ信号

    print(f"hdr_pq_to_srgb: {os.path.basename(input_path)}")
    print(f"  input: {img_rgb.shape}, {bit_depth}bit, "
          f"signal range [{img_norm.min():.4f}, {img_norm.max():.4f}]")

    # --- Step 1: PQ EOTF → リニアライト (cd/m²) ---
    linear_nits = _pq_eotf(img_norm)
    print(f"  linear nits: [{linear_nits.min():.2f}, {linear_nits.max():.2f}] cd/m²")

    # --- Step 2: 露出調整 ---
    linear_nits *= exposure

    # --- Step 3: BT.2020 → BT.709 ---
    h, w, _ = linear_nits.shape
    pixels = linear_nits.reshape(-1, 3)  # (N, 3)
    pixels_709 = pixels @ _MAT_2020_TO_709.T
    pixels_709 = np.maximum(pixels_709, 0.0)
    linear_709 = pixels_709.reshape(h, w, 3)

    # SDR基準輝度で正規化 (100 cd/m² = 1.0)
    linear_01 = linear_709 / max_nits

    # --- Step 4: トーンマッピング ---
    if tonemap == 'aces':
        # ACES Filmic (Narkowicz 2015 近似)
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        x = linear_01
        mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    elif tonemap == 'reinhard':
        mapped = linear_01 / (1.0 + linear_01)
    elif tonemap == 'hable':
        # Hable / Uncharted 2 Filmic
        def _hable_partial(x):
            A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
            return ((x * (A * x + C * B) + D * E) /
                    (x * (A * x + B) + D * F)) - E / F
        curr = _hable_partial(linear_01 * 2.0)  # 2.0 = exposure bias
        white_scale = 1.0 / _hable_partial(np.array([11.2]))  # W = 11.2
        mapped = curr * white_scale
    else:
        raise ValueError(f"Unknown tonemap: {tonemap!r} (use 'aces', 'reinhard', 'hable')")

    mapped = np.clip(mapped, 0.0, 1.0)

    # --- Step 5: sRGB OETF ---
    srgb = _srgb_oetf(mapped)

    # --- Step 6: 量子化 & 保存 ---
    if output_bit_depth == 16:
        out_img = (srgb * 65535.0 + 0.5).astype(np.uint16)
    else:
        out_img = (srgb * 255.0 + 0.5).astype(np.uint8)

    # RGB→BGR for cv2.imwrite
    cv2.imwrite(output_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    # --- sRGB ICC プロファイル埋め込み (macOS sips) ---
    if embed_profile:
        srgb_icc = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
        if os.path.isfile(srgb_icc):
            subprocess.run(
                ["sips", "--embedProfile", srgb_icc, output_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"  sRGB ICC profile embedded")
        else:
            print(f"  WARNING: sRGB ICC profile not found at {srgb_icc}")

    print(f"  output: {output_path} ({output_bit_depth}bit, tonemap={tonemap})")
    return output_path


