"""動画/シーケンス I/O のトップレベル関数群

- rendered_npys_to_mov : npy 群 → 動画ファイル
- rearrange_wide_video : 横長動画の並び替え
- rendered_mov_to_seq  : 動画 → 連番画像/UHDR シーケンス
- export_segments      : 動画を等分セグメント出力 (A/B群、reverse 対応)
"""
import os
import gc
import re
import glob
import json
import shutil
import subprocess
import psutil
import cv2
import numpy as np
import av

from ._utils import extract_base_string


def rendered_npys_to_mov(out_dir, npys_path=None, out_fps=30, sep_start=0, sep_end=None,
                         out_type=None, images=None, per_frame=False, separate_out=False):
    """
    npy データを動画に変換する統合関数。
    以下のすべてのユースケースに対応:

    1. sep-*.npy フォルダ → 1本の動画に結合（デフォルト）
       rendered_npys_to_mov(out_dir, npys_path='tmp/', sep_start=1, sep_end=6)

    2. 単一 .npy ファイル → 1本の動画
       rendered_npys_to_mov(out_dir, npys_path='output.npy')

    3. メモリ上の numpy 配列 → 1本の動画
       rendered_npys_to_mov(out_dir, images=my_array, out_fps=60)

    4. 1フレーム1npy のフォルダ → 1本の動画 (per_frame=True)
       rendered_npys_to_mov(out_dir, npys_path='frames/', per_frame=True)

    5. sep-*.npy フォルダ → sep毎に個別の動画 (separate_out=True)
       rendered_npys_to_mov(out_dir, npys_path='tmp/', separate_out=True)

    out_type:
        None   → 従来の cv2 mp4v（後方互換、4096px以下向け）
        1 (OUT_H264)         → FFmpeg H.264 (SDR 8bit)
        2 (OUT_H265)         → FFmpeg H.265 (HDR10 PQ 10bit)
        3 (OUT_PRORES_422)   → FFmpeg ProRes 422 HQ (HDR 10bit)
        4 (OUT_PRORES_4444)  → FFmpeg ProRes 4444 (HDR 10bit)
        5 (OUT_H265_SDR)     → FFmpeg H.265 (SDR BT.709 10bit)
        6 (OUT_PRORES_422_SDR) → FFmpeg ProRes 422 HQ (SDR BT.709)

    色順序の自動判定:
        uint8  npy → cv2由来 (BGR) → RGB に自動変換
        uint16 npy → PyAV由来 (RGB) → 変換なし
    """
    # --- out_type 定数 ---
    OT_H264         = 1
    OT_H265         = 2
    OT_PRORES_422   = 3
    OT_PRORES_4444  = 4
    OT_H265_SDR     = 5
    OT_PRORES_422_SDR = 6

    # ========================================================
    # 入力の正規化: すべてのモードを sorted_files リストに統一
    # ========================================================
    if images is not None:
        # --- モード3: メモリ上の numpy 配列 ---
        tmp_npy = os.path.join(os.path.dirname(out_dir) or '.', '_tmp_rendered_array.npy')
        np.save(tmp_npy, images)
        sorted_files = [tmp_npy]
        print(f"images array: shape={images.shape}, dtype={images.dtype}")
        del images
        gc.collect()
        _cleanup_tmp = [tmp_npy]
    elif npys_path is not None and os.path.isfile(npys_path):
        # --- モード2: 単一 .npy ファイル ---
        sorted_files = [os.path.abspath(npys_path)]
        _cleanup_tmp = []
    elif npys_path is not None and os.path.isdir(npys_path):
        if per_frame:
            # --- モード4: 1フレーム1npy のフォルダ ---
            npy_files = sorted(
                glob.glob(os.path.join(npys_path, '*.npy')),
                key=lambda f: int(re.search(r'(\d+)\.npy$', f).group(1)) if re.search(r'(\d+)\.npy$', f) else -1
            )
            if not npy_files:
                raise ValueError(f"No .npy files found in {npys_path}")
            sorted_files = npy_files
        else:
            # --- モード1/5: sep-*.npy フォルダ ---
            file_list = [f for f in os.listdir(npys_path) if not f.startswith('.') and f.endswith('.npy')]
            if not file_list:
                raise ValueError(f"No .npy files found in {npys_path}")
            base_string = extract_base_string(file_list[0])
            file_count = len(file_list)
            if sep_end is None:
                file_list = [base_string + "sep-" + str(i) + ".npy" for i in range(sep_start, sep_start + file_count)]
            else:
                file_list = [base_string + "sep-" + str(i) + ".npy" for i in range(sep_start, sep_end)]
            print(file_list)
            sorted_files = [os.path.abspath(os.path.join(npys_path, f)) for f in file_list]
        _cleanup_tmp = []
    else:
        raise ValueError("npys_path (ファイルまたはフォルダ) か images (numpy配列) のどちらかを指定してください")

    # ========================================================
    # モード5: separate_out=True → sep毎に個別動画出力
    # ========================================================
    if separate_out and not per_frame:
        for idx, npy_file in enumerate(sorted_files):
            sep_out_dir = out_dir + f"_sep{idx}"
            print(f"\n--- separate_out: {idx+1}/{len(sorted_files)} ---")
            rendered_npys_to_mov(sep_out_dir, npys_path=npy_file, out_fps=out_fps, out_type=out_type)
        return

    # ========================================================
    # サイズ・dtype 取得
    # ========================================================
    first_data = np.load(sorted_files[0])
    if per_frame:
        # 1フレーム1npy: shape=(H,W,3)
        height, width, channels = first_data.shape
        total_frames = len(sorted_files)
    else:
        # sep-npy or 単一npy: shape=(N,H,W,3)
        n_frames, height, width, channels = first_data.shape
        total_frames = n_frames * len(sorted_files)
    dtype = first_data.dtype
    print(f"resolution: {width}x{height}, dtype: {dtype}, total_frames: {total_frames}")
    del first_data
    gc.collect()

    print("video-preference")

    # ========================================================
    # 拡張子決定
    # ========================================================
    if out_type in (OT_PRORES_422, OT_PRORES_4444, OT_PRORES_422_SDR):
        ext = ".mov"
    elif out_type in (OT_H264, OT_H265, OT_H265_SDR):
        ext = ".mp4"
    else:
        ext = ".mp4"
    out_path = out_dir + ext

    # ========================================================
    # フレーム書き込み用ヘルパー
    # ========================================================
    def _prepare_frame(frame, use_16bit):
        """BGR/RGB判定・bit深度変換を行い、FFmpegに送るbytesを返す"""
        # uint8 → cv2由来 (BGR) → RGB に変換
        # uint16 → PyAV由来 (RGB) → そのまま
        if frame.dtype == np.uint8:
            frame = frame[:, :, ::-1]  # BGR→RGB
        if use_16bit:
            if frame.dtype == np.uint8:
                return (frame.astype(np.uint16) * 257).tobytes()
            else:
                return frame.astype(np.uint16).tobytes()
        else:
            return frame.astype(np.uint8).tobytes()

    def _print_progress(frame_number, total_frames):
        progressbarsize = 50
        filled = int(frame_number * progressbarsize / total_frames)
        bar = '■' * filled + '.' * (progressbarsize - filled)
        print(f"\r\033[K[\033[31m{bar}\033[39m] {frame_number / total_frames * 100:.02f}%", end="")

    # ========================================================
    # 従来 cv2 モード (out_type=None)
    # ========================================================
    if out_type is None:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height), 1)
        frame_number = 0
        for npy_path in sorted_files:
            gc.collect()
            print(f"\nmemory={psutil.virtual_memory().percent}%")
            data = np.load(npy_path)
            if per_frame:
                video.write(data)
                frame_number += 1
                _print_progress(frame_number, total_frames)
            else:
                for i in range(data.shape[0]):
                    video.write(data[i])
                    frame_number += 1
                    _print_progress(frame_number, total_frames)
            del data
            gc.collect()
        video.release()
        # tmp クリーンアップ
        for tmp in _cleanup_tmp:
            if os.path.exists(tmp): os.remove(tmp)
        print(f"\nRender Complete: {out_path}")
        return out_path

    # ========================================================
    # FFmpeg パイプモード
    # ========================================================
    use_16bit = out_type in (OT_H265, OT_PRORES_422, OT_PRORES_4444, OT_H265_SDR, OT_PRORES_422_SDR)

    cmd = ["ffmpeg", "-y"]

    if out_type == OT_H264:
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}", "-r", str(out_fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "23", "-tag:v", "avc1",
            out_path
        ]
    elif out_type == OT_H265:
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s:v", f"{width}x{height}", "-r", str(out_fps), "-i", "-",
            "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
            "-x265-params",
            "hdr-opt=1:repeat-headers=1:"
            "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
            "master-display=G(13250,34500)B(7500,3000)"
            "R(34000,16000)WP(15635,16450)L(10000000,50):"
            "max-cll=1000,400",
            out_path
        ]
    elif out_type == OT_PRORES_422:
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s:v", f"{width}x{height}", "-r", str(out_fps), "-i", "-",
            "-c:v", "prores_ks", "-pix_fmt", "yuv422p10le", "-profile:v", "3",
            "-color_primaries", "bt2020", "-colorspace", "bt2020nc", "-color_trc", "smpte2084",
            out_path
        ]
    elif out_type == OT_PRORES_4444:
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s:v", f"{width}x{height}", "-r", str(out_fps), "-i", "-",
            "-c:v", "prores_ks", "-pix_fmt", "yuv444p10le", "-profile:v", "4",
            "-color_primaries", "bt2020", "-colorspace", "bt2020nc", "-color_trc", "smpte2084",
            out_path
        ]
    elif out_type == OT_H265_SDR:
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s:v", f"{width}x{height}", "-r", str(out_fps), "-i", "-",
            "-c:v", "libx265", "-pix_fmt", "yuv422p10le", "-tag:v", "hvc1",
            "-x265-params", "repeat-headers=1:hdr-opt=0:colorprim=bt709:transfer=bt709:colormatrix=bt709",
            out_path
        ]
    elif out_type == OT_PRORES_422_SDR:
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s:v", f"{width}x{height}", "-r", str(out_fps), "-i", "-",
            "-c:v", "prores_ks", "-pix_fmt", "yuv422p10le", "-profile:v", "3",
            "-color_primaries", "bt709", "-colorspace", "bt709", "-color_trc", "bt709",
            out_path
        ]
    else:
        raise ValueError(f"Unsupported out_type: {out_type}")

    print("FFmpeg CMD:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # --- フレーム書き込み ---
    frame_number = 0
    for npy_path in sorted_files:
        gc.collect()
        print(f"\nmemory={psutil.virtual_memory().percent}%  loading: {os.path.basename(npy_path)}")
        data = np.load(npy_path)
        print(f"memory={psutil.virtual_memory().percent}%")
        if per_frame:
            # 1フレーム1npy: shape=(H,W,3)
            proc.stdin.write(_prepare_frame(data, use_16bit))
            frame_number += 1
            _print_progress(frame_number, total_frames)
        else:
            # sep-npy or 単一npy: shape=(N,H,W,3)
            for i in range(data.shape[0]):
                proc.stdin.write(_prepare_frame(data[i], use_16bit))
                frame_number += 1
                _print_progress(frame_number, total_frames)
        del data
        gc.collect()

    try:
        proc.stdin.flush()
        proc.stdin.close()
    except Exception as e:
        print(f"\n⚠️ Failed to close FFmpeg stdin: {e}")
    proc.wait()
    # tmp クリーンアップ
    for tmp in _cleanup_tmp:
        if os.path.exists(tmp): os.remove(tmp)
    print(f"\nRender Complete: {out_path}")
    return out_path


def rearrange_wide_video(output_path, input_path=None, npys_path=None,
                         sep_start=0, sep_end=None,
                         roll_offset=0, out_fps=30, out_type=None):
    """
    横長動画（W×H）を中央で分割し上下に再配置して (W//2)×(2H) の動画を生成する。
    入力は動画ファイル(input_path)または sep-*.npy フォルダ(npys_path)のどちらかを指定。
    roll_offset で横方向の循環シフト（ロール）を指定可能。

    配置イメージ（roll_offset=1549, 入力 11012×1080 の場合）:
        ┌──────────────────────────────────┐
        │  上段: src x=1549 .. 7054        │  5506 × 1080
        ├──────────────┬───────────────────┤
        │ src x=7055.. │ src x=0..1548     │  5506 × 1080
        │  11011       │  (wrap)           │
        └──────────────┴───────────────────┘
        出力: 5506 × 2160

    Parameters:
        output_path : 出力パス（拡張子なし、out_type に応じて自動付与）
        input_path  : 入力動画パス（動画ファイルから読む場合）
        npys_path   : sep-*.npy があるフォルダ（npy から直接レンダリングする場合）
        sep_start   : npy モード時の開始 sep 番号（デフォルト 0）
        sep_end     : npy モード時の終了 sep 番号（None = 全件）
        roll_offset : 横方向の循環シフト量（ピクセル、デフォルト 0）
        out_fps     : 出力 fps（デフォルト 30）
        out_type    : 出力コーデック（drawManeuver 定数、None = cv2 mp4v）
    """
    if input_path is None and npys_path is None:
        raise ValueError("input_path か npys_path のどちらかを指定してください")

    # --- out_type 定数 ---
    OT_H264         = 1
    OT_H265         = 2
    OT_PRORES_422   = 3
    OT_PRORES_4444  = 4
    OT_H265_SDR     = 5
    OT_PRORES_422_SDR = 6

    # --- 拡張子 ---
    if out_type in (OT_PRORES_422, OT_PRORES_4444, OT_PRORES_422_SDR):
        ext = ".mov"
    elif out_type in (OT_H264, OT_H265, OT_H265_SDR):
        ext = ".mp4"
    else:
        ext = ".mp4"
    out_path = output_path + ext

    # ================================================================
    # 入力ソースの準備
    # ================================================================
    if npys_path is not None:
        # --- npy モード: sep-*.npy からフレームリストを構築 ---
        file_list = [f for f in os.listdir(npys_path) if not f.startswith('.') and f.endswith('.npy')]
        if not file_list:
            raise ValueError(f"No .npy files found in {npys_path}")
        base_string = extract_base_string(file_list[0])
        file_count = len(file_list)
        if sep_end is None:
            file_list = [base_string + "sep-" + str(i) + ".npy" for i in range(sep_start, sep_start + file_count)]
        else:
            file_list = [base_string + "sep-" + str(i) + ".npy" for i in range(sep_start, sep_end)]
        sorted_files = [os.path.abspath(os.path.join(npys_path, f)) for f in file_list]

        first = np.load(sorted_files[0])
        src_h, src_w = first.shape[1], first.shape[2]
        npy_dtype = first.dtype
        frames_per_file = first.shape[0]
        total_frames = frames_per_file * len(sorted_files)
        del first
        gc.collect()
        print(f"rearrange_wide_video [npy mode]")
        print(f"  Input : {src_w}x{src_h}, {total_frames} frames, dtype={npy_dtype}, files={len(sorted_files)}")
    else:
        # --- 動画モード ---
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {input_path}")
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        npy_dtype = np.dtype('uint8')
        print(f"rearrange_wide_video [video mode]")
        print(f"  Input : {src_w}x{src_h} @ {src_fps:.2f}fps, {total_frames} frames")

    half_w = src_w // 2
    out_w, out_h = half_w, src_h * 2
    print(f"  Output: {out_w}x{out_h} @ {out_fps}fps, roll_offset={roll_offset}")

    # --- プログレス表示 ---
    def _print_progress(current, total):
        filled = int(current * 50 / total)
        bar = '■' * filled + '.' * (50 - filled)
        print(f"\r\033[K[\033[31m{bar}\033[39m] {current / total * 100:.02f}%", end="")

    # --- 1フレーム変換: ロール → 分割 → 上下積み ---
    def _rearrange_frame(frame):
        if roll_offset != 0:
            frame = np.concatenate([frame[:, roll_offset:, :], frame[:, :roll_offset, :]], axis=1)
        return np.vstack([frame[:, :half_w, :], frame[:, half_w:half_w * 2, :]])

    # --- FFmpeg へ書き込む1フレーム分のバイト列を返す ---
    use_16bit = out_type in (OT_H265, OT_PRORES_422, OT_PRORES_4444, OT_H265_SDR, OT_PRORES_422_SDR)

    def _frame_to_bytes(frame):
        # npy(uint8=BGR, uint16=RGB) / video(cv2=BGR uint8) → RGB
        if frame.dtype == np.uint8:
            rgb = frame[:, :, ::-1]  # BGR→RGB
        else:
            rgb = frame  # PyAV由来 = すでにRGB
        if use_16bit:
            if rgb.dtype == np.uint8:
                return (rgb.astype(np.uint16) * 257).tobytes()
            else:
                return rgb.astype(np.uint16).tobytes()
        else:
            return rgb.astype(np.uint8).tobytes()

    # ================================================================
    # ライター準備
    # ================================================================
    if out_type is None:
        # --- cv2 モード ---
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h), True)

        def _write(frame):
            writer.write(_rearrange_frame(frame))  # cv2はBGRのまま書く

        def _close():
            writer.release()
    else:
        # --- FFmpeg パイプモード ---
        cmd = ["ffmpeg", "-y"]
        pix_fmt_in = "rgb48le" if use_16bit else "rgb24"
        if out_type == OT_H264:
            cmd += [
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s:v", f"{out_w}x{out_h}", "-r", str(out_fps), "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", "-crf", "23", "-tag:v", "avc1", out_path
            ]
        elif out_type == OT_H265:
            cmd += [
                "-f", "rawvideo", "-pix_fmt", "rgb48le",
                "-s:v", f"{out_w}x{out_h}", "-r", str(out_fps), "-i", "-",
                "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
                "-x265-params",
                "hdr-opt=1:repeat-headers=1:"
                "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                "master-display=G(13250,34500)B(7500,3000)"
                "R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,400",
                out_path
            ]
        elif out_type == OT_PRORES_422:
            cmd += [
                "-f", "rawvideo", "-pix_fmt", "rgb48le",
                "-s:v", f"{out_w}x{out_h}", "-r", str(out_fps), "-i", "-",
                "-c:v", "prores_ks", "-pix_fmt", "yuv422p10le", "-profile:v", "3",
                "-color_primaries", "bt2020", "-colorspace", "bt2020nc", "-color_trc", "smpte2084",
                out_path
            ]
        elif out_type == OT_PRORES_4444:
            cmd += [
                "-f", "rawvideo", "-pix_fmt", "rgb48le",
                "-s:v", f"{out_w}x{out_h}", "-r", str(out_fps), "-i", "-",
                "-c:v", "prores_ks", "-pix_fmt", "yuv444p10le", "-profile:v", "4",
                "-color_primaries", "bt2020", "-colorspace", "bt2020nc", "-color_trc", "smpte2084",
                out_path
            ]
        elif out_type == OT_H265_SDR:
            cmd += [
                "-f", "rawvideo", "-pix_fmt", "rgb48le",
                "-s:v", f"{out_w}x{out_h}", "-r", str(out_fps), "-i", "-",
                "-c:v", "libx265", "-pix_fmt", "yuv422p10le", "-tag:v", "hvc1",
                "-x265-params", "repeat-headers=1:hdr-opt=0:colorprim=bt709:transfer=bt709:colormatrix=bt709",
                out_path
            ]
        elif out_type == OT_PRORES_422_SDR:
            cmd += [
                "-f", "rawvideo", "-pix_fmt", "rgb48le",
                "-s:v", f"{out_w}x{out_h}", "-r", str(out_fps), "-i", "-",
                "-c:v", "prores_ks", "-pix_fmt", "yuv422p10le", "-profile:v", "3",
                "-color_primaries", "bt709", "-colorspace", "bt709", "-color_trc", "bt709",
                out_path
            ]
        else:
            if npys_path is None:
                cap.release()
            raise ValueError(f"Unsupported out_type: {out_type}")

        print("FFmpeg CMD:", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        def _write(frame):
            proc.stdin.write(_frame_to_bytes(_rearrange_frame(frame)))

        def _close():
            try:
                proc.stdin.flush()
                proc.stdin.close()
            except Exception as e:
                print(f"\n⚠️ Failed to close FFmpeg stdin: {e}")
            proc.wait()

    # ================================================================
    # フレーム書き込みループ
    # ================================================================
    frame_number = 0

    if npys_path is not None:
        # --- npy モード ---
        for npy_file in sorted_files:
            gc.collect()
            print(f"\nmemory={psutil.virtual_memory().percent}%  loading: {os.path.basename(npy_file)}")
            images = np.load(npy_file)
            for i in range(images.shape[0]):
                _write(images[i])
                frame_number += 1
                _print_progress(frame_number, total_frames)
            del images
            gc.collect()
    else:
        # --- 動画モード ---
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            _write(frame)
            frame_number += 1
            _print_progress(frame_number, total_frames)
        cap.release()

    _close()
    print(f"\nRender Complete: {out_path}")
    return out_path


 # videoパスから、指定した数の静止画を出力する
def rendered_mov_to_seq(video_path, divide_num=None, img_format='jpg',
                        frame_array=None, color_mode='source'):
    """動画からフレームを静止画として切り出す (ffmpeg ベース / HDR 色空間対応)。

    Parameters
    ----------
    video_path : str
        入力動画パス。
    divide_num : int or None
        均等分割数。frame_array より優先。None なら frame_array を使う。
    img_format : str
        出力フォーマット。
        'ultrahdr' — Ultra HDR JPEG (Gain Map HDR, iOS18+/Android14+/Instagram対応)
        'png'      — 16bit PNG (HDR色域そのまま保持、最も安全)
        'jpg'      — 8bit JPEG (SDR/Ultra HDR)
        'avif'     — 10bit AVIF (HDR対応、ブラウザ互換◎)
        'npy'      — numpy配列 (cv2フォールバック)
    frame_array : array-like or None
        書き出すフレーム番号の配列。None + divide_num=None で全フレーム書き出し。
    color_mode : str
        'source' — 入力の色空間メタデータをそのまま保持 (デフォルト)
        'sdr'    — BT.709 SDR にトーンマップ (互換性最大)
        'hlg'    — PQ→HLG 変換 (要 zscale/libzimg、無ければ source にフォールバック)

    Returns
    -------
    str : 出力フォルダパス
    """
    import subprocess, json, shutil

    FFMPEG  = shutil.which("ffmpeg")  or "/opt/homebrew/bin/ffmpeg"
    FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

    # ── 1. ffprobe で総フレーム数 & 入力色空間を取得 ───────────────────
    probe_cmd = [
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path,
    ]
    probe = json.loads(subprocess.check_output(probe_cmd, text=True))
    vstream = probe["streams"][0]
    total_frames = int(vstream.get("nb_frames", 0))
    if total_frames == 0:
        fps = eval(vstream.get("r_frame_rate", "30/1"))
        dur = float(vstream.get("duration", 0))
        total_frames = int(dur * fps)
    src_trc   = vstream.get("color_transfer", "unknown")
    src_pri   = vstream.get("color_primaries", "unknown")
    src_space = vstream.get("color_space", "unknown")
    print(f"[mov_to_seq] frames={total_frames}  "
          f"src: pri={src_pri}  trc={src_trc}  space={src_space}")

    digits = max(4, len(str(total_frames)))

    # ── フィルタ有無チェック ────────────────────────────────────────────
    chk = subprocess.run([FFMPEG, "-hide_banner", "-filters"],
                         capture_output=True, text=True)
    has_zscale  = "zscale"   in chk.stdout
    has_scale_vt = "scale_vt" in chk.stdout   # macOS VideoToolbox HW色変換

    # ── 2. frame_array 決定 ───────────────────────────────────────────
    if divide_num is not None:
        step = max(1, total_frames // (divide_num - 1) - 1)
        frame_array = np.arange(0, total_frames, step)
        frame_array[-1] = total_frames - 1
        print(f"[mov_to_seq] divide_num={divide_num} → {len(frame_array)} frames")

    # ── 3. 出力フォルダ ──────────────────────────────────────────────
    directory, filename = os.path.split(video_path)
    filename_without_ext = os.path.splitext(filename)[0]
    tag = f"{img_format}_{color_mode}"
    output_folder = os.path.join(directory, f"{filename_without_ext}_frames_{tag}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"[mov_to_seq] output → {output_folder}")

    # ── Ultra HDR 専用パス (ffmpeg + ultrahdr_app) ────────────────────
    if img_format == "ultrahdr":
        width  = int(vstream.get("width", 0))
        height = int(vstream.get("height", 0))
        if frame_array is None:
            frame_array = np.arange(total_frames)
        return _rendered_mov_to_ultrahdr_seq(
            video_path, frame_array, output_folder,
            total_frames, digits, width, height, src_trc, FFMPEG)

    # ── 4. vf フィルタ構築 ───────────────────────────────────────────
    #  色変換の優先順位:
    #    1) zscale (libzimg)  — SW、最も正確
    #    2) scale_vt          — macOS VideoToolbox HW (要 -init_hw_device)
    #    3) フォールバック     — source モード(変換なし、メタのみ)
    use_hw = False          # scale_vt 使用フラグ (-init_hw_device 付与用)
    vf_parts = []

    # select フィルタ (特定フレームのみ) — HW upload より前に置く
    if frame_array is not None:
        sel = "+".join(f"eq(n\\,{int(n)})" for n in frame_array)
        vf_parts.append(f"select='{sel}'")

    if color_mode == "hlg":
        if has_zscale:
            vf_parts.append(
                "zscale=t=arib-std-b67:p=bt2020:m=bt2020nc"
                ":tin=smpte2084:pin=bt2020:min=bt2020nc"
                ":r=tv:rin=tv"
            )
        elif has_scale_vt:
            use_hw = True
            print("[mov_to_seq] zscale無し → scale_vt (VideoToolbox) で PQ→HLG 変換")
            vf_parts += [
                "format=nv12", "hwupload",
                "scale_vt=color_matrix=bt2020nc"
                ":color_primaries=bt2020"
                ":color_transfer=arib-std-b67",
                "hwdownload", "format=nv12",
            ]
        else:
            print("[mov_to_seq] 色変換フィルタ無し → source フォールバック")
            color_mode = "source"

    elif color_mode == "sdr":
        if has_zscale:
            vf_parts += [
                "zscale=t=linear:npl=100:tin=smpte2084:pin=bt2020:min=bt2020nc",
                "format=gbrpf32le",
                "tonemap=tonemap=hable:desat=0",
                "zscale=t=bt709:p=bt709:m=bt709",
                "format=yuvj420p",
            ]
        elif has_scale_vt:
            use_hw = True
            print("[mov_to_seq] zscale無し → scale_vt (VideoToolbox) で SDR 変換")
            vf_parts += [
                "format=nv12", "hwupload",
                "scale_vt=color_matrix=bt709"
                ":color_primaries=bt709"
                ":color_transfer=bt709",
                "hwdownload", "format=nv12",
            ]
        else:
            print("[mov_to_seq] 色変換フィルタ無し → source フォールバック")
            color_mode = "source"

    vf_str = ",".join(vf_parts) if vf_parts else None

    # ── 5. 出力コーデック / メタデータ設定 ────────────────────────────
    out_pattern = os.path.join(output_folder, f"frame_%0{digits}d.{img_format}")

    if img_format == "avif":
        # libsvtav1 で AVIF 書き出し (10bit 対応)
        enc_args = ["-c:v", "libsvtav1", "-crf", "23",
                    "-pix_fmt", "yuv420p10le", "-svtav1-params", "tune=0"]
    elif img_format == "png":
        enc_args = ["-pix_fmt", "rgb48be"]   # 16bit PNG
    elif img_format == "jpg":
        # yuvj420p = full-range YUV (MJPEG必須)。SDRモードではvfで変換済み。
        enc_args = ["-q:v", "2", "-pix_fmt", "yuvj420p"]
    elif img_format == "npy":
        return _rendered_mov_to_seq_cv2(video_path, divide_num, frame_array,
                                        output_folder, total_frames, digits)
    else:
        enc_args = []

    # 色メタデータ付与
    meta_args = []
    if color_mode == "hlg":
        meta_args += ["-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
                      "-colorspace", "bt2020nc", "-color_range", "tv"]
    elif color_mode == "sdr":
        meta_args += ["-color_primaries", "bt709", "-color_trc", "bt709",
                      "-colorspace", "bt709", "-color_range", "tv"]
    elif color_mode == "source":
        if src_pri != "unknown":
            meta_args += ["-color_primaries", src_pri]
        if src_trc != "unknown":
            meta_args += ["-color_trc", src_trc]
        if src_space != "unknown":
            meta_args += ["-colorspace", src_space]

    # ── 6. ffmpeg 実行 ───────────────────────────────────────────────
    cmd = [FFMPEG, "-y", "-hide_banner"]
    if use_hw:
        cmd += ["-init_hw_device", "videotoolbox=vt"]
    cmd += ["-i", video_path]
    if vf_str:
        cmd += ["-vf", vf_str]
    if use_hw:
        cmd += ["-filter_hw_device", "vt"]
    cmd += ["-fps_mode", "vfr"]
    cmd += enc_args + meta_args
    cmd += [out_pattern]

    print("[mov_to_seq] cmd:", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[mov_to_seq] ffmpeg error (returncode={ret.returncode})")
        return None

    import glob as _glob
    out_files = _glob.glob(os.path.join(output_folder, f"frame_*.{img_format}"))
    print(f"[mov_to_seq] {len(out_files)} frames saved → {output_folder}")
    return output_folder


def _rendered_mov_to_seq_cv2(video_path, divide_num, frame_array,
                              output_folder, total_frames, digits):
    """npy フォーマット用フォールバック (cv2 ベース)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None
    if divide_num is not None and frame_array is None:
        frame_array = np.arange(0, total_frames,
                                max(1, int(total_frames // (divide_num - 1) - 1)))
        frame_array[-1] = total_frames - 1
    frame_count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_array is None or frame_count in frame_array:
            fname = f'frames_{frame_count}.npy'
            np.save(os.path.join(output_folder, fname), frame)
            saved += 1
        frame_count += 1
        print(f"Progress: {frame_count/total_frames*100:.1f}%", end='\r')
    cap.release()
    print(f"\n{saved} npy frames saved.")
    return output_folder

def _rendered_mov_to_ultrahdr_seq(video_path, frame_array, output_folder,
                                   total_frames, digits, width, height,
                                   src_trc, FFMPEG):
    """Ultra HDR JPEG (Gain Map HDR / Adobe hdrgm 互換) フレーム書き出し。

    処理フロー (フレームごと):
      1. ffmpeg で P010 (10bit YUV) raw 抽出
      2. ultrahdr_app で Ultra HDR JPEG 生成 (ISO 21496 形式)
      3. _convert_to_adobe_ultrahdr() で Adobe hdrgm 互換形式に変換
         → macOS Finder / Preview / iOS Photos で正常表示
    """
    import subprocess, shutil, tempfile

    ULTRAHDR = shutil.which("ultrahdr_app") or "/opt/homebrew/bin/ultrahdr_app"
    if not os.path.isfile(ULTRAHDR):
        print("[mov_to_seq] ultrahdr_app が見つかりません。\n"
              "  brew install libultrahdr でインストールしてください。")
        return None

    trc_map = {"smpte2084": "2", "arib-std-b67": "1"}
    t_flag = trc_map.get(src_trc, "2")
    c_flag = "2"  # bt2100/bt2020

    saved = 0
    tmpdir = tempfile.mkdtemp(prefix="ultrahdr_")

    try:
        for idx, fnum in enumerate(frame_array):
            fnum = int(fnum)
            raw_path = os.path.join(tmpdir, f"frame_{fnum}.p010")
            uhdr_tmp = os.path.join(tmpdir, f"frame_{fnum}_uhdr.jpg")
            out_name = f"frame_{str(fnum).zfill(digits)}.jpg"
            out_path = os.path.join(output_folder, out_name)

            # ── 1. ffmpeg: P010 raw 抽出 ─────────────────────────────
            ff_cmd = [
                FFMPEG, "-y", "-hide_banner", "-loglevel", "warning",
                "-i", video_path,
                "-vf", f"select='eq(n\\,{fnum})'",
                "-fps_mode", "vfr",
                "-f", "rawvideo", "-pix_fmt", "p010le",
                raw_path,
            ]
            ret = subprocess.run(ff_cmd)
            if ret.returncode != 0 or not os.path.isfile(raw_path):
                print(f"  [!] frame {fnum}: ffmpeg P010 抽出失敗")
                continue

            # ── 2. ultrahdr_app: P010 → ISO 21496 Ultra HDR ──────────
            uhdr_cmd = [
                ULTRAHDR, "-m", "0",
                "-p", raw_path, "-a", "0",
                "-w", str(width), "-h", str(height),
                "-C", c_flag, "-t", t_flag,
                "-q", "100", "-Q", "95", "-s", "4",
                "-z", uhdr_tmp,
            ]
            ret = subprocess.run(uhdr_cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"  [!] frame {fnum}: ultrahdr_app 失敗: {ret.stderr.strip()}")
                continue

            # ── 3. Adobe hdrgm 互換形式に変換 ────────────────────────
            _convert_to_adobe_ultrahdr(uhdr_tmp, out_path)

            # tmp 削除
            for p in (raw_path, uhdr_tmp):
                if os.path.isfile(p):
                    os.remove(p)
            saved += 1
            pct = (idx + 1) / len(frame_array) * 100
            print(f"  Ultra HDR: {saved}/{len(frame_array)}  ({pct:.0f}%)", end='\r')
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n[mov_to_seq] {saved} Ultra HDR frames saved → {output_folder}")
    return output_folder


def _convert_to_adobe_ultrahdr(iso_uhdr_path, out_path):
    """ISO 21496 形式の Ultra HDR JPEG を Adobe hdrgm 互換形式に変換。

    変換内容:
      - Primary JPEG から ISO 21496 APP2 マーカーを除去
      - GainMap JPEG に hdrgm XMP メタデータを注入
      - MPF APP2 を再構築 (MP Image Type = 0x050000 = Gain Map Image)
      - Finder / Preview / iOS Photos 互換のマーカー順序に整列

    Adobe Camera RAW の出力構造に準拠:
      Primary:  [SOI][APP1 Exif][...][APP2 MPF][image data][EOI]
      GainMap:  [SOI][APP1 XMP(hdrgm)][image data][EOI]
    """
    import struct

    with open(iso_uhdr_path, 'rb') as f:
        data = f.read()

    # ── Primary / GainMap 分離 ────────────────────────────────────────
    gm_offset = data.rfind(b'\xff\xd8', 1)
    if gm_offset < 0:
        # ゲインマップが見つからない → そのままコピー
        import shutil
        shutil.copy2(iso_uhdr_path, out_path)
        return
    primary_bytes = data[:gm_offset]
    gainmap_bytes = data[gm_offset:]

    # ── Primary: ISO 21496 APP2 マーカーを除去 ─────────────────────────
    clean_primary = _strip_iso21496_markers(primary_bytes)

    # ── GainMap: hdrgm XMP を注入 ─────────────────────────────────────
    # ISO 21496 バイナリからゲインマップパラメータを推定
    # (libultrahdr のデフォルト値を使用)
    hdrgm_xmp = _build_hdrgm_xmp()
    xmp_app1 = _build_xmp_app1(hdrgm_xmp)
    # GainMap JPEG の SOI 直後に XMP APP1 を挿入
    clean_gainmap = _strip_iso21496_markers(gainmap_bytes)
    gainmap_with_xmp = clean_gainmap[:2] + xmp_app1 + clean_gainmap[2:]

    # ── MPF APP2 構築 ─────────────────────────────────────────────────
    # MPF を既存マーカー群の後（SOS の直前）に挿入する
    # → 先頭に ICC/JFIF が来て Finder 互換になる
    # SOS マーカー (FFDA) の位置を探す
    sos_pos = clean_primary.find(b'\xff\xda')
    if sos_pos < 0:
        sos_pos = 2  # fallback: SOI直後

    dummy_mpf = _build_mpf_app2(primary_size=0, gainmap_size=0)
    mpf_len = len(dummy_mpf)
    final_primary_size = len(clean_primary) + mpf_len
    mpf_app2 = _build_mpf_app2(
        primary_size=final_primary_size,
        gainmap_size=len(gainmap_with_xmp),
        mpf_offset_in_file=sos_pos,  # MPF の挿入位置
    )
    primary_with_mpf = (clean_primary[:sos_pos] + mpf_app2 +
                        clean_primary[sos_pos:])

    # ── 結合して書き出し ──────────────────────────────────────────────
    with open(out_path, 'wb') as f:
        f.write(primary_with_mpf)
        f.write(gainmap_with_xmp)


def _strip_iso21496_markers(jpeg_bytes):
    """JPEG バイナリから ISO 21496 URN APP2 マーカーを除去。"""
    result = bytearray()
    i = 0
    while i < len(jpeg_bytes):
        if i + 4 < len(jpeg_bytes) and jpeg_bytes[i:i+2] == b'\xff\xe2':
            seg_len = (jpeg_bytes[i+2] << 8) | jpeg_bytes[i+3]
            seg_data = jpeg_bytes[i+4:i+2+seg_len]
            if b'urn:iso:std:iso:ts:21496' in seg_data:
                # ISO 21496 マーカー → スキップ
                i += 2 + seg_len
                continue
            if seg_data[:4] == b'MPF\x00':
                # 既存MPF → スキップ (後で再構築)
                i += 2 + seg_len
                continue
        result.append(jpeg_bytes[i])
        i += 1
    return bytes(result)


def _build_hdrgm_xmp(gain_min=0.0, gain_max=3.5, gamma=1.0,
                      hdr_capacity_max=3.5, offset_sdr=0.015625,
                      offset_hdr=0.015625):
    """Adobe hdrgm 互換の XMP メタデータ文字列を生成。"""
    xmp = f'''<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="imgtrans2026">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
   hdrgm:Version="1.0"
   hdrgm:BaseRenditionIsHDR="False"
   hdrgm:HDRCapacityMin="0"
   hdrgm:HDRCapacityMax="{hdr_capacity_max}"
   hdrgm:OffsetSDR="{offset_sdr}"
   hdrgm:OffsetHDR="{offset_hdr}"
   hdrgm:GainMapMin="{gain_min}"
   hdrgm:GainMapMax="{gain_max}"
   hdrgm:Gamma="{gamma}"/>
 </rdf:RDF>
</x:xmpmeta>'''
    return xmp


def _build_xmp_app1(xmp_str):
    """XMP 文字列を JPEG APP1 セグメントとしてパック。"""
    import struct
    ns = b'http://ns.adobe.com/xap/1.0/\x00'
    xmp_bytes = xmp_str.encode('utf-8')
    payload = ns + xmp_bytes
    seg_len = len(payload) + 2  # +2 for length field itself
    return b'\xff\xe1' + struct.pack('>H', seg_len) + payload


def _build_mpf_app2(primary_size, gainmap_size, mpf_offset_in_file=2):
    """Adobe 互換の MPF (Multi-Picture Format) APP2 セグメントを構築。

    MP Entry format (16 bytes per entry):
      - Individual Image Attribute (4 bytes)
      - Individual Image Size (4 bytes)
      - Individual Image Data Offset (4 bytes) ← MPF APP2 先頭からの相対
      - Dependent Image 1 & 2 Entry Number (2+2 bytes)

    Image Attribute bits:
      [31:30] Dependent Image flag
      [29:27] Representative Image flag
      [26:24] Image Data Format (0=JPEG)
      [23:0]  Type code: 0x000000=Undefined, 0x030000=Multi-Frame Panorama,
              0x050000=Baseline Multi-Profile (used for Gain Map by Adobe)
    mpf_offset_in_file: MPF APP2 "MPF\\x00" のファイル内絶対位置。
        GainMap のオフセットは "MPF\\x00" からの相対で計算。
    """
    import struct

    # TIFF Header (Big Endian)
    tiff_hdr = b'MM'
    tiff_hdr += struct.pack('>H', 0x002A)
    tiff_hdr += struct.pack('>I', 8)           # first IFD offset

    # IFD: 3 tags
    ifd = struct.pack('>H', 3)

    # Tag 1: MPFVersion (B000)
    ifd += struct.pack('>HH', 0xB000, 7)
    ifd += struct.pack('>I', 4)
    ifd += b'0100'

    # Tag 2: NumberOfImages (B001)
    ifd += struct.pack('>HH', 0xB001, 4)
    ifd += struct.pack('>I', 1)
    ifd += struct.pack('>I', 2)

    # Tag 3: MPEntry (B002), 32 bytes
    mp_entry_offset = 8 + 2 + 3*12 + 4
    ifd += struct.pack('>HH', 0xB002, 7)
    ifd += struct.pack('>I', 32)
    ifd += struct.pack('>I', mp_entry_offset)

    ifd += struct.pack('>I', 0)                # next IFD: none

    # MP Entry 1: Primary (representative)
    entry1  = struct.pack('>I', 0x20030000)
    entry1 += struct.pack('>I', primary_size)
    entry1 += struct.pack('>I', 0)
    entry1 += struct.pack('>HH', 0, 0)

    # MP Entry 2: Gain Map
    # CIPA DC-007: offset = from start of "MPF\x00" in file
    # "MPF\x00" is at file position (mpf_offset_in_file + 4)
    # ゲインマップ SOI は primary_size の位置
    mpf_header_pos = mpf_offset_in_file + 4    # "MPF\x00" の後 = TIFF header 開始
    gm_data_offset = primary_size - mpf_header_pos

    entry2  = struct.pack('>I', 0x00050000)    # Gain Map type
    entry2 += struct.pack('>I', gainmap_size)
    entry2 += struct.pack('>I', max(0, gm_data_offset))
    entry2 += struct.pack('>HH', 0, 0)

    mpf_payload = b'MPF\x00' + tiff_hdr + ifd + entry1 + entry2
    seg_len = len(mpf_payload) + 2

    return b'\xff\xe2' + struct.pack('>H', seg_len) + mpf_payload



def export_segments(video_path, out_dir, segment_sec=10, segment_count=None, out_fps=60, with_frame_num=True, recfps=480, export_only="both"):
    """ソース映像をA群(順再生)・B群(hflip+逆再生)のセグメントに分割書き出し。
    フレーム番号とタイムコードをオーバーレイ描画する。
    Args:
        video_path: ソース映像パス
        out_dir: 出力ディレクトリ（A/, B/ サブディレクトリを作成）
        segment_sec: 1セグメントのソース秒数（デフォルト10）
        segment_count: セグメント数（Noneなら映像全体）
        out_fps: 出力fps（デフォルト60）
        with_frame_num: フレーム番号を描画するか
    """
    import subprocess, av
    if export_only in ("both", "A"):
        os.makedirs(os.path.join(out_dir, "A"), exist_ok=True)
    if export_only in ("both", "B"):
        os.makedirs(os.path.join(out_dir, "B"), exist_ok=True)

    container = av.open(video_path)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate)
    total_frames = stream.frames or int(stream.duration * stream.time_base * src_fps)
    step = max(1, int(round(recfps / out_fps)))  # 480/60=8: 8フレームに1枚で等倍再生
    frames_per_seg = int(segment_sec * recfps)   # 10real_sec * 480 = 4800 file frames

    if segment_count is None:
        segment_count = int(total_frames / frames_per_seg)

    w = stream.codec_context.width
    h = stream.codec_context.height

    for seg_idx in range(1, segment_count + 1):
        start_frame = (seg_idx - 1) * frames_per_seg
        end_frame = start_frame + frames_per_seg

        print(f"Segment {seg_idx}/{segment_count}: frames {start_frame}-{end_frame} ({start_frame/src_fps:.1f}s-{end_frame/src_fps:.1f}s)")

        seg_frames = []
        seg_frame_nums = []
        container.seek(int(start_frame / src_fps * av.time_base), any_frame=False)
        frame_count = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                fn = frame.pts * stream.time_base * src_fps if frame.pts is not None else frame_count
                fn = int(round(fn))
                if fn < start_frame:
                    continue
                if fn >= end_frame:
                    break
                if (fn - start_frame) % step == 0:
                    img = frame.to_ndarray(format="rgb24")
                    seg_frames.append(img)
                    seg_frame_nums.append(fn)
                frame_count += 1
            else:
                continue
            break

        if not seg_frames:
            print(f"  Warning: no frames for segment {seg_idx}")
            continue

        # A群: 順再生書き出し
        if export_only in ("both", "A"):
            a_path = os.path.join(out_dir, "A", f"A{seg_idx}.mov")
            a_cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s:v", f"{w}x{h}", "-r", str(out_fps), "-i", "-",
                     "-c:v", "prores_ks", "-profile:v", "3",
                     "-pix_fmt", "yuv422p10le", a_path]
            a_proc = subprocess.Popen(a_cmd, stdin=subprocess.PIPE)
            for j, frm in enumerate(seg_frames):
                if with_frame_num:
                    fn_a = seg_frame_nums[j]
                    bgr = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
                    rc_sec = int(fn_a // recfps)
                    rc_frac = int(fn_a - rc_sec * recfps)
                    text = f"{rc_sec}s{rc_frac}f"
                    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    tx = (w - tw) // 2
                    cv2.putText(bgr, text, (tx, h - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    frm = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                a_proc.stdin.write(frm.tobytes())
            a_proc.stdin.close()
            a_proc.wait()
            print(f"  A{seg_idx}: {a_path} ({len(seg_frames)} frames)")

        # B群: hflip + reverse書き出し（テキストはhflip前の素フレームに描画→hflip）
        if export_only in ("both", "B"):
            b_path = os.path.join(out_dir, "B", f"B{seg_idx}.mov")
            b_cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s:v", f"{w}x{h}", "-r", str(out_fps), "-i", "-",
                     "-c:v", "prores_ks", "-profile:v", "3",
                     "-pix_fmt", "yuv422p10le", b_path]
            b_proc = subprocess.Popen(b_cmd, stdin=subprocess.PIPE)
            for i, frm in enumerate(reversed(seg_frames)):
                fn_b = seg_frame_nums[len(seg_frames) - 1 - i]
                # 先にhflip
                flipped = np.fliplr(frm).copy()
                if with_frame_num:
                    # hflip後にテキスト描画（文字が反転しない）
                    bgr = cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR)
                    rc_sec = int(fn_b // recfps)
                    rc_frac = int(fn_b - rc_sec * recfps)
                    text = f"{rc_sec}s{rc_frac}f"
                    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    tx = (w - tw) // 2
                    cv2.putText(bgr, text, (tx, h - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    flipped = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                b_proc.stdin.write(flipped.tobytes())
            b_proc.stdin.close()
            b_proc.wait()
            print(f"  B{seg_idx}: {b_path} ({len(seg_frames)} frames)")

    container.close()
    print(f"export_segments done: {segment_count} segments to {out_dir}")
