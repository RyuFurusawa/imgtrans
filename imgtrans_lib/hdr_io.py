
import subprocess, shlex
import av
import numpy as np

# --- PQ (SMPTE ST 2084) 定数 ---
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0

def _pq_eotf(signal, L_p=10000.0):
    """PQ EOTF: signal [0,1] → linear light cd/m² (0..L_p)"""
    s = np.maximum(np.asarray(signal, dtype=np.float64), 0.0)
    Np = s ** (1.0 / _PQ_M2)
    num = np.maximum(Np - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * Np
    Y = num / np.maximum(den, 1e-12)
    return (Y ** (1.0 / _PQ_M1)) * L_p

def _pq_inverse_eotf(linear, L_p=10000.0):
    """PQ inverse-EOTF (= OETF): linear light cd/m² → signal [0,1]"""
    Y = np.maximum(np.asarray(linear, dtype=np.float64) / L_p, 0.0)
    Ym = Y ** _PQ_M1
    return ((_PQ_C1 + _PQ_C2 * Ym) / (1.0 + _PQ_C3 * Ym)) ** _PQ_M2

def _hlg_eotf(signal, peak_nits=1000.0, gamma=1.2):
    """HLG inverse-OETF + OOTF: signal [0,1] → display light cd/m²"""
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * np.log(4.0 * a)
    E = np.clip(np.asarray(signal, dtype=np.float64), 0.0, 1.0)
    L = np.where(E <= 0.5, (E ** 2) / 3.0, (np.exp((E - c) / a) + b) / 12.0)
    return peak_nits * np.maximum(L, 0.0) ** gamma

def _hlg_inverse_eotf(linear, peak_nits=1000.0, gamma=1.2):
    """HLG OOTF^-1 + OETF: display light cd/m² → signal [0,1]"""
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * np.log(4.0 * a)
    Y = np.maximum(np.asarray(linear, dtype=np.float64) / peak_nits, 0.0)
    L = Y ** (1.0 / gamma)
    E = np.where(L <= 1.0 / 12.0, np.sqrt(np.maximum(3.0 * L, 0.0)), a * np.log(np.maximum(12.0 * L - b, 1e-12)) + c)
    return np.clip(E, 0.0, 1.0)

def open_hdr_reader(path, pix_fmt_out="rgb48le"):
    """
    Decode a 10‑bit HDR video using PyAV/FFmpeg and yield frames as numpy arrays.
    Default output is interleaved RGB 16‑bit (uint16) = "rgb48le".
    """
    container = av.open(path)
    vstream = container.streams.video[0]
    for frame in container.decode(vstream):
        arr = frame.to_ndarray(format=pix_fmt_out)  # (H,W,3) uint16
        yield arr

def eotf_to_scene_linear(arr_uint16, transfer="pq"):
    """
    Convert display‑referred HDR (PQ/HLG) uint16 RGB to scene‑linear float32.
    Input expected in range [0..65535].
    """
    x = (arr_uint16.astype(np.float32) / 65535.0)
    t = transfer.lower()
    if t in ("pq", "st2084", "smpte2084"):
        return _pq_eotf(x, L_p=10000.0).astype(np.float32)
    elif t in ("hlg", "bt2100-hlg", "arib-std-b67"):
        return _hlg_eotf(x).astype(np.float32)
    else:
        return x

def oetf_from_scene_linear(arr_linear, transfer="pq"):
    """
    Convert scene‑linear float32 RGB (0..N nits) back to display‑referred (PQ/HLG) in 0..1.
    """
    t = transfer.lower()
    if t in ("pq", "st2084", "smpte2084"):
        return _pq_inverse_eotf(arr_linear, L_p=10000.0).astype(np.float32)
    elif t in ("hlg", "bt2100-hlg", "arib-std-b67"):
        return _hlg_inverse_eotf(arr_linear).astype(np.float32)
    else:
        return np.clip(arr_linear, 0.0, 1.0)

def open_hdr10_writer(width, height, fps, out_path,
                      master_display="G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)",
                      max_cll="1000,400"):
    """
    Open an FFmpeg process that accepts raw RGB48LE on stdin and writes HDR10 (HEVC 10‑bit) to out_path.
    Pass frames with shape (H,W,3) dtype=uint16 in display‑referred PQ (not linear).
    """
    cmd = f"""
ffmpeg -y -f rawvideo -pix_fmt rgb48le -s {width}x{height} -r {fps} -i - -vf "zscale=primaries=bt2020:matrix=bt2020nc:transfer=smpte2084,format=yuv420p10le" -c:v libx265 -pix_fmt yuv420p10le -crf 14 -preset slow -x265-params "hdr-opt=1:repeat-headers=1:master-display={master_display}:max-cll={max_cll}" -color_primaries bt2020 -colorspace bt2020nc -color_trc smpte2084 {shlex.quote(out_path)}
"""
    return subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)

def write_hdr10_frames(frames_uint16_iter, width, height, fps, out_path, **kw):
    """
    Convenience: write an iterable of uint16 RGB frames (display‑referred PQ) to HDR10.
    """
    proc = open_hdr10_writer(width, height, fps, out_path, **kw)
    try:
        for f in frames_uint16_iter:
            assert f.dtype == np.uint16 and f.shape[0] == height and f.shape[1] == width and f.shape[2] == 3
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        return proc.wait()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

def tonemap_preview(arr_linear):
    """
    Simple Reinhard-ish tonemap for 8‑bit preview (returns uint8 sRGB-ish).
    Do NOT use this for final HDR output.
    """
    x = arr_linear / (1.0 + arr_linear)
    x = np.power(np.clip(x, 0.0, 1.0), 1/2.2)
    return (x * 255.0).astype(np.uint8)
