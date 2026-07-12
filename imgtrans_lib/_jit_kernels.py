"""Numba JIT 高速ピクセルコピー (利用可能時のみ)

スリットスキャンのピクセル単位コピーループを Numba で最適化する。
Numba 不在時は通常 Python 関数として動作する (パフォーマンスは落ちる)。
"""

# ===== Numba JIT 高速ピクセルコピー (利用可能時のみ) =====
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    # Numba未対応時: デコレータを無効化するダミー
    def njit(*args, **kwargs):
        def _wrapper(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrapper

@njit(cache=True)
def _process_frame_vertical_jit(img, i_arr, p_arr, src_x_arr, images, gap):
    """縦スキャン: images[i+gap, :, p] = img[:, src_x] を高速実行"""
    h = img.shape[0]
    n = len(i_arr)
    c = img.shape[2] if img.ndim == 3 else 0
    if c > 0:
        for k in range(n):
            di = i_arr[k] + gap
            pv = p_arr[k]
            sx = src_x_arr[k]
            for row in range(h):
                for ch in range(c):
                    images[di, row, pv, ch] = img[row, sx, ch]
    else:
        for k in range(n):
            di = i_arr[k] + gap
            pv = p_arr[k]
            sx = src_x_arr[k]
            for row in range(h):
                images[di, row, pv] = img[row, sx]

@njit(cache=True)
def _process_frame_horizontal_jit(img, i_arr, p_arr, src_y_arr, images, gap):
    """横スキャン: images[i+gap, p, :] = img[src_y, :] を高速実行"""
    w = img.shape[1]
    n = len(i_arr)
    c = img.shape[2] if img.ndim == 3 else 0
    if c > 0:
        for k in range(n):
            di = i_arr[k] + gap
            pv = p_arr[k]
            sy = src_y_arr[k]
            for col in range(w):
                for ch in range(c):
                    images[di, pv, col, ch] = img[sy, col, ch]
    else:
        for k in range(n):
            di = i_arr[k] + gap
            pv = p_arr[k]
            sy = src_y_arr[k]
            for col in range(w):
                images[di, pv, col] = img[sy, col]

# === YUV-native JIT (422 slit-scan) ===
if _HAS_NUMBA:
    @njit(cache=True)
    def _process_frame_vertical_yuv_jit(y_img, cb_img, cr_img,
                                         i_arr, p_arr, src_x_arr,
                                         img_y, img_cb, img_cr, gap):
        """縦スキャン YUV422: Y列コピー + Cb/Cr は x//2"""
        h = y_img.shape[0]
        n = len(i_arr)
        for k in range(n):
            di = i_arr[k] + gap
            pv = p_arr[k]
            sx = src_x_arr[k]
            for row in range(h):
                img_y[di, row, pv] = y_img[row, sx]
            pc = pv // 2
            sc = sx // 2
            for row in range(h):
                img_cb[di, row, pc] = cb_img[row, sc]
                img_cr[di, row, pc] = cr_img[row, sc]

    @njit(cache=True)
    def _process_frame_horizontal_yuv_jit(y_img, cb_img, cr_img,
                                           i_arr, p_arr, src_y_arr,
                                           img_y, img_cb, img_cr, gap):
        """横スキャン YUV422: Y行コピー + Cb/Cr行コピー（幅は半分）"""
        w_y = img_y.shape[2]
        w_c = img_cb.shape[2]
        n = len(i_arr)
        for k in range(n):
            di = i_arr[k] + gap
            pv = p_arr[k]
            sy = src_y_arr[k]
            for col in range(w_y):
                img_y[di, pv, col] = y_img[sy, col]
            for col in range(w_c):
                img_cb[di, pv, col] = cb_img[sy, col]
                img_cr[di, pv, col] = cr_img[sy, col]

if _HAS_NUMBA:
    print("Numba JIT: enabled (高速レンダリング)")
else:
    print("Numba JIT: disabled (NumPy互換性問題 - pip install 'numpy<2.1' で有効化可能)")
