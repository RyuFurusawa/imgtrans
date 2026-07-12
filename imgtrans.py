"""imgtrans - 後方互換シム

このファイル自体には実装はなく、`imgtrans_lib` パッケージから全公開 API を
再エクスポートしている。`import imgtrans as imgtrans` のような既存の
呼び出しコードはそのまま動作する。

実装の本体は以下のモジュールを参照:

    imgtrans_lib/
        _jit_kernels.py        Numba JIT カーネル
        _hdr.py                HDR EOTF/OETF, hdr_pq_to_srgb
        _utils.py              小規模ヘルパ
        _video_io.py           rendered_npys_to_mov など動画 I/O
        _dm_core.py            drawManeuver: __init__ / 入力解析 / 静的 EOTF
        _dm_io_video.py        drawManeuver: ffmpeg/PyAV シンク
        _dm_frame_proc.py      drawManeuver: フレーム処理
        _dm_data_ops.py        drawManeuver: append/zArange など
        _dm_transforms_apply.py drawManeuver: applyTime*/applySpace*/apply*Blur
        _dm_transforms_add.py  drawManeuver: addTrans/addCycle/rooting*
        _dm_rendering.py       drawManeuver: transprocess / new_transprocess
        _dm_visualize.py       drawManeuver: maneuver_2dplot/3dplot/animationout

元の単一ファイル版は imgtrans2026_pre_modular_backup.py にバックアップしている。
"""

from imgtrans_lib import *
from imgtrans_lib import (
    drawManeuver,
    # 内部でも使われる「アンダースコア付き」だが歴史的に公開されているもの
    _HAS_NUMBA,
    _process_frame_vertical_jit,
    _process_frame_horizontal_jit,
    _pq_eotf,
    _srgb_oetf,
    _hlg_eotf,
    _tonemap_hdr_rgb01_to_srgb,
    _probe_video_transfer,
    # video_io 内部ヘルパ (UltraHDR / cv2 連番)
    _rendered_mov_to_seq_cv2,
    _rendered_mov_to_ultrahdr_seq,
    _convert_to_adobe_ultrahdr,
    _strip_iso21496_markers,
    _build_hdrgm_xmp,
    _build_xmp_app1,
    _build_mpf_app2,
)

try:
    from imgtrans_lib import (
        _process_frame_vertical_yuv_jit,
        _process_frame_horizontal_yuv_jit,
    )
except ImportError:
    pass


if __name__ == '__main__':
    print("imgtrans (modular shim) — see imgtrans_lib/ for implementation.")
