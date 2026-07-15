"""imgtrans_lib パッケージ

旧 imgtrans2026.py を機能別モジュールに分割した実装。
後方互換のため、`import imgtrans2026 as imgtrans` でアクセスしていた
全ての公開 API はこのパッケージのトップレベルからも import 可能。

主要構成:
  _jit_kernels      Numba JIT (スリットスキャン高速ピクセルコピー)
  _hdr              PQ/HLG EOTF/OETF, hdr_pq_to_srgb など HDR ヘルパ
  _utils            split_*, addmovfile, returnfps, bezier_interpolation 等の小ヘルパ
  _video_io         rendered_npys_to_mov / rearrange_wide_video / rendered_mov_to_seq /
                    convert_npy_to_jpg / export_segments など、トップレベルの動画 I/O
  _dm_core          drawManeuver: __init__, 入力解析, 静的 EOTF/OETF, ICC 埋め込み
  _dm_io_video      drawManeuver: ffmpeg/PyAV シンク管理、yuv420 → rgb 変換
  _dm_frame_proc    drawManeuver: フレームマッピング、ピクセル書き込み、進捗表示
  _dm_data_ops      drawManeuver: append/prepend/zArange/dataCheck 等のデータ操作
  _dm_transforms_apply  drawManeuver: applyTime*/applySpace*/apply*Blur など適用系
  _dm_transforms_add    drawManeuver: addTrans/addCycle/rooting* など追加系
  _dm_rendering     drawManeuver: transprocess / new_transprocess / pretransprocess /
                                  overlay_tc_rate (メインレンダリング)
  _dm_visualize     drawManeuver: maneuver_2dplot/3dplot/imgplot/animationout など可視化
  _dm_audio         drawManeuver: audio_render/scd_out_v2/maneuver_fourier_out (音声出力)
"""

# ---- スタンドアロン関数群（旧 imgtrans2026 のトップレベル） ----
from ._jit_kernels import (
    _HAS_NUMBA,
    _process_frame_vertical_jit,
    _process_frame_horizontal_jit,
)
try:
    from ._jit_kernels import (
        _process_frame_vertical_yuv_jit,
        _process_frame_horizontal_yuv_jit,
    )
except ImportError:
    pass

from ._hdr import (
    _pq_eotf,
    _srgb_oetf,
    _hlg_eotf,
    _tonemap_hdr_rgb01_to_srgb,
    _probe_video_transfer,
    hdr_pq_to_srgb,
)

from ._utils import (
    split_list_based_on_elements,
    split_array,
    split_uniq_based_on_original,
    addmovfile,
    returnfps,
    closest_value,
    search,
    addCsvHeader,
    append_to_logfile,
    bezier_interpolation,
    extract_number,
    extract_base_string,
    convert_npy_to_jpg,
    custom_blur,
    custom_onedimention_blur,
    onedimention_LoopBlur,
    frames_to_min_sec,
    calculate_parallel_perpendicular,
    create_video_from_images,
    double_first_dimension_with_interpolation,
)

from ._video_io import (
    rendered_npys_to_mov,
    rearrange_wide_video,
    rendered_mov_to_seq,
    export_segments,
    # 後方互換: アンダースコア付きだが旧 imgtrans2026 のトップレベルにあった内部ヘルパ
    _rendered_mov_to_seq_cv2,
    _rendered_mov_to_ultrahdr_seq,
    _convert_to_adobe_ultrahdr,
    _strip_iso21496_markers,
    _build_hdrgm_xmp,
    _build_xmp_app1,
    _build_mpf_app2,
)

# ---- drawManeuver ミックスイン ----
from ._dm_core import CoreMixin
from ._dm_io_video import IOVideoMixin
from ._dm_frame_proc import FrameProcMixin
from ._dm_data_ops import DataOpsMixin
from ._dm_transforms_apply import TransformsApplyMixin
from ._dm_transforms_add import TransformsAddMixin
from ._dm_rendering import RenderingMixin
from ._dm_visualize import VisualizeMixin
from ._dm_audio import AudioMixin


class drawManeuver(
    CoreMixin,
    IOVideoMixin,
    FrameProcMixin,
    DataOpsMixin,
    TransformsApplyMixin,
    TransformsAddMixin,
    RenderingMixin,
    VisualizeMixin,
    AudioMixin,
):
    """画像変換マニューバ用のメインクラス。

    各機能はミックスインに分割されているが、利用側からは従来通り
    `drawManeuver(...)` でインスタンス化し、`.transprocess()` や
    `.addTrans()` などをそのまま呼べる。
    """
    pass
