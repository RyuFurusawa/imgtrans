"""drawManeuver のフレーム処理 (FrameProcMixin)

含むもの:
- print_progress     : 進捗表示 (時間 / メモリ / フレーム)
- _process_frame*    : 1 フレームをスリットスキャンして images[] に書き込み (RGB / YUV)
- _build_frame_map   : maneuver 配列から (frame -> [(i, p, src_x_or_y), ...]) マップを構築
- _iterate_frames    : 入力動画/シーケンスをフレーム単位で取り出すジェネレータ
- _determine_output_path / _should_render_now / _should_open_sink
- _render_images_to_sink / _render_images_to_sink_yuv : セパレート毎の書き出し
"""
import os
import sys
import gc
import time
import psutil
import cv2
import numpy as np
import av

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
    _process_frame_vertical_yuv_jit = None
    _process_frame_horizontal_yuv_jit = None


class FrameProcMixin:
    def print_progress(self,
                    current: int,
                    total: int,
                    color_code: str = "31",
                    suffix: str = "",
                    newline: bool = False):
        """
        進捗バーを表示する（常に1本のみ）
        current   : 現在の進行数
        total     : 全体数
        color_code: ANSIカラーコード (31=赤, 32=緑, 33=黄など)
        suffix    : バーの後に表示する文字列（fps, ms/slit など）
        """
        current = int(current)
        total   = max(1, int(total))  # ゼロ割り防止

        filled = int(current * self.progressbarsize / total)
        empty  = self.progressbarsize - filled
        bar = '■' * filled + '.' * empty
        percent = current / total * 100

        # print(f"\r\033[K[\033[{color_code}m{bar}\033[39m] {percent:.0f}% {suffix}", end="")
        if newline:
            end_char = "\n"
        else:
            end_char = ""

        sys.stdout.write(
            f"\r\033[K[\033[{color_code}m{bar}\033[39m] {percent:.0f}% {suffix}{end_char}"
        )
        sys.stdout.flush()

    #added 20250921
    # def _process_frame(
    #     self, img, frame_index, wr_arrays, images, seq_read_s,
    #     minz, maxz, totalnum, totalslits,
    #     stime, memory_report=False, memory_stats=None,
    #     progress_interval=100
    #     ):
    #     """
    #     1フレーム処理（スキャン＋進捗表示＋メモリ監視）
    #     """
    #     last_size = 0
    #     slitscounter = 0
    #     for i, wr in enumerate(wr_arrays, start=seq_read_s):
    #         indices = np.where(wr[:, 1] == frame_index)[0]
    #         if indices.size == 0:
    #             continue
    #         if self.scan_direction % 2 == 0:  # 横スキャン
    #             for p in indices:
    #                 images[i - seq_read_s, p, :] = img[wr[p, 0], :]
    #         else:  # 縦スキャン
    #             for p in indices:
    #                 images[i - seq_read_s, :, p] = img[:, wr[p, 0]]
    #         slitscounter += indices.size
    #         last_size = indices.size

    #     # --- 時間計測 ---
    #     etime = time.time()
    #     interval = ((etime - stime) / (last_size or 1)) * 1000

    #     # --- 進捗表示（間引き可能） ---
    #     if frame_index % progress_interval == 0:
    #         suffix = (
    #             f"frame{(frame_index - minz)/totalnum*100:.02f}% "
    #             f"({minz}>{frame_index}>{maxz}) "
    #             f"Slits{slitscounter/totalslits*100:.03f}% "
    #             f"({slitscounter}/{int(totalslits)}) : {round(interval,2)}ms/slit"
    #         )
    #         self.print_progress(
    #             current=frame_index - minz,
    #             total=totalnum,
    #             color_code="33",
    #             suffix=suffix
    #         )

    #     # --- メモリ使用状況を記録 ---
    #     if memory_report and memory_stats is not None:
    #         vmem = psutil.virtual_memory()
    #         memory_stats.append([
    #             interval, images.nbytes, vmem.total, vmem.available,
    #             vmem.used, vmem.percent, vmem.active, vmem.inactive,
    #             vmem.wired, vmem.free
    #         ])

    #2025-1004書き換え　np.where 全廃止 → 代わりに frame_to_indices を使う
    #2026-0320 Numba JIT 高速化
    def _process_frame(
        self, img, frame_index, frame_to_indices, wr_arrays, images,gap=0,slit_step=1):
        slits = frame_to_indices.get(frame_index, [])
        if not slits:
            print("処理するスリットなし",frame_index)
            return 0

        # 配列化
        i_arr = np.array([i for i, _ in slits], dtype=np.int32)
        p_arr = np.array([p for _, p in slits], dtype=np.int32)

        if _HAS_NUMBA:
            # --- Numba JIT 高速パス ---
            src_coords = wr_arrays[i_arr, p_arr, 0].astype(np.int32)
            if self.scan_direction % 2 == 0:
                img_h = img[:, ::slit_step] if slit_step != 1 else img
                _process_frame_horizontal_jit(img_h, i_arr, p_arr, src_coords, images, gap)
            else:
                img_v = img[::slit_step] if slit_step != 1 else img
                _process_frame_vertical_jit(img_v, i_arr, p_arr, src_coords, images, gap)
        else:
            # --- Numba未対応時: 従来のNumPyパス ---
            if self.scan_direction % 2 == 0:
                src_y = wr_arrays[i_arr, p_arr, 0]
                images[i_arr + gap, p_arr, :] = img[src_y, ::slit_step]
            else:
                for i_val, p_val in zip(i_arr, p_arr):
                    src_x = wr_arrays[i_val, p_val, 0]
                    images[i_val + gap, :, p_val] = img[::slit_step, src_x]

        return len(slits)   # ← このフレームで処理したスリット数を返す

    def _process_frame_yuv(
        self, y_img, cb_img, cr_img,
        frame_index, frame_to_indices, wr_arrays,
        img_y, img_cb, img_cr, gap=0, slit_step=1):
        """
        YUV-native スリットスキャン処理。
        Y/Cb/Cr の3プレーンをそれぞれ独立に処理する。
        422 の場合 Cb/Cr は水平半幅なので、列インデックスを //2 する。
        """
        slits = frame_to_indices.get(frame_index, [])
        if not slits:
            return 0

        i_arr = np.array([i for i, _ in slits], dtype=np.int32)
        p_arr = np.array([p for _, p in slits], dtype=np.int32)
        src_coords = wr_arrays[i_arr, p_arr, 0].astype(np.int32)

        if _HAS_NUMBA:
            if self.scan_direction % 2 == 0:
                y_h  = y_img[:, ::slit_step]  if slit_step != 1 else y_img
                cb_h = cb_img[:, ::slit_step] if slit_step != 1 else cb_img
                cr_h = cr_img[:, ::slit_step] if slit_step != 1 else cr_img
                _process_frame_horizontal_yuv_jit(
                    y_h, cb_h, cr_h, i_arr, p_arr, src_coords,
                    img_y, img_cb, img_cr, gap)
            else:
                y_v  = y_img[::slit_step]  if slit_step != 1 else y_img
                cb_v = cb_img[::slit_step] if slit_step != 1 else cb_img
                cr_v = cr_img[::slit_step] if slit_step != 1 else cr_img
                _process_frame_vertical_yuv_jit(
                    y_v, cb_v, cr_v, i_arr, p_arr, src_coords,
                    img_y, img_cb, img_cr, gap)
        else:
            # NumPy fallback
            if self.scan_direction % 2 == 0:
                src_y = wr_arrays[i_arr, p_arr, 0]
                img_y[i_arr + gap, p_arr, :]  = y_img[src_y, ::slit_step]
                img_cb[i_arr + gap, p_arr, :] = cb_img[src_y, ::slit_step]
                img_cr[i_arr + gap, p_arr, :] = cr_img[src_y, ::slit_step]
            else:
                for i_val, p_val in zip(i_arr, p_arr):
                    src_x = wr_arrays[i_val, p_val, 0]
                    img_y[i_val + gap, :, p_val]      = y_img[::slit_step, src_x]
                    img_cb[i_val + gap, :, p_val // 2] = cb_img[::slit_step, src_x // 2]
                    img_cr[i_val + gap, :, p_val // 2] = cr_img[::slit_step, src_x // 2]

        return len(slits)

    # ====================================================================
    # transprocess 用ヘルパー群 (2026-03 追加)
    # ====================================================================

    def _build_frame_map(self, wr_arrays, fps_scale=1):
        """
        frame_to_indices を NumPy でベクトル化構築する。
        旧: Python 二重ループ O(N*M) → 新: NumPy O(N*M) だがC層で実行

        戻り値: dict[int, (np.ndarray, np.ndarray)]
            key   = frame_index (z値)
            value = (i_arr, p_arr) — 共にint32の1D配列
        """
        # wr_arrays shape: (n_frames, n_slits, >=2)  列1がz値
        z_all = wr_arrays[:, :, 1]  # shape (F, S)
        if fps_scale != 1:
            z_keys = (z_all * fps_scale).astype(np.int32)
        else:
            z_keys = z_all.astype(np.int32)

        # フラットにして (i, p) のインデックスを作る
        n_frames, n_slits = z_keys.shape
        i_idx, p_idx = np.mgrid[0:n_frames, 0:n_slits]
        z_flat = z_keys.ravel()
        i_flat = i_idx.ravel().astype(np.int32)
        p_flat = p_idx.ravel().astype(np.int32)

        # ソートしてグルーピング
        order = np.argsort(z_flat, kind='mergesort')
        z_sorted = z_flat[order]
        i_sorted = i_flat[order]
        p_sorted = p_flat[order]

        # 境界を見つける
        breaks = np.where(np.diff(z_sorted) != 0)[0] + 1
        z_groups = np.split(z_sorted, breaks)
        i_groups = np.split(i_sorted, breaks)
        p_groups = np.split(p_sorted, breaks)

        frame_map = {}
        for zg, ig, pg in zip(z_groups, i_groups, p_groups):
            frame_map[int(zg[0])] = (ig, pg)

        return frame_map

    def _process_frame_v2(self, img, frame_index, frame_map, wr_arrays,
                          images, gap=0, slit_step=1):
        """
        _process_frame のベクトル化版。
        縦スキャンも完全ベクトル化（ループ廃止）。
        """
        entry = frame_map.get(frame_index)
        if entry is None:
            return 0

        i_arr, p_arr = entry
        src_coords = wr_arrays[i_arr, p_arr, 0]  # 参照座標

        if self.scan_direction % 2 == 0:
            # 横スキャン: images[i, p, :] = img[src_y, ::step]
            images[i_arr + gap, p_arr, :] = img[src_coords, ::slit_step]
        else:
            # 縦スキャン: images[i, :, p] = img[::step, src_x]
            # 完全ベクトル化: src列を一括gather → transpose代入
            cols = img[::slit_step, src_coords]  # shape (H_eff, N)
            # cols[:, k] → images[i_arr[k]+gap, :, p_arr[k]]
            # advanced indexing で一括代入
            dest_i = i_arr + gap
            # images の shape: (F, H_eff, S, 3) ではなく (F, H_eff, S) (3ch は最終軸)
            # 実際 images shape = (F, H, S, 3)... いや (F, H, S) は scan_direction=1 のとき
            # images[dest_i, :, p_arr] に cols.T の各行を入れたい
            # images[dest_i[k], :, p_arr[k]] = cols[:, k]
            # → numpy advanced indexing: 軸0,2 を指定、軸1 は全スライス
            for k in range(len(i_arr)):
                images[dest_i[k], :, p_arr[k]] = cols[:, k]

        return len(i_arr)

    def _iterate_frames(self, minz, maxz, needed_frames, fps_for_pyav=None):
        """
        ソースフレームを順番に yield する統一ジェネレータ。
        OpenCV / PyAV の分岐を呼び出し側から隠蔽する。

        yields: (frame_index: int, frame_data: np.ndarray)
        """
        if self.container is not None:
            # === PyAV ===
            pyav_fmt = "rgb48le" if self.is_morethan_8bit else "rgb24"
            read_fps = fps_for_pyav if fps_for_pyav else self.inputmovfps
            start_sec = minz / read_fps
            start_pts = int(start_sec / self.stream.time_base)
            self.container.seek(start_pts, stream=self.stream,
                                any_frame=False, backward=True)
            for frame in self.container.decode(self.stream):
                idx = int(round(frame.pts * self.stream.time_base * read_fps))
                if idx < minz:
                    continue
                if idx > maxz:
                    break
                if idx not in needed_frames:
                    continue
                yield idx, frame.to_ndarray(format=pyav_fmt)
                needed_frames.discard(idx)
                if not needed_frames:
                    break
        else:
            # === OpenCV ===
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(minz))
            num = int(minz)
            while num <= int(maxz):
                ret, frame = self.cap.read()
                if not ret:
                    break
                if num in needed_frames:
                    # OpenCV は BGR で返すが、パイプライン全体は RGB 統一
                    yield num, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    needed_frames.discard(num)
                    if not needed_frames:
                        break
                num += 1

    def _determine_output_path(self, videostr, out_type, s, sep_start_num, sepVideoOut):
        """出力ファイルパスを決定する"""
        ext = '.mov' if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444,
                                      self.OUT_PRORES_422_SDR) else '.mp4'
        if sepVideoOut <= 0:
            return videostr + ext
        elif sepVideoOut == 1:
            return f"{videostr}_sep{s}" + ext
        else:
            block_start = s - ((s - sep_start_num) % sepVideoOut)
            block_end = block_start + sepVideoOut - 1
            return f"{videostr}_sep{block_start}-{block_end}" + ext

    def _should_render_now(self, s, sep_start_num, sep_end_num, sepVideoOut, separate_num):
        """このセグメント s の後にレンダリングすべきか判定"""
        if sepVideoOut == 1 or separate_num == 1:
            return True
        if sepVideoOut > 1:
            at_block_end = (s - sep_start_num) % sepVideoOut == sepVideoOut - 1
            at_last = s == sep_end_num - 1
            return at_block_end or at_last
        return False

    def _should_open_sink(self, s, sep_start_num, sepVideoOut, separate_num):
        """このセグメント s の前にsinkを開くべきか判定"""
        if sepVideoOut <= 1 or separate_num == 1:
            return True
        return (s - sep_start_num) % sepVideoOut == 0

    def _render_images_to_sink(self, images, sink_kind, sink_obj, out_type,
                               xy_trans_out, rotate_direction):
        """images配列をビデオsinkに書き出す"""
        for i in range(images.shape[0]):
            if out_type == 0:
                img_name = f"img/{self.ORG_NAME}_{i}p{self.imgtype}"
                self._save_image_with_profile(img_name, images[i])
            else:
                self._write_video_frame(sink_kind, sink_obj, images[i], out_type,
                                        xy_trans_out=xy_trans_out,
                                        rotate_direction=rotate_direction)
            if i % 50 == 0:  # gc.collectの頻度を大幅削減（毎フレーム→50フレーム毎）
                gc.collect()
            self.print_progress(current=i, total=images.shape[0],
                                color_code="31", suffix=f"rendering {i}/{images.shape[0]}")
        print()

    def _render_images_to_sink_yuv(self, img_y, img_cb, img_cr, sink_kind, sink_obj, out_type,
                                    xy_trans_out, rotate_direction):
        """YUV-native: Y/Cb/Crプレーンをビデオsinkに書き出す"""
        n_frames = img_y.shape[0]
        for i in range(n_frames):
            frame_tuple = (img_y[i], img_cb[i], img_cr[i])
            self._write_video_frame(sink_kind, sink_obj, frame_tuple, out_type,
                                    xy_trans_out=xy_trans_out,
                                    rotate_direction=rotate_direction,
                                    use_yuv_native=True)
            if i % 50 == 0:
                gc.collect()
            self.print_progress(current=i, total=n_frames,
                                color_code="31", suffix=f"rendering(YUV) {i}/{n_frames}")
        print()

       #added 2023 10/6
