"""drawManeuver のメインレンダリング (RenderingMixin)

self.data にエンコードされた maneuver を実映像/連番画像にレンダリングする中核:
- transprocess        : 旧来の主処理 (CPU/メモリを節約しつつ巨大な動画を作る)
- new_transprocess    : YUV ネイティブ等を含む新パイプライン
- pretransprocess     : 解像度を落とした事前確認レンダ
- overlay_tc_rate     : 既にレンダ済みの動画に TC や速度を焼き込む
"""
import os
import gc
import csv
import math
import time
import shutil
import subprocess
import inspect
import psutil
import cv2
import numpy as np
import av

from .hdr_io import oetf_from_scene_linear

from ._utils import append_to_logfile


class RenderingMixin:
    def transprocess(self, separate_num=None, sep_start_num=0, sep_end_num=None,
                     out_type=1, xy_trans_out=False, render_mode=0,
                     title_atr: str = None, tmp_para_images=False, del_data=True,
                     auto_memory_clear=False, memory_report=False, tmp_save=False,
                     render_clip_start=0, render_clip_end=None,
                     tmp_path_base=None, slit_step=1, scan_step=1, use_pyav=False):
        """
        new_transprocess() のリファクタリング版。
        構造改善:
          - _build_frame_map()       : frame_to_indices構築を NumPy ベクトル化 (旧: Python二重ループ)
          - _process_frame_v2()      : スリット配置（縦横）の統一処理
          - _iterate_frames()        : OpenCV/PyAV のフレーム読み込みを統一ジェネレータ化
          - _should_render_now()     : セグメント出力タイミング判定を関数化
          - _render_images_to_sink() : 出力処理を統一（6箇所のコピペ → 1箇所）
          - gc.collect() 呼び出し頻度の削減 (毎フレーム → 50フレーム毎)
        """
        # --- パラメータ補正 ---
        slit_step = max(1, slit_step)
        scan_step = max(1, scan_step)
        eff_height = int(self.height) // slit_step
        eff_width  = int(self.width)  // slit_step

        if slit_step > 1 or scan_step > 1:
            print(f"slit_step={slit_step}, scan_step={scan_step}: "
                  f"output reduced (slit: {int(self.height)}→{eff_height} / "
                  f"{int(self.width)}→{eff_width}, "
                  f"scan_nums: {self.data.shape[1]}→{self.data.shape[1]//scan_step})")

        # ============================================================
        # Phase 1: バリデーション
        # ============================================================
        if self.input_pix_fmt == 'yuv420p10le' and out_type != 0:
            msg = "Error! yuv420p10le is not supported for video output."
            print(msg)
            append_to_logfile(msg)
            return

        if self.cap is not None and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
            print("capの映像データが外れている。再読み込みします。")
            self.cap = cv2.VideoCapture(self.VIDEO_PATH)

        if np.amin(self.data[:, :, -1]) < 0:
            print("z<0, error")
            append_to_logfile("z<0,error")
            return
        if np.amax(self.data[:, :, -1]) > self.count:
            print(f"z>video_count, error, data_zmax: {np.amax(self.data[:,:,1])} > count: {self.count}")
            append_to_logfile(f"z>video_count,error")

        # ============================================================
        # Phase 2: メモリ計算・分割数決定
        # ============================================================
        if render_clip_end is None:
            render_clip_end = self.data.shape[0]
        render_range = render_clip_end - render_clip_start

        color_depth = 16 if getattr(self, "input_bit_depth", 8) > 8 else 8
        file_size_bytes = (render_range * (self.slit_length // slit_step) *
                           (self.data.shape[1] // scan_step) * color_depth * 3) / 8
        file_size_mb = file_size_bytes / (1024 * 1024)

        if separate_num is None:
            avail_mb = psutil.virtual_memory().available / (1024 ** 2)
            separate_num = math.ceil(file_size_mb / (avail_mb * (self.memory_percent / 100)))
        print(f"separate_num={separate_num}, active memory={psutil.virtual_memory().available/(1024**2):.0f}mb")

        if sep_end_num is None:
            sep_end_num = separate_num

        # ============================================================
        # Phase 3: 出力名・ログ
        # ============================================================
        runFirstTime = time.time()
        XY_Name = "Y" if self.scan_direction % 2 == 0 else "X"

        if render_mode in (0, 2):
            videostr = f"{self.ORG_NAME}_{self.out_name_attr}"
        else:
            videostr = f"{self.ORG_NAME}_{self.out_name_attr}({sep_start_num}-{sep_end_num}sep)"

        if render_mode == 3 and tmp_path_base is None:
            tmp_path_base = f"{self.ORG_NAME}_{self.out_name_attr}"
            if title_atr:
                tmp_path_base += title_atr

        if not self.embedHistory_intoName:
            videostr = f"{self.ORG_NAME}_process{self.log}"
        if title_atr:
            videostr += title_atr

        append_to_logfile(f"transprocess:{separate_num}separate")
        append_to_logfile(f"active memory={psutil.virtual_memory().available/(1024**2):.0f}mb")
        append_to_logfile(f"memory_usage={((file_size_mb / separate_num) / (psutil.virtual_memory().available/(1024**2)))*100:.1f}%")
        append_to_logfile(videostr)
        rotate_direction = False

        print(f"framecount={render_range} ({render_range/self.recfps:.1f}sec) "
              f"{XY_Name}(out)={self.data.shape[1]} refer={self.data.shape[2]}")
        _data_slice = self.data[render_clip_start:render_clip_end]
        print(f"z min-max = {np.amin(_data_slice[:,:,-1])} "
              f"{np.amax(_data_slice[:,:,-1])}")
        print(f"space min-max = {np.amin(_data_slice[:,:,0])} "
              f"{np.amax(_data_slice[:,:,0])}")
        append_to_logfile(f"outfps={self.outfps}")
        append_to_logfile(f"framecount={render_range}")
        append_to_logfile(f"z min-max ={np.amin(_data_slice[:,:,-1])}-{np.amax(_data_slice[:,:,-1])}")
        append_to_logfile(f"space min-max ={np.amin(_data_slice[:,:,0])}-{np.amax(_data_slice[:,:,0])}")

        # ============================================================
        # Phase 4: wr_array 準備
        # ============================================================
        wr_array = self.data[render_clip_start:render_clip_end, ::scan_step, :].astype(np.int32)

        # 分割切れ目プロット
        cut_points = []
        if render_clip_start != 0:
            cut_points.append(render_clip_start)
        for i in range(1, separate_num):
            cut_points.append(wr_array.shape[0] * i // separate_num + render_clip_start)
        if render_clip_end != self.data.shape[0]:
            cut_points.append(render_clip_end)
        self.maneuver_2dplot(x_positions=cut_points, s_frame=render_clip_start, e_frame=render_clip_end)

        if del_data:
            del self.data

        # ディレクトリ準備
        if out_type == 0 and not os.path.isdir("img"):
            os.makedirs("img")
        if self.sepVideoOut != 1 and not os.path.isdir("tmp"):
            os.makedirs("tmp")

        # ============================================================
        # Phase 5: ソース映像のオープン（OpenCV or PyAV）
        # ============================================================
        if self.is_morethan_8bit or use_pyav:
            if out_type == 1 and self.is_morethan_8bit:
                out_type = 2
            self.container = av.open(self.VIDEO_PATH)
            self.stream = self.container.streams.video[0]
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(self.VIDEO_PATH)
            self.container = None

        # ============================================================
        # Phase 6: メインループ（セグメント単位）
        # ============================================================
        if render_mode < 3:
            for s in range(sep_start_num, sep_end_num):
                if memory_report:
                    memory_stats = []

                # --- 出力画像バッファ確保 ---
                base_frames = wr_array.shape[0] // separate_num
                seg_frames = base_frames + (wr_array.shape[0] % separate_num if s == sep_end_num - 1 else 0)

                if self.scan_direction % 2 == 1:
                    img_shape = (eff_height, int(wr_array.shape[1]), 3)
                else:
                    img_shape = (int(wr_array.shape[1]), eff_width, 3)

                image_dtype = np.uint16 if self.is_morethan_8bit else np.uint8
                images = np.zeros((seg_frames, *img_shape), dtype=image_dtype)

                seq_read_s = s * base_frames
                seq_read_e = seq_read_s + seg_frames

                print(f"img-declare {seq_read_s} -> {seq_read_e} (dtype={image_dtype}, shape={images.shape})")

                # --- z範囲 ---
                minz = int(np.amin(wr_array[seq_read_s:seq_read_e, :, 1]))
                maxz = int(np.amax(wr_array[seq_read_s:seq_read_e, :, 1]))
                print(f"minz={minz}, maxz={maxz}")

                if minz < 0:
                    print("z<0, error")
                    return

                totalslits = seg_frames * wr_array.shape[1]
                slitscounter = 0
                next_update = totalslits // 100

                # --- fps_conversion チェック ---
                depth_slice = self.depth_to_sel_recfps[render_clip_start + seq_read_s:render_clip_start + seq_read_e]
                fps_conversion = len(depth_slice) > 0

                print(f"count={self.count}, processing-Frames={seg_frames}, fps_conversion={fps_conversion}")

                sstime = time.time()

                # ===========================================
                # ソース: 画像シーケンス
                # ===========================================
                if self.cap is None and self.container is None:
                    image_files = sorted([
                        f for f in os.listdir(f"{self.ORG_PATH}/{self.ORG_NAME}")
                        if f.endswith(('.png', '.jpg', '.tif', '.jpeg', '.bmp', '.npy'))
                    ])
                    wr_seg = wr_array[seq_read_s:seq_read_e]
                    frame_map = self._build_frame_map(wr_seg)

                    for num in sorted(frame_map.keys()):
                        stime = time.time()
                        image_path = os.path.join(f"{self.ORG_PATH}/{self.ORG_NAME}", image_files[num])
                        frame = np.load(image_path) if image_path.endswith('.npy') else cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                        processed = self._process_frame_v2(
                            frame, num, frame_map, wr_seg, images,
                            slit_step=slit_step
                        )
                        slitscounter += processed

                        if slitscounter >= next_update:
                            self.print_progress(
                                current=slitscounter, total=totalslits,
                                color_code="33",
                                suffix=f"frame={num} slits={slitscounter}/{totalslits} "
                                       f"{time.time()-stime:.2f}sec/f"
                            )
                            next_update += totalslits // 100
                    print()

                # ===========================================
                # ソース: 動画（複数FPS）
                # ===========================================
                elif fps_conversion:
                    print("複数fpsレンダリング")
                    renderfps_array = np.unique(depth_slice)
                    print(renderfps_array)

                    for fps in renderfps_array:
                        print(f"sel fps={fps}")
                        # ソース映像を切り替え
                        fps_idx = self.some_recfps_array.index(fps)
                        if self.is_morethan_8bit or use_pyav:
                            self.container = av.open(self.another_videos[fps_idx])
                            self.stream = self.container.streams.video[0]
                        else:
                            self.cap = cv2.VideoCapture(self.another_videos[fps_idx])

                        # 連続区間に分割
                        indices_raw = np.where(depth_slice == fps)[0]
                        breaks = np.where(np.diff(indices_raw) > 1)[0] + 1
                        some_indices = np.split(indices_raw, breaks)

                        for seg_indices in some_indices:
                            range_start = seg_indices[0] + seq_read_s
                            range_end = seg_indices[-1] + seq_read_s + 1
                            fps_scale = self.renderfps_scales[fps_idx]
                            gap = seg_indices[0]  # images 内でのオフセット

                            seg_minz = int(np.amin(wr_array[range_start:range_end, :, 1]) * fps_scale)
                            seg_maxz = int(np.amax(wr_array[range_start:range_end, :, 1]) * fps_scale)
                            print(f"range={range_start}-{range_end}, fps_scale={fps_scale}, z={seg_minz}-{seg_maxz}")

                            if seg_minz < 0:
                                print("z<0, error")
                                return

                            wr_seg = wr_array[range_start:range_end]
                            frame_map = self._build_frame_map(wr_seg, fps_scale=fps_scale)
                            needed = set(frame_map.keys())

                            for fidx, fdata in self._iterate_frames(seg_minz, seg_maxz, needed,
                                                                     fps_for_pyav=fps):
                                processed = self._process_frame_v2(
                                    fdata, fidx, frame_map, wr_seg, images,
                                    gap=gap, slit_step=slit_step
                                )
                                slitscounter += processed
                                if slitscounter >= next_update:
                                    self.print_progress(
                                        current=slitscounter, total=totalslits,
                                        color_code="33",
                                        suffix=f"frame={fidx} slits={slitscounter}/{totalslits}"
                                    )
                                    next_update += totalslits // 100

                        fps_done = np.where(renderfps_array == fps)[0][0] + 1
                        print(f"\n{fps_done}/{len(renderfps_array)} ({fps}fps) done in sep {s+1}/{separate_num}")

                # ===========================================
                # ソース: 動画（単一FPS）
                # ===========================================
                else:
                    wr_seg = wr_array[seq_read_s:seq_read_e]
                    frame_map = self._build_frame_map(wr_seg)
                    needed = set(frame_map.keys())

                    for fidx, fdata in self._iterate_frames(minz, maxz, needed):
                        processed = self._process_frame_v2(
                            fdata, fidx, frame_map, wr_seg, images,
                            slit_step=slit_step
                        )
                        slitscounter += processed
                        if slitscounter >= next_update:
                            self.print_progress(
                                current=slitscounter, total=totalslits,
                                color_code="33",
                                suffix=f"frame={fidx} slits={slitscounter}/{totalslits}"
                            )
                            next_update += totalslits // 100
                    print()

                Interval = time.time() - sstime

                # ===========================================
                # 出力フェーズ
                # ===========================================
                w = int(wr_array.shape[1]) if self.scan_direction % 2 == 1 else eff_width
                h = eff_height if self.scan_direction % 2 == 1 else int(wr_array.shape[1])

                if self._should_render_now(s, sep_start_num, sep_end_num,
                                            self.sepVideoOut, separate_num) and out_type != 0:
                    print(f"video-preference {Interval:.2f}sec")
                    out_path = self._determine_output_path(videostr, out_type, s,
                                                            sep_start_num, self.sepVideoOut)
                    self.out_videopath = out_path

                    if self._should_open_sink(s, sep_start_num, self.sepVideoOut, separate_num):
                        _sink_kind, _sink_obj = self._open_video_sink(out_path, w, h, self.outfps, out_type, use_yuv_native=use_yuv_native)

                    if use_yuv_native:
                        self._render_images_to_sink_yuv(img_y, img_cb, img_cr, _sink_kind, _sink_obj, out_type,
                                                        xy_trans_out, rotate_direction)
                    else:
                        self._render_images_to_sink(images, _sink_kind, _sink_obj, out_type,
                                                    xy_trans_out, rotate_direction)

                    if self._should_render_now(s, sep_start_num, sep_end_num,
                                                self.sepVideoOut, separate_num):
                        self._close_video_sink(_sink_kind, _sink_obj)

                elif out_type == 0:
                    # 画像シーケンス出力
                    print(f"image sequence output {Interval:.2f}sec")
                    for i in range(images.shape[0]):
                        img_name = f"img/{videostr}_{seq_read_s + i}p{self.imgtype}"
                        self._save_image_with_profile(img_name, images[i])
                        self.print_progress(current=i, total=images.shape[0], color_code="32")
                    print()

                elif self.sepVideoOut >= 2:
                    # ブロック途中 → レンダリングだけして sink は閉じない
                    if self._should_open_sink(s, sep_start_num, self.sepVideoOut, separate_num):
                        out_path = self._determine_output_path(videostr, out_type, s,
                                                                sep_start_num, self.sepVideoOut)
                        self.out_videopath = out_path
                        _sink_kind, _sink_obj = self._open_video_sink(out_path, w, h, self.outfps, out_type, use_yuv_native=use_yuv_native)
                    print(f"sep Rendering: {s}")
                    if use_yuv_native:
                        self._render_images_to_sink_yuv(img_y, img_cb, img_cr, _sink_kind, _sink_obj, out_type,
                                                        xy_trans_out, rotate_direction)
                    else:
                        self._render_images_to_sink(images, _sink_kind, _sink_obj, out_type,
                                                    xy_trans_out, rotate_direction)

                else:
                    # tmp保存
                    print(f"tmp file Writing {Interval:.2f}sec")
                    tmp_name = f"tmp/{videostr}_sep-{s}"
                    if use_yuv_native:
                        np.save(tmp_name + "_y", img_y)
                        np.save(tmp_name + "_cb", img_cb)
                        np.save(tmp_name + "_cr", img_cr)
                        print(f"tmp-npydata_saved(YUV) Y={img_y.shape} Cb={img_cb.shape} Cr={img_cr.shape}")
                    else:
                        np.save(tmp_name, images)
                        print(f"tmp-npydata_saved {images.shape}")

                # --- セグメント完了ログ ---
                if use_yuv_native:
                    del img_y, img_cb, img_cr
                else:
                    del images
                render_rate = round((time.time() - sstime) / max(1, seg_frames / 30), 2)
                print(f"done: {s+1}/{separate_num}")
                append_to_logfile(
                    f"done:{s+1}/{separate_num}({seq_read_s+render_clip_start}-{seq_read_e+render_clip_start}) "
                    f"{time.time()-sstime:.2f}sec slits:{totalslits} "
                    f"z:{minz}->{maxz} rate={render_rate} mem={psutil.virtual_memory().percent}%"
                )
                print(psutil.virtual_memory())
                print()

                if memory_report:
                    with open(f'memory_stats{self.ORG_NAME}-{s+1}-{separate_num}.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['render_time', 'images_size', 'total', 'available',
                                         'used', 'percent', 'active', 'inactive', 'wired', 'free'])
                        writer.writerows(memory_stats)
                gc.collect()

        # ============================================================
        # Phase 7: tmp → 最終出力（sepVideoOut==0 の場合）
        # ============================================================
        if self.cap is not None:
            self.cap.release()

        if self.sepVideoOut == 0 and out_type != 0 and separate_num != 1 and render_mode != 2:
            print("video-preference (final assembly)")
            gc.collect()
            print(psutil.virtual_memory())

            ext = '.mov' if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444,
                                          self.OUT_PRORES_422_SDR) else '.mp4'
            self.out_videopath = videostr + ext
            w = int(wr_array.shape[1]) if self.scan_direction % 2 == 1 else eff_width
            h = eff_height if self.scan_direction % 2 == 1 else int(wr_array.shape[1])
            _sink_kind, _sink_obj = self._open_video_sink(self.out_videopath, w, h, self.outfps, out_type, use_yuv_native=use_yuv_native)

            if tmp_para_images:
                r_start = 0 if render_mode == 0 else int(sep_start_num * wr_array.shape[0] / separate_num)
                r_end = wr_array.shape[0] if render_mode == 0 else int(sep_end_num * wr_array.shape[0] / separate_num)
                for i in range(r_start, r_end):
                    npy_path = f"tmp/{videostr}_{i}.npy"
                    last_img = np.load(npy_path)
                    if (last_img.dtype != np.uint16 and
                            out_type in (self.OUT_H265, self.OUT_PRORES_422, self.OUT_PRORES_4444)):
                        last_img = (last_img.astype(np.uint16) * 257)
                    self._write_video_frame(_sink_kind, _sink_obj, last_img, out_type,
                                            xy_trans_out, rotate_direction)
                    del last_img
                    if not tmp_save:
                        os.remove(npy_path)
                    self.print_progress(current=i, total=wr_array.shape[0], color_code="31")
            else:
                s_start = 0 if render_mode == 0 else sep_start_num
                s_end = separate_num if render_mode == 0 else sep_end_num
                total_count = 0
                for s in range(s_start, s_end):
                    gc.collect()
                    if render_mode != 3:
                        npy_path_base_s = f"tmp/{videostr}_sep-{s}"
                    else:
                        npy_path_base_s = f"tmp/{tmp_path_base}_sep-{s}"
                        print(f"read>> {npy_path_base_s}")

                    if use_yuv_native:
                        # YUV-native: 3プレーン別ファイルを読み込み
                        last_y  = np.load(npy_path_base_s + "_y.npy", mmap_mode='r')
                        last_cb = np.load(npy_path_base_s + "_cb.npy", mmap_mode='r')
                        last_cr = np.load(npy_path_base_s + "_cr.npy", mmap_mode='r')
                        for i in range(last_y.shape[0]):
                            frame_tuple = (last_y[i], last_cb[i], last_cr[i])
                            self._write_video_frame(_sink_kind, _sink_obj, frame_tuple, out_type,
                                                    xy_trans_out, rotate_direction, use_yuv_native=True)
                            self.print_progress(current=total_count + i, total=wr_array.shape[0],
                                                color_code="31")
                        total_count += last_y.shape[0]
                        del last_y, last_cb, last_cr
                        gc.collect()
                        if not tmp_save:
                            for suffix in ("_y.npy", "_cb.npy", "_cr.npy"):
                                p = npy_path_base_s + suffix
                                if os.path.exists(p):
                                    os.remove(p)
                    else:
                        # mmap_mode='r' でメモリに全展開せずディスクから直接読み出す
                        npy_path = npy_path_base_s + ".npy"
                        last_images = np.load(npy_path, mmap_mode='r')
                        for i in range(last_images.shape[0]):
                            self._write_video_frame(_sink_kind, _sink_obj, last_images[i], out_type,
                                                    xy_trans_out, rotate_direction)
                            self.print_progress(current=total_count + i, total=wr_array.shape[0],
                                                color_code="31")
                        total_count += last_images.shape[0]
                        del last_images
                        gc.collect()
                        if not tmp_save:
                            os.remove(npy_path)

            self._close_video_sink(_sink_kind, _sink_obj)
            if not tmp_save and os.path.isdir("tmp"):
                shutil.rmtree("tmp")

        total_time = round(time.time() - runFirstTime, 2)
        print(f"All Done {total_time}sec")
        append_to_logfile(f"All Done {total_time}sec")

    def new_transprocess(self,separate_num=None,sep_start_num=0,sep_end_num=None,out_type=1,xy_trans_out=False,render_mode=0,title_atr:str=None,title_replace:str=None,tmp_para_images=False,del_data=True,auto_memory_clear=False,memory_report=False,tmp_save = False,render_clip_start=0,render_clip_end=None,tmp_path_base = None,slit_step=1,scan_step=1,use_pyav=False):
        # slit_step: スリットレングス縮小 (1=等倍, 2=1/2, 3=1/3, 4=1/4)
        # scan_step: スキャンナムズ間引き (1=等倍, 2=1/2, 3=1/3, 4=1/4) — self.data非破壊
        if slit_step < 1: slit_step = 1
        if scan_step < 1: scan_step = 1
        eff_height = int(self.height) // slit_step
        eff_width  = int(self.width)  // slit_step
        if slit_step > 1 or scan_step > 1:
            print(f"slit_step={slit_step}, scan_step={scan_step}: output reduced (slit: {int(self.height)}→{eff_height} / {int(self.width)}→{eff_width}, scan_nums: {self.data.shape[1]}→{self.data.shape[1]//scan_step})")
        if self.input_pix_fmt == 'yuv420p10le' and out_type != 0:
            print("Error! The yuv420p10le video format is not supported for video output (out_type != 0). If the format is yuv444 or yuv422, it should be possible. Please convert the video data format and try loading it again.")
            append_to_logfile("Error! The yuv420p10le video format is not supported for video output.")
            return

        if self.cap is not None and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1 :
            print(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0.0)
            print("capに読み込ませた映像データが外れてる。transprcocessを連続的に実行する場合にcap.releaseされたままの状態になるため、再度読み込ませる必要がある。")
            self.cap = cv2.VideoCapture(self.VIDEO_PATH)

        if np.amin(self.data[:,:,-1])<0:
            print("z<0,error")
            append_to_logfile("z<0,error")
            return
        if np.amax(self.data[:,:,-1])>self.count:
            print("z>video_count,error,data_zmax:",np.amax(self.data[:,:,1])," > count:",self.count)
            append_to_logfile("z>video_count,error,data_zmax:"+str(np.amax(self.data[:,:,1]))+" > count:"+str(self.count))
            # return
        #self.outfpsはグローバルで定義

        if render_clip_end == None : render_clip_end = self.data.shape[0]
        render_range =render_clip_end-render_clip_start
        #HDR対応
        color_depth = 16 if getattr(self, "input_bit_depth", 8) > 8 else 8
        num_channels = 3
        file_size_bytes = (render_range*(self.slit_length//slit_step)*(self.data.shape[1]//scan_step) * color_depth * num_channels) / 8
        # Convert the file size to megabytes (1 MB = 1024 * 1024 bytes)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        if separate_num == None :
            separate_num = math.ceil(file_size_megabytes /((psutil.virtual_memory().available/(1024)**2)* (self.memory_percent/100)))
        print("separate_num=",separate_num,"active memory=",(psutil.virtual_memory().available/(1024)**2),"mb")
        if sep_end_num == None:sep_end_num = separate_num
        runFirstTime = time.time()
        XY_Name = "Y" if self.scan_direction%2 == 0 else "X"
        videostr = self.ORG_NAME+"_"+self.out_name_attr if render_mode == 0 or render_mode == 2 else self.ORG_NAME+"_"+self.out_name_attr+"("+str(sep_start_num)+"-"+str(sep_end_num)+"sep)"
        if render_mode == 3 and tmp_path_base == None: 
            if title_atr != None : 
                tmp_path_base = self.ORG_NAME+"_"+self.out_name_attr+title_atr
            else:
                tmp_path_base = self.ORG_NAME+"_"+self.out_name_attr 

        if title_replace is not None:
            videostr = title_replace
        else:
            if self.embedHistory_intoName == False :
                videostr = self.ORG_NAME+"_process"+str(self.log)
            if title_atr != None : videostr+=title_atr
        append_to_logfile("transprocess:"+str(separate_num)+"separate")
        append_to_logfile("active memory="+str(psutil.virtual_memory().available/(1024)**2)+"mb")
        append_to_logfile("memory_usage="+str(((file_size_megabytes / separate_num) / (psutil.virtual_memory().available/(1024)**2))*100)+"%")
        append_to_logfile(videostr)
        rotate_direction = False
        print("framecount=",render_range,"(",render_range/self.recfps,"sec)",XY_Name+"(out)=",self.data.shape[1],"refer(in"+XY_Name+"-out"+XY_Name+"-ZP)=",self.data.shape[2])
        _data_slice = self.data[render_clip_start:render_clip_end]
        print("z min-max =",np.amin(_data_slice[:,:,-1]),np.amax(_data_slice[:,:,-1]))
        print("space min-max =",np.amin(_data_slice[:,:,0]),np.amax(_data_slice[:,:,0]))
        append_to_logfile("outfps="+str(self.outfps))
        append_to_logfile("framecount="+str(render_range))
        append_to_logfile("z min-max ="+str(np.amin(_data_slice[:,:,-1]))+"-"+str(np.amax(_data_slice[:,:,-1])))
        append_to_logfile("space min-max ="+str(np.amin(_data_slice[:,:,0]))+"-"+str(np.amax(_data_slice[:,:,0])))
       
        #audioようにfloatで計算していたのをintへ戻す。この方法だと小数点以下は切り捨て
        # scan_step: axis=1を間引き（self.data非破壊）
        wr_array = self.data[render_clip_start:render_clip_end, ::scan_step, :].astype(np.int32)

        # 分割の切れ目の数値を格納する配列を初期化
        cut_points = []
        if render_clip_start != 0 :cut_points.append(render_clip_start)
        # 切れ目を計算して配列に追加
        for i in range(1,separate_num):
            cut = wr_array.shape[0] * i // separate_num  # 整数の除算を使用して切れ目を計算
            cut_points.append(cut+render_clip_start)
        if render_clip_end != self.data.shape[0] : cut_points.append(render_clip_end)
        self.maneuver_2dplot(x_positions=cut_points,s_frame=render_clip_start,e_frame=render_clip_end)

        if del_data :
            #メモリの最適化のため
            del self.data

        if out_type == 0:
            if os.path.isdir("img")==False:
                os.makedirs("img")
        if self.sepVideoOut != 1:  #sepVideoOut　はglobal関数。セパレートしない場合、rawでnpアレイファイルをテンポファイルとしてハードディスクに貯めておき、全てのアレイが準備できてからレンダリングする。そのためHD容量を100GBとか普通に食う。
            if os.path.isdir("tmp")==False:
                os.makedirs("tmp")
        
        # HDR 対応 / PyAV選択
        # is_morethan_8bit または use_pyav=True のとき PyAV を使う
        if self.is_morethan_8bit or use_pyav:
            if out_type == 1 and self.is_morethan_8bit:
                out_type = 2
            self.container = av.open(self.VIDEO_PATH)
            self.stream = self.container.streams.video[0]
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(self.VIDEO_PATH)
            self.container = None
        # HDR 対応 / PyAV選択ここまで

        # === YUV-native パイプライン判定 ===
        # HDR 10bit + ProRes + 同一トランスファー → RGB変換を回避し色精度を維持
        _hdr_mode_matches = (
            self.force_hdr_mode is None
            or (self.force_hdr_mode == "pq" and getattr(self, "input_transfer", "") == "smpte2084")
            or (self.force_hdr_mode == "hlg" and getattr(self, "input_transfer", "") == "arib-std-b67")
        )
        use_yuv_native = (
            self.is_morethan_8bit
            and out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444)
            and _hdr_mode_matches
            and out_type != 0
            and self.container is not None  # PyAVが必要
        )
        if use_yuv_native:
            print("🎨 YUV-native pipeline: ON (RGB変換スキップ、色精度最大)")
        else:
            print("🎨 YUV-native pipeline: OFF (RGB経由)")

        if render_mode < 3 : #3の場合は、tmpの書き出しのみを行う。
            for s in range(sep_start_num,sep_end_num):
                # メモリ統計を格納するためのリスト
                if memory_report : memory_stats = []
                if self.scan_direction % 2 == 1:
                    img_shape = (eff_height, int(wr_array.shape[1]), 3)
                else:
                    img_shape = (int(wr_array.shape[1]), eff_width, 3)
                base_images_frame_num = wr_array.shape[0]//separate_num
                images_frame_num = base_images_frame_num
                if s == sep_end_num-1:
                    images_frame_num +=  wr_array.shape[0] % separate_num
                
                print("img-declare",int(s*wr_array.shape[0]/separate_num),"->",int((s+1)*wr_array.shape[0]/separate_num))
                # HDR対応
                image_dtype = np.uint16 if self.is_morethan_8bit else np.uint8
                if use_yuv_native:
                    # YUV-native: Y/Cb/Crプレーン別バッファ（RGB変換なし）
                    n_slits = int(wr_array.shape[1])
                    if self.scan_direction % 2 == 1:
                        # vertical scan: (F, H, slits)
                        img_y  = np.zeros((images_frame_num, eff_height, n_slits), dtype=np.uint16)
                        img_cb = np.zeros((images_frame_num, eff_height, (n_slits + 1) // 2), dtype=np.uint16)
                        img_cr = np.zeros((images_frame_num, eff_height, (n_slits + 1) // 2), dtype=np.uint16)
                    else:
                        # horizontal scan: (F, slits, W)
                        img_y  = np.zeros((images_frame_num, n_slits, eff_width), dtype=np.uint16)
                        img_cb = np.zeros((images_frame_num, n_slits, eff_width // 2), dtype=np.uint16)
                        img_cr = np.zeros((images_frame_num, n_slits, eff_width // 2), dtype=np.uint16)
                    images = None  # RGBバッファは使わない
                    print(f"YUV buffers: Y={img_y.shape}, Cb={img_cb.shape}, Cr={img_cr.shape}")
                else:
                    images = np.zeros((images_frame_num, *img_shape), dtype=image_dtype)
                    img_y = img_cb = img_cr = None
                print("images.dtype =", image_dtype, img_shape if not use_yuv_native else "YUV-native")

                seq_read_s=int(s*base_images_frame_num)
                seq_read_e=int(seq_read_s+images_frame_num)#分割できなかった余剰フレームは最後の演算の時に加える

                #ビデオの設定
                interval_first = None
                minz = np.amin(wr_array[seq_read_s:seq_read_e,:,1])
                maxz = np.amax(wr_array[seq_read_s:seq_read_e,:,1])
                print("minz",minz)
                print("maxz",maxz)
                #depthが低いー高い
                # print("depth=", self.depth_to_sel_recfps[seq_read_s:seq_read_e])
                # 配列内の全ての値が同一かどうかをチェック
                if len(self.depth_to_sel_recfps[render_clip_start+seq_read_s:render_clip_start+seq_read_e]) > 0 :
                    fps_conversion = True
                else : fps_conversion = False
                print("fps_conversion=",fps_conversion)
                totalnum=maxz-minz
                totalslits=wr_array.shape[0]/separate_num*wr_array.shape[1]
                slitscounter=0
                next_update = totalslits // 100  # 1%ごと
                if totalnum == 0 : totalnum=1
                progresscale=self.progressbarsize/totalnum
                # progressAllScale=self.progressbarsize/(round(self.count))
                writingTotalNum=int(wr_array.shape[0]/separate_num)
                writingScale = self.progressbarsize/writingTotalNum
                if minz < 0 :
                    print("z<0,error")
                    return
                num = minz
                print("count",self.count)
                print("processing-Frames",writingTotalNum)
                # print("num=",num)
                if self.cap == None and self.container == None : # 画像ファイルのリストを取得、動画ではなくシーケンスの連番を、ソースとして使う
                    sstime = time.time()
                    image_files = [f for f in os.listdir(self.ORG_PATH +"/"+ self.ORG_NAME) if f.endswith(('.png', '.jpg', '.tif' ,'.jpeg', '.bmp','.npy'))]
                    image_files.sort()  # ソートして順番を保証
                    # S区間の配列をソートし、重複を省略した一時配列を作成
                    sorted_unique_array = np.sort(np.unique(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1]))
                    for num in sorted_unique_array : 
                        stime = time.time()
                        image_path = os.path.join(self.ORG_PATH +"/"+ self.ORG_NAME, image_files[num])
                        frame = np.load(image_path) if image_path.endswith('.npy') else cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                        if self.scan_direction%2 == 0 :
                            for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                                wr =np.array(wr_array[i])
                                indices = np.where(wr[:, 1] == num)
                                for p in indices[0]:
                                    images[i,p,:]= frame[wr[p,0],::slit_step]
                                slitscounter += len(indices[0])
                        else :
                            for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                                wr =np.array(wr_array[i])
                                indices = np.where(wr[:, 1] == num)
                                for p in indices[0]:
                                    images[i,:,p]=frame[::slit_step,wr[p,0]]
                                slitscounter += len(indices[0])
                        etime = time.time()
                        Interval = etime - stime
                        if interval_first == None :
                            interval_first = Interval
                        elif Interval < interval_first :
                            interval_first=Interval
                        #progressbar
                        bar= '■'*int((num-minz)*progresscale) + "."*int((totalnum-(num-minz))*progresscale)
                        print(f"\r\033[K[\033[33m{bar}\033[39m] frame{(num-minz)/totalnum*100:.02f}%({minz}>{num}>{maxz}) Slits{slitscounter/totalslits*100:.02f}%({slitscounter}/{int(totalslits)}) : {round(Interval,2)}({round(interval_first,2)})sec/f",end="")
                    print("\r")
                    lbstime = time.time()
                    Interval=lbstime-sstime
                    if out_type > 0 and (self.sepVideoOut == 1 or separate_num == 1) :
                        print("video-preference",round(Interval,2))
                        #HDR分岐対応
                        self.out_videopath = videostr + ('.mov' if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444, self.OUT_PRORES_422_SDR) else '.mp4')
                        w = int(wr_array.shape[1]) if self.scan_direction % 2 == 1 else eff_width
                        h = eff_height              if self.scan_direction % 2 == 1 else int(wr_array.shape[1])
                        _sink_kind, _sink_obj = self._open_video_sink(self.out_videopath, w, h, self.outfps, out_type, use_yuv_native=use_yuv_native)
                    else : print("file Writing",round(Interval,2))
                    if tmp_para_images :
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wstime = time.time()
                            if out_type == 0:
                                img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                                exec('cv2.imwrite(img_name,img%d)' %(i))
                            elif out_type != 0 :
                                if self.sepVideoOut == 1 or separate_num == 1:
                                    
                                    # HDR分岐
                                    frame_bgr = eval(f"img{i}")

                                    self._write_video_frame(
                                        _sink_kind, _sink_obj,
                                        frame_bgr,
                                        out_type,
                                        xy_trans_out=xy_trans_out,
                                        rotate_direction=rotate_direction
                                    )

                                else :
                                    tmp_name="tmp/"+self.ORG_NAME+"_"+str(i)
                                    exec('np.save(tmp_name,img%d)' %(i))
                            exec("del img%d"%(i))
                            gc.collect()
                            wetime = time.time()
                            knterval = round(wetime-wstime,2)
                            ci=i-int(s*wr_array.shape[0]/separate_num)
                            bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                            if self.sepVideoOut == 1 or separate_num == 1:
                                print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                            else :
                                print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                        if self.sepVideoOut == 1  or separate_num == 1 : 
                            self._close_video_sink(_sink_kind, _sink_obj)
                    else : 
                        if out_type > 0 and (self.sepVideoOut != 1 or separate_num != 1):
                            tmp_name="tmp/"+self.ORG_NAME+"_sep-"+str(s)
                            np.save(tmp_name,images)
                            print("npydata_saved")
                        else : 
                            for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                                wstime = time.time()
                                if out_type == 0:
                                    img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                                    cv2.imwrite(img_name,images[i])
                                if out_type > 0 :
                                    # HDR対応
                                    self._write_video_frame(_sink_kind, _sink_obj, images[i], out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction)
                                gc.collect()
                                wetime = time.time()
                                knterval = round(wetime-wstime,2)
                                ci=i-int(s*wr_array.shape[0]/separate_num)
                                bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                                if self.sepVideoOut == 1 or separate_num == 1:
                                    print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                                else :
                                    print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                            if self.sepVideoOut == 1  or separate_num == 1 : self._close_video_sink(_sink_kind, _sink_obj)
                
                    print("done:",s+1,"/",separate_num)
                    append_to_logfile("done:"+str(s+1)+"/"+str(separate_num)+" "+str(round(time.time()-sstime,2))+"sec wrote-Slits:"+str(int(totalslits))+"("+str(round(totalslits/(wr_array.shape[0]*wr_array.shape[1])*100,2))+"%) scan-Frames:"+str(minz)+"->"+str(maxz))
                    print("\r")
                    gc.collect()
                else : 
                    # fpsの違う映像ソースの切り替え
                    if fps_conversion:
                        print("複数fpsレンダリング")
                        renderfps_array=np.unique(self.depth_to_sel_recfps[render_clip_start+seq_read_s:render_clip_start+seq_read_e])
                        print(renderfps_array)
                        # 指定された値
                        for fps in renderfps_array:
                            # 配列内でvが続いているインデックスを見つける
                            print("sel fps=",fps)
                            # HDR 対応 / PyAV選択
                            if (self.is_morethan_8bit and out_type > 1) or use_pyav:
                                self.container = av.open(self.another_videos[self.some_recfps_array.index(fps)])
                                self.stream = self.container.streams.video[0]
                            else:
                                self.cap=cv2.VideoCapture(self.another_videos[self.some_recfps_array.index(fps)])
                                print("video reload",self.another_videos[self.some_recfps_array.index(fps)])
                                print(self.cap.get(cv2.CAP_PROP_FPS))
                                print(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            
                            indices_raw = np.where(self.depth_to_sel_recfps[render_clip_start+seq_read_s:render_clip_start+seq_read_e] == fps)[0]
                            diff = np.diff(indices_raw)
                            breaks = np.where(diff > 1)[0] + 1
                            
                            # Splitting the array into subarrays of consecutive numbers
                            some_indices = np.split(indices_raw,breaks)
                            print("some_indices=",len(some_indices))
                            for indices in some_indices :
                                range_start = indices[0]+seq_read_s
                                range_end = indices[-1]+seq_read_s+1
                                print("range_start,range_end=",range_start,range_end)
                                fps_scale=self.renderfps_scales[self.some_recfps_array.index(fps)]
                                print("fps_scale=",fps_scale)
                                #numの切り替え,FPSに応じて、フレーム数を掛け算しないといけない。
                                minz = np.amin(wr_array[range_start:range_end ,:,1])*fps_scale
                                maxz = np.amax(wr_array[range_start:range_end ,:,1])*fps_scale
                                print("minz",minz)
                                print("maxz",maxz)
                                totalnum=maxz-minz
                                totalslits=wr_array.shape[0]/separate_num*wr_array.shape[1]
                                slitscounter=0
                                if totalnum == 0 : totalnum=1
                                progresscale=self.progressbarsize/totalnum
                                writingTotalNum=int(wr_array.shape[0]/separate_num)
                                writingScale = self.progressbarsize/writingTotalNum
                                if minz < 0 :
                                    print("z<0,error")
                                    return
                                num = minz
                                #fpsの異なるソース映像の交換はない,単一の映像データから読み込む場合
                                if (not self.is_morethan_8bit and not use_pyav) or out_type == 1:
                                    print("OpenCv処理",inspect.currentframe().f_lineno)
                                    # === SDR (8bit) → OpenCV ===
                                    self.cap.set(cv2.CAP_PROP_POS_FRAMES,num)
                                    sstime = time.time()
                                    Interval=0
                                    while(self.cap.isOpened()):
                                        ret, frame = self.cap.read()
                                        if ret:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB統一
                                        reedfull_bool = num < (maxz+1)
                                        if ret == True and reedfull_bool :
                                            stime = time.time()
                                            if (num > minz or num == minz):
                                                if self.scan_direction%2 == 0 :
                                                    for i in range(range_start,range_end):
                                                        wr =np.array(wr_array[i])
                                                        indices = np.where(np.floor(wr[:, 1]*fps_scale) == np.floor(num))
                                                        for p in indices[0]:
                                                            images[i-seq_read_s,p,:] = frame[wr[p,0],::slit_step]
                                                        slitscounter += len(indices[0])
                                                else :
                                                    for i in range(range_start,range_end):
                                                        wr =np.array(wr_array[i])
                                                        indices = np.where(np.floor(wr[:, 1]*fps_scale) == np.floor(num))
                                                        # print(num,i,len(indices[0]))
                                                        for p in indices[0]:
                                                            # print(num,i,len(indices[0]))
                                                            images[i-seq_read_s,:,p] = frame[::slit_step,wr[p,0]]
                                                        slitscounter += len(indices[0])
                                            num += 1
                                            etime = time.time()
                                            newterval=(etime - stime)/(len(indices[0])+1)*1000
                                            if Interval*3 < newterval and Interval!=0 and auto_memory_clear : 
                                                gc.collect()
                                                print("memoryclear", newterval ,Interval)
                                            Interval = newterval
                                            bar= '■'*int((num-minz)*progresscale) + "."*int((totalnum-(num-minz))*progresscale)
                                            print(f"\r\033[K[\033[33m{bar}\033[39m] frame{(num-minz)/totalnum*100:.02f}%({minz}>{num}>{maxz}) Slits{slitscounter/totalslits*100:.03f}%({slitscounter}/{int(totalslits)}) : {round(Interval,2)}ms/slit",end="")
                                            if memory_report : 
                                                vmem = psutil.virtual_memory()
                                                memory_stats.append([
                                                    Interval,images.nbytes,vmem.total, vmem.available, vmem.used, vmem.percent,
                                                    vmem.active, vmem.inactive, vmem.wired, vmem.free
                                                ])
                                        else: break
                                    print("\r")
                                else : 
                                     # === PyAV読み込み（HDR or use_pyav） ===
                                    pyav_fmt = "rgb24" if (use_pyav and not self.is_morethan_8bit) else "rgb48le"
                                    print(f"PyAV処理 (format={pyav_fmt})")
                                    sstime = time.time()
                                    Interval=0
                                    # wr_arrays = wr_array[seq_read_s:seq_read_e]
                                    wr_arrays = wr_array[range_start:range_end]
                                    # wr_arrays = wr_array[range_start:range_end][indices[0]:indices[-1]]
                                    # # === ここで前処理（1回だけ） ===
                                    frame_to_indices = {} #dictionary型？
                                    for local_i, wr in enumerate(wr_arrays):  # seq_read_s を引いたインデックスで保存
                                        # print("事前にindiciesを作成中：",local_i)
                                        for p, z in enumerate(wr[:, 1]):
                                            frame_to_indices.setdefault(int(z*fps_scale), []).append((local_i, p))
                                    # needed_frames = set(wr_arrays[:, :, 1]*fps_scale.astype(int).ravel())
                                    needed_frames = set((wr_arrays[:, :, 1] * fps_scale).astype(int).ravel())
                                    print(images.shape,len(needed_frames))
                                    
                                    start_time_sec = minz / fps #秒数単位に戻す
                                    start_pts = int(start_time_sec / self.stream.time_base)
                                    self.container.seek(start_pts, stream=self.stream, any_frame=False, backward=True)
                                    for frame in self.container.decode(self.stream):
                                        frame_index = int(round(frame.pts * self.stream.time_base * fps))
                                        # print(frame_index)

                                        if frame_index < minz:
                                            print("ffmpeg continue-1",frame_index)
                                            continue
                                        if frame_index > maxz:
                                            print("ffmpeg break",frame_index)
                                            break
                                        if frame_index not in needed_frames:
                                            print("ffmpeg continue-2",frame_index,fps)                                            
                                            continue
                                        # print("ffmpeg streamline")

                                        img = frame.to_ndarray(format=pyav_fmt)

                                        processed = self._process_frame(
                                            img,
                                            frame_index,
                                            frame_to_indices,
                                            wr_arrays,
                                            images,
                                            gap=indices[0],
                                            slit_step=slit_step,
                                        )
                                        # print(processed)
                                        slitscounter += processed

                                        # --- 進捗表示 ---
                                        if slitscounter >= next_update:
                                            self.print_progress(
                                                current=slitscounter,
                                                total=totalslits,
                                                color_code="33",
                                                suffix=f"frame={frame_index} slits={slitscounter}/{totalslits}"
                                            )
                                            next_update += totalslits // 100

                                        # --- 早期終了 ---
                                        needed_frames.remove(frame_index)
                                        if not needed_frames:
                                            break
                                    lbstime = time.time()
                                    Interval=lbstime-sstime

                            print(np.where(renderfps_array == fps)[0][0]+1,"/",len(renderfps_array),"(",fps,"fps)"," done in sep",s+1,"/",separate_num)
                        # print("\r")
                        # lbstime = time.time()
                        # Interval=lbstime-sstime
                        # if self.sepVideoOut == 1 or separate_num == 1:
                        #     print("video-preference",round(Interval,2))
                        #     #HDR分岐
                        #     self.out_videopath = videostr + ('.mov' if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444, self.OUT_PRORES_422_SDR) else '.mp4')
                        #     w = int(wr_array.shape[1]) if self.scan_direction % 2 == 1 else int(self.width)
                        #     h = int(self.height)        if self.scan_direction % 2 == 1 else int(wr_array.shape[1])

                        #     _sink_kind, _sink_obj = self._open_video_sink(self.out_videopath, w, h, self.outfps, out_type)
                        # else : print("file Writing",round(Interval,2))
                        # if tmp_para_images :
                        #     for i in range(seq_read_s,seq_read_e):
                        #         wstime = time.time()
                        #         if out_type != 1:
                        #             img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                        #             cv2.imwrite(img_name,images[i-seq_read_s])
                        #         elif out_type != 0 :
                        #             if self.sepVideoOut == 1 or separate_num == 1:
                                        
                        #                 # HDR対応
                        #                 # 1) 区間内のローカルインデックスに正規化
                        #                 li = i - seq_read_s
                        #                 if li < 0 or li >= images.shape[0]:
                        #                     continue  # 念のためガード

                        #                 # 2) フレーム取得
                        #                 frame_bgr = images[li]

                        #                 # 3) 出力（SDRでもHDRでも同じ呼び出しに）
                        #                 self._write_video_frame(
                        #                     _sink_kind, _sink_obj,
                        #                     frame_bgr,
                        #                     out_type,
                        #                     xy_trans_out=xy_trans_out,
                        #                     rotate_direction=rotate_direction
                        #                 )

                        #             else :
                        #                 tmp_name="tmp/"+videostr+"_"+str(i)
                        #                 np.save(tmp_name,images[i-seq_read_s])
                        #         gc.collect()
                        #         wetime = time.time()
                        #         knterval = round(wetime-wstime,2)
                        #         ci=i-int(s*wr_array.shape[0]/separate_num)
                        #         bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                        #         if self.sepVideoOut == 1 or separate_num == 1:
                        #             print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                        #         else :
                        #             print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                        #     if self.sepVideoOut == 1  or separate_num == 1 : self._close_video_sink(_sink_kind, _sink_obj)
                        # else:
                        #     if out_type != 0 and self.sepVideoOut != 1 and separate_num != 1:
                        #         wstime = time.time()
                        #         tmp_name="tmp/"+videostr+"_sep-"+str(s)
                        #         np.save(tmp_name,images)
                        #         knterval = round(time.time()-wstime,2)
                        #         print("tmp-npydata_saved",images.shape,knterval,"sec")
                        #     else : 
                        #         print("direct render")
                        #         for i in range(images.shape[0]):
                        #             wstime = time.time()
                        #             if out_type != 1:
                        #                 img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                        #                 cv2.imwrite(img_name,images[i])
                        #             if out_type != 0 :
                        #                 #HDR対応
                        #                 self._write_video_frame(_sink_kind, _sink_obj, images[i], out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction)
                        #             gc.collect()
                        #             wetime = time.time()
                        #             knterval = round(wetime-wstime,2)
                        #             ci=i
                        #             bar= '■'*int(ci*writingScale)+ "."*int((images.shape[0]-ci)*writingScale)
                        #             if self.sepVideoOut == 1 or separate_num == 1:
                        #                 print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/images.shape[0]*100:.02f}% ({knterval:.02f})sec/f",end="")
                        #             else :
                        #                 print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/images.shape[0]*100:.02f}% ({knterval:.02f})sec/f",end="")
                        #         if self.sepVideoOut == 1  or separate_num == 1 : self._close_video_sink(_sink_kind, _sink_obj)
                        # del images
                        # print("done:",s+1,"/",separate_num)
                        # render_rate = round((time.time()-sstime)/ ((wr_array.shape[0]/separate_num) / 30),2)
                        # # 配列の要素を文字列に変換し、カンマで結合して1つの文字列にする
                        # renderfps_string = ', '.join(renderfps_array.astype(str))
                        # append_to_logfile("done:"+str(s+1)+"/"+str(separate_num)+"("+str(seq_read_s+render_clip_start)+"-"+str(seq_read_e+render_clip_start)+")"+" "+str(round(time.time()-sstime,2))+"sec wrote-Slits:"+str(int(totalslits))+"("+str(round(totalslits/(wr_array.shape[0]*wr_array.shape[1])*100,2))+"%) scan-Frames:"+str(minz)+"->"+str(maxz)+" render_rate="+str(render_rate)+" memory="+str(psutil.virtual_memory().percent)+"%"+" fps=[" + renderfps_string + "]")
                        # print(psutil.virtual_memory())
                        # print("\r")
                        # if memory_report :
                        #     with open('memory_stats'+self.ORG_NAME+'-'+str(s+1)+'-'+str(separate_num)+'.csv', 'w', newline='') as file:
                        #         writer = csv.writer(file)
                        #         # CSV ヘッダー
                        #         writer.writerow([
                        #             'render_time','images_size','total', 'available', 'used', 'percent', 
                        #             'active', 'inactive', 'wired', 'free'
                        #         ])
                        #         # データの書き込み
                        #         writer.writerows(memory_stats)
                        # gc.collect()
                    else :
                        #fpsの異なるソース映像の交換はない,単一の映像データから読み込む場合
                        if (not self.is_morethan_8bit and not use_pyav) or out_type == 1:
                            print("psの異なるソース映像の交換はない,単一の映像データから読み込む場合->OpenCv処理",inspect.currentframe().f_lineno)
                            # === SDR (8bit) → OpenCV ===
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, num)
                            print(self.cap,num,self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, num)
                            sstime = time.time()
                            Interval = 0
                            wr_arrays = wr_array[seq_read_s:seq_read_e]

                            # --- 事前マップ作成 ---
                            frame_to_indices = {}
                            for local_i, wr in enumerate(wr_arrays):
                                for p, z in enumerate(wr[:, 1]):
                                    frame_to_indices.setdefault(int(z), []).append((local_i, p))
                            needed_frames = set(frame_to_indices.keys())
                            # 開始位置へ1回だけシーク、以降は連続read（seekループ回避）
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(minz))
                            num = int(minz)
                            while num <= int(maxz):
                                ret, frame = self.cap.read()
                                if not ret:
                                    break
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB統一
                                if num in needed_frames:
                                    processed = self._process_frame(
                                        frame,
                                        num,
                                        frame_to_indices,
                                        wr_arrays,
                                        images,
                                        slit_step=slit_step,
                                    )
                                    slitscounter += processed
                                    needed_frames.discard(num)
                                    # --- 進捗表示 ---
                                    if slitscounter >= next_update:
                                        self.print_progress(
                                            current=slitscounter,
                                            total=totalslits,
                                            color_code="33",
                                            suffix=f"frame={num} slits={slitscounter}/{totalslits}"
                                        )
                                        next_update += totalslits // 100
                                    if not needed_frames:
                                        break
                                num += 1
                            #         lbstime = time.time()
                            #         Interval=lbstime-sstime
                            #         if out_type != 0 and self.sepVideoOut != 1 and separate_num == 1:
                            #             print("video-preference",round(Interval,2))
                            #             #HDR分岐
                            #             self.out_videopath = videostr + ('.mov' if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444, self.OUT_PRORES_422_SDR) else '.mp4')
                            #             w = int(wr_array.shape[1]) if self.scan_direction % 2 == 1 else int(self.width)
                            #             h = int(self.height)        if self.scan_direction % 2 == 1 else int(wr_array.shape[1])
                            #             _sink_kind, _sink_obj = self._open_video_sink(self.out_videopath, w, h, self.outfps, out_type)
                            #         else : print("file Writing",round(Interval,2))
                            #         if tmp_para_images :
                            #             for i in range(seq_read_s,seq_read_e):
                            #                 wstime = time.time()
                            #                 if out_type != 1:
                            #                     img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                            #                     cv2.imwrite(img_name,images[i-seq_read_s])
                            #                 elif out_type != 0 :
                            #                     if self.sepVideoOut == 1 or separate_num == 1:
                            #                         if xy_trans_out :
                            #                             if rotate_direction: video.write(images[i].transpose(1,0,2)[:,::-1])
                            #                             else : video.write(images[i].transpose(1,0,2)[::-1])
                            #                         else: video.write(images[i-seq_read_s])
                            #                     else :
                            #                         tmp_name="tmp/"+videostr+"_"+str(i)
                            #                         np.save(tmp_name,images[i-seq_read_s])
                            #                 wetime = time.time()
                            #                 knterval = round(wetime - wstime, 2)
                            #                 ci = i - int(s * wr_array.shape[0] / separate_num)
                            #                 color = "31" if (self.sepVideoOut == 1 or separate_num == 1) else "32"
                            #                 self.print_progress(
                            #                     current = ci,
                            #                     total   = writingTotalNum,
                            #                     color_code = color,
                            #                     suffix = f"({knterval:.02f})sec/f"
                            #                 )
                            #             if self.sepVideoOut == 1  or separate_num == 1 : self._close_video_sink(_sink_kind, _sink_obj)
                            #         else:
                            #             if out_type != 0 and self.sepVideoOut != 1 and separate_num != 1:
                            #                 wstime = time.time()
                            #                 tmp_name="tmp/"+videostr+"_sep-"+str(s)
                            #                 np.save(tmp_name,images)
                            #                 knterval = round(time.time()-wstime,2)
                            #                 print("tmp-npydata_saved",images.shape,knterval,"sec")
                            #             else : 
                            #                 print("direct render")
                            #                 for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            #                     wstime = time.time()
                            #                     if out_type == 0 :
                            #                         img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                            #                         cv2.imwrite(img_name,images[i])
                            #                     else :
                            #                         #HDR対応
                            #                         self._write_video_frame(_sink_kind, _sink_obj, images[i], out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction)
                            #                     wetime = time.time()
                            #                     knterval = round(wetime - wstime, 2)
                            #                     ci = i - int(s * wr_array.shape[0] / separate_num)
                            #                     color = "31" if (self.sepVideoOut == 1 or separate_num == 1) else "32"
                            #                     self.print_progress(
                            #                         current = ci,
                            #                         total   = writingTotalNum,
                            #                         color_code = color,
                            #                         suffix = f"({knterval:.02f})sec/f"
                            #                     )
                            #                 if out_type != 0 and self.sepVideoOut != 1 : 
                            #                     self._close_video_sink(_sink_kind, _sink_obj)
                                    # break
                            # del images
                        else:
                            # === PyAV読み込み（HDR 10bit以上 or use_pyav=True） ===
                            if use_yuv_native:
                                print(f"PyAV処理 (YUV-native yuv422p10le)", inspect.currentframe().f_lineno)
                            else:
                                pyav_fmt = "rgb24" if (use_pyav and not self.is_morethan_8bit) else "rgb48le"
                                print(f"PyAV処理 (format={pyav_fmt})", inspect.currentframe().f_lineno)
                            sstime = time.time()
                            Interval=0
                            wr_arrays = wr_array[seq_read_s:seq_read_e]
                            # # === ここで前処理（1回だけ） ===
                            frame_to_indices = {} #dictionary型？
                            for local_i, wr in enumerate(wr_arrays):  # seq_read_s を引いたインデックスで保存
                                # print("事前にindiciesを作成中：",local_i)
                                for p, z in enumerate(wr[:, 1]):
                                    frame_to_indices.setdefault(z, []).append((local_i, p))
                            needed_frames = set(wr_arrays[:, :, 1].astype(int).ravel())
                            if use_yuv_native:
                                print(f"Y={img_y.shape}, Cb={img_cb.shape}, Cr={img_cr.shape}")
                            else:
                                print(images.shape)

                            start_time_sec = num / self.inputmovfps # self.recfpsは使わずに、入力動画フォーマットに依存させる
                            start_pts = int(start_time_sec / self.stream.time_base)
                            self.container.seek(start_pts, stream=self.stream, any_frame=False, backward=True)
                            for frame in self.container.decode(self.stream):
                                frame_index = int(round(frame.pts * self.stream.time_base * self.inputmovfps))

                                if frame_index < num:
                                    continue
                                if frame_index > maxz:
                                    break
                                if frame_index not in needed_frames:
                                    continue

                                if use_yuv_native:
                                    # === YUV-native: プレーン直接取得（RGB変換なし） ===
                                    y_plane  = np.frombuffer(frame.planes[0], dtype=np.uint16).reshape(
                                        frame.planes[0].height, frame.planes[0].line_size // 2)[:, :frame.width]
                                    cb_plane = np.frombuffer(frame.planes[1], dtype=np.uint16).reshape(
                                        frame.planes[1].height, frame.planes[1].line_size // 2)[:, :frame.width // 2]
                                    cr_plane = np.frombuffer(frame.planes[2], dtype=np.uint16).reshape(
                                        frame.planes[2].height, frame.planes[2].line_size // 2)[:, :frame.width // 2]

                                    processed = self._process_frame_yuv(
                                        y_plane, cb_plane, cr_plane,
                                        frame_index,
                                        frame_to_indices,
                                        wr_arrays,
                                        img_y, img_cb, img_cr,
                                        slit_step=slit_step,
                                    )
                                else:
                                    img = frame.to_ndarray(format=pyav_fmt)
                                    processed = self._process_frame(
                                        img,
                                        frame_index,
                                        frame_to_indices,
                                        wr_arrays,
                                        images,
                                        slit_step=slit_step,
                                    )
                                slitscounter += processed

                                # --- 進捗表示 ---
                                if slitscounter >= next_update:
                                    self.print_progress(
                                        current=slitscounter,
                                        total=totalslits,
                                        color_code="33",
                                        suffix=f"frame={frame_index} slits={slitscounter}/{totalslits}"
                                    )
                                    next_update += totalslits // 100

                                # --- 早期終了 ---
                                needed_frames.remove(frame_index)
                                if not needed_frames:
                                    break
                            lbstime = time.time()
                            Interval=lbstime-sstime

                    '''
                    === Rendering condition ===
                    実行条件（out_type != 0 のときのみレンダリング）:
                    1. sepVideoOut == 1 の場合 → すべての s でレンダリング（s=0も含む）または、自動的にseparate_num=1が割り当てられたケースは、ダイレクトレンダリングする
                    2. それ以外の場合:
                        - s がブロック終端 (s % sepVideoOut == sepVideoOut - 1) のときにレンダリング
                        - または s が最後の処理 (s == sep_end_num - 1) のときも必ずレンダリング
                    
                    この条件により、ブロック区切り (0-4, 5-9, 10-14...) の最後で出力され、
                    sepVideoOut == 1 の場合は全フレームを出力。
                    カスタム設定の sep_end_num に到達した場合も確実に出力される。
                    '''
                    if (out_type == 0) or (
                        (self.sepVideoOut != 0 or separate_num == 1) and (
                            self.sepVideoOut == 1 or
                            (self.sepVideoOut > 1 and (s-sep_start_num) % self.sepVideoOut == self.sepVideoOut - 1) or
                            s == sep_end_num - 1
                        )
                    ):
                        print()
                        print("video-preference",round(Interval,2))
                        #HDR分岐
                        if self.sepVideoOut <= 0:
                            # 分割なし（通常出力）
                            print("direct video render")
                            self.out_videopath = (
                                videostr
                                + (".mov" if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444, self.OUT_PRORES_422_SDR) else ".mp4")
                            )

                        elif self.sepVideoOut == 1:
                            # 各フレームごとに出力 (例: _sep0, _sep1, _sep2 ...)
                            print(f"separate video render (frame {s})")
                            self.out_videopath = (
                                f"{videostr}_sep{s}"
                                + (".mov" if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444, self.OUT_PRORES_422_SDR) else ".mp4")
                            )

                        else:
                            # ブロック単位出力 (例: 0-4, 5-9, 10-14...)
                            if s % self.sepVideoOut == 0:
                                start_num = s
                                end_num = s - (self.sepVideoOut - 1)
                                print(f"Last separate video render (range {start_num}-{end_num})")
                                self.out_videopath = (
                                    f"{videostr}_sep{start_num}-{end_num}"
                                    + (".mov" if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444, self.OUT_PRORES_422_SDR) else ".mp4")
                                )

                        w = int(wr_array.shape[1]) if self.scan_direction % 2 == 1 else eff_width
                        h = eff_height              if self.scan_direction % 2 == 1 else int(wr_array.shape[1])
                        if (out_type != 0 and self.sepVideoOut <= 1) or (out_type != 0 and s % self.sepVideoOut == 0):
                            _sink_kind, _sink_obj = self._open_video_sink(self.out_videopath, w, h, self.outfps, out_type, use_yuv_native=use_yuv_native)
                        n_render_frames = img_y.shape[0] if use_yuv_native else images.shape[0]
                        for i in range(n_render_frames):
                            if self.sepVideoOut <= 1:
                                wstime = time.time()
                                if out_type == 0 :
                                    img_name="img/"+videostr+"_"+str(i)+"p"+self.imgtype
                                    self._save_image_with_profile(img_name, images[i])  # PNG, TIFならOK
                                else :
                                    #HDR対応
                                    if use_yuv_native:
                                        self._write_video_frame(_sink_kind, _sink_obj, (img_y[i], img_cb[i], img_cr[i]), out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction, use_yuv_native=True)
                                    else:
                                        self._write_video_frame(_sink_kind, _sink_obj, images[i], out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction)
                                gc.collect()
                                wetime = time.time()
                                knterval = round(wetime - wstime)
                                color = "31" if (self.sepVideoOut == 1 or separate_num == 1) else "32"
                                self.print_progress(
                                    current = i,
                                    total   = n_render_frames,
                                    color_code = color,
                                    suffix = f"({knterval:.02f})sec/f"
                                )
                            else:
                                # --- ブロック単位出力 (例: 0–4, 5–9, 10–14...) ---
                                wstime = time.time()
                                if out_type == 0 :
                                    img_name="img/"+videostr+"_"+str(i)+"p"+self.imgtype
                                    self._save_image_with_profile(img_name, images[i])  # PNG, TIFならOK
                                else :
                                    #HDR対応
                                    if use_yuv_native:
                                        self._write_video_frame(_sink_kind, _sink_obj, (img_y[i], img_cb[i], img_cr[i]), out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction, use_yuv_native=True)
                                    else:
                                        self._write_video_frame(_sink_kind, _sink_obj, images[i], out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction)
                                gc.collect()
                                wetime = time.time()
                                knterval = round(wetime - wstime)
                                color = "31"
                                self.print_progress(
                                    current = i,
                                    total   = n_render_frames,
                                    color_code = color,
                                    suffix = f"({knterval:.02f})sec/f"
                                )

                        if out_type != 0 :
                            # ブロック単位に限らず、必ず、クローズさせる
                            self._close_video_sink(_sink_kind, _sink_obj)
                        print()
                    else :
                        print()
                        if self.sepVideoOut >= 2:
                            if (s-sep_start_num) % self.sepVideoOut == 0 and out_type != 0:
                                print("sep Rendering First preference",round(Interval,2))
                                start_num = s
                                end_num = s+self.sepVideoOut-1
                                print(f"separate video render (range {start_num}-{end_num})")
                                append_to_logfile("sep Rendering First")
                                self.out_videopath = (
                                    f"{videostr}_sep{start_num}-{end_num}"
                                    + (".mov" if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444, self.OUT_PRORES_422_SDR) else ".mp4")
                                )
                                w = int(wr_array.shape[1]) if self.scan_direction % 2 == 1 else eff_width
                                h = eff_height              if self.scan_direction % 2 == 1 else int(wr_array.shape[1])
                                _sink_kind, _sink_obj = self._open_video_sink(self.out_videopath, w, h, self.outfps, out_type, use_yuv_native=use_yuv_native)
                            print("sep Rendering:",s)
                            n_render_frames_sep = img_y.shape[0] if use_yuv_native else images.shape[0]
                            for i in range(n_render_frames_sep):
                                # --- ブロック単位出力 (例: 0–4, 5–9, 10–14...) ---
                                wstime = time.time()
                                if out_type == 0 :
                                    img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype
                                    self._save_image_with_profile(img_name, images[i])  # PNG, TIFならHDR OK
                                else :
                                    #HDR対応
                                    if use_yuv_native:
                                        self._write_video_frame(_sink_kind, _sink_obj, (img_y[i], img_cb[i], img_cr[i]), out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction, use_yuv_native=True)
                                    else:
                                        self._write_video_frame(_sink_kind, _sink_obj, images[i], out_type, xy_trans_out=xy_trans_out, rotate_direction=rotate_direction)
                                gc.collect()
                                wetime = time.time()
                                knterval = round(wetime - wstime)
                                color = "31"
                                self.print_progress(
                                    current = i,
                                    total   = n_render_frames_sep,
                                    color_code = color,
                                    suffix = f"({knterval:.02f})sec/f"
                                )
                        else :
                            print("tmp file Writing",round(Interval,2))
                            wstime = time.time()
                            tmp_name="tmp/"+videostr+"_sep-"+str(s)
                            if use_yuv_native:
                                np.save(tmp_name + "_y", img_y)
                                np.save(tmp_name + "_cb", img_cb)
                                np.save(tmp_name + "_cr", img_cr)
                            else:
                                np.save(tmp_name,images)
                            knterval = round(time.time()-wstime,2)
                            if use_yuv_native:
                                print(f"tmp-npydata_saved(YUV) Y={img_y.shape} Cb={img_cb.shape} Cr={img_cr.shape}", knterval, "sec")
                            else:
                                print("tmp-npydata_saved",images.shape,knterval,"sec")
                                
                print("done:",s+1,"/",separate_num)
                render_rate = round((time.time()-sstime)/ ((wr_array.shape[0]/separate_num) / 30),2)
                append_to_logfile("done:"+str(s+1)+"/"+str(separate_num)+"("+str(seq_read_s+render_clip_start)+"-"+str(seq_read_e+render_clip_start)+")"+" "+str(round(time.time()-sstime,2))+"sec wrote-Slits:"+str(int(totalslits))+"("+str(round(totalslits/(wr_array.shape[0]*wr_array.shape[1])*100,2))+"%) scan-Frames:"+str(minz)+"->"+str(maxz)+" render_rate="+str(render_rate)+" memory="+str(psutil.virtual_memory().percent)+"%")
                print(psutil.virtual_memory())
                print("\r")
                if memory_report :
                    with open('memory_stats'+self.ORG_NAME+'-'+str(s+1)+'-'+str(separate_num)+'.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        # CSV ヘッダー
                        writer.writerow([
                            'render_time','images_size','total', 'available', 'used', 'percent', 
                            'active', 'inactive', 'wired', 'free'
                        ])
                        # データの書き込み
                        writer.writerows(memory_stats)
                gc.collect()
        if self.cap != None : self.cap.release()
        if self.sepVideoOut == 0 and out_type != 0 and separate_num != 1 and render_mode != 2:
            print("video-preference")
            gc.collect()
            print(psutil.virtual_memory())
            #HDR分岐
            self.out_videopath = videostr + ('.mov' if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444,  self.OUT_PRORES_422_SDR) else '.mp4')
            w = int(wr_array.shape[1]) if self.scan_direction % 2 == 1 else eff_width
            h = eff_height              if self.scan_direction % 2 == 1 else int(wr_array.shape[1])
            _sink_kind, _sink_obj = self._open_video_sink(self.out_videopath, w, h, self.outfps, out_type, use_yuv_native=use_yuv_native)
            if tmp_para_images :
                if render_mode == 0:
                    render_start=0
                    render_end=wr_array.shape[0]
                elif render_mode == 1:
                    render_start=int(sep_start_num*wr_array.shape[0]/separate_num)
                    render_end= int(sep_end_num*wr_array.shape[0]/separate_num)
                for i in range(render_start,render_end):
                    npy_path = f"tmp/{videostr}_{i}.npy"
                    last_img = np.load(npy_path)  # shape=(H,W,3)
                    # HDRコーデック系なら 8bit→16bit の持ち上げ（ファイルがuint8の場合の保険）
                    if (last_img.dtype != np.uint16 and
                        out_type in (self.OUT_H265, self.OUT_PRORES_422, self.OUT_PRORES_4444)):
                        last_img = (last_img.astype(np.uint16) * 257)
                    self._write_video_frame(
                        _sink_kind, _sink_obj,
                        last_img, out_type,
                        xy_trans_out, rotate_direction
                    )
                    del last_img
                    if not tmp_save:
                        os.remove(npy_path)
                    if tmp_save != True : os.remove("tmp/"+videostr+"_"+str(i)+".npy")
                    self.print_progress(
                        current = i,
                        total   = wr_array.shape[0],
                        color_code = "31"
                    )
            else:
                if render_mode == 0:
                    s_start = 0
                    s_end = separate_num
                elif render_mode == 1 or render_mode == 3:
                    s_start = sep_start_num
                    s_end = sep_end_num
                total_last_imagecounts=0
                for s in range(s_start,s_end):
                    gc.collect()
                    print(psutil.virtual_memory())
                    if render_mode != 3 :
                        npy_path_base_s = "tmp/"+videostr+"_sep-"+str(s)
                    else :
                        npy_path_base_s = "tmp/"+tmp_path_base+"_sep-"+str(s)
                        print("reed>>",npy_path_base_s)

                    if use_yuv_native:
                        # YUV-native: 3プレーン別ファイルを読み込み
                        last_y  = np.load(npy_path_base_s + "_y.npy", mmap_mode='r')
                        last_cb = np.load(npy_path_base_s + "_cb.npy", mmap_mode='r')
                        last_cr = np.load(npy_path_base_s + "_cr.npy", mmap_mode='r')
                        for i in range(last_y.shape[0]):
                            frame_tuple = (last_y[i], last_cb[i], last_cr[i])
                            self._write_video_frame(
                                _sink_kind, _sink_obj,
                                frame_tuple, out_type,
                                xy_trans_out, rotate_direction,
                                use_yuv_native=True
                            )
                            ci = i + total_last_imagecounts
                            self.print_progress(
                                current = ci,
                                total   = wr_array.shape[0],
                                color_code = "31"
                            )
                        total_last_imagecounts += last_y.shape[0]
                        del last_y, last_cb, last_cr
                        gc.collect()
                        if tmp_save != True:
                            for suffix in ("_y.npy", "_cb.npy", "_cr.npy"):
                                p = npy_path_base_s + suffix
                                if os.path.exists(p):
                                    os.remove(p)
                    else:
                        # mmap_mode='r' でメモリに全展開せずディスクから直接読み出す
                        npy_path = npy_path_base_s + ".npy"
                        last_images = np.load(npy_path, mmap_mode='r')
                        for i in range(last_images.shape[0]):
                            frame = last_images[i]
                            self._write_video_frame(
                                _sink_kind, _sink_obj,
                                frame, out_type,
                                xy_trans_out, rotate_direction
                            )
                            ci = i + total_last_imagecounts
                            self.print_progress(
                                current = ci,
                                total   = wr_array.shape[0],
                                color_code = "31"
                            )
                        total_last_imagecounts += last_images.shape[0]
                        del last_images
                        gc.collect()
                        if tmp_save != True : os.remove(npy_path)
            self._close_video_sink(_sink_kind, _sink_obj)
            if tmp_save != True : 
                shutil.rmtree("tmp")
        runOverTime = time.time()
        lnterval = round(runOverTime-runFirstTime,2)
        print("All Done",lnterval,"sec")
        append_to_logfile("All Done"+str(lnterval)+"sec")

    def pretransprocess(self,outnums=100,xy_trans_out=False):
        #self.outfpsはグローバルで定義
        out_type=1
        separate_num=1
        sep_start_num=0
        sep_end_num=1
        runFirstTime = time.time()
        XY_Name = "Y" if self.scan_direction%2 == 0 else "X"
        videostr = self.ORG_NAME+"_"+self.out_name_attr 
        if self.embedHistory_intoName == False :
            videostr = self.ORG_NAME+"_process"+str(self.log)
        rotate_direction = False
        if np.amin(self.data[:,:,-1])<0:
            print("z<0,error")
            return
        if np.amax(self.data[:,:,-1])>self.count:
            print("z>video_count,error",np.amax(self.data[:,:,-1]),self.count)
            return
        #audioようにfloatで計算していたのをintへ戻す。この方法だと小数点以下は切り捨て
        dim=int(self.data.shape[0] / outnums)
        beforefps=self.outfps
        self.outfps=10
        wr_array =  self.data[::dim].astype(np.int32)
        # self.data=wr_array
        s=0
        for i in range(0,int(wr_array.shape[0])):
            if self.scan_direction%2==1 :
                exec("img%d =  np.zeros((int(self.height),int(wr_array.shape[1]),3),np.uint8)" % (i))
            else :
                exec("img%d =  np.zeros((int(wr_array.shape[1]),int(self.width),3),np.uint8)" % (i))
        #ビデオの設定
        if self.sepVideoOut == 1 :
            print("video-preference")
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
            if self.scan_direction%2==1 :
                video = cv2.VideoWriter(videostr +'_pre.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if xy_trans_out == False else  cv2.VideoWriter(videostr +'sep_index='+str(s)+'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
            else :
                video = cv2.VideoWriter(videostr +'_pre.mp4', fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if xy_trans_out == False else  cv2.VideoWriter(videostr +'sep_index='+str(s)+'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
        interval_first = None
        block_num = 0
        print("minz=",np.amin(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1]))
        print("maxz",np.amax(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1]))
        minz = np.amin(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1])
        maxz = np.amax(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1])
        totalnum=maxz-minz
        totalslits=wr_array.shape[0]/separate_num*wr_array.shape[1]
        slitscounter=0
        if totalnum == 0 : totalnum=1
        progresscale=self.progressbarsize/totalnum
        progressAllScale=self.progressbarsize/(round(self.count))
        writingTotalNum=int(wr_array.shape[0]/separate_num)
        writingScale = self.progressbarsize/writingTotalNum
        if minz < 0 :
            print("z<0,error")
            return
        num = minz
        print("count",self.count)
        print("writingTotalNum",writingTotalNum)
        print("num=",num)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, num)
        sstime = time.time()
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB統一
            reedfull_bool = num < (maxz+1)
            if ret == True and reedfull_bool :
                stime = time.time()
                if (num > minz or num == minz):
                    if self.scan_direction%2 == 0 :
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wr =np.array(wr_array[i])
                            indices = np.where(wr[:, 1] == num)
                            for p in indices[0]:
                                exec("img%d[p,:] = frame[wr[p,0],:]" % (i))
                            slitscounter += len(indices[0])
                    else :
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wr =np.array(wr_array[i])
                            indices = np.where(wr[:, 1] == num)
                            for p in indices[0]:
                                exec("img%d[:,p] = frame[:,wr[p,0]]" % (i))
                            slitscounter += len(indices[0])
                num += 1
                etime = time.time()
                Interval = etime - stime
                if interval_first == None :
                    interval_first = Interval
                elif Interval < interval_first :
                    interval_first=Interval
                #progressbar
                bar= '■'*int((num-minz)*progresscale) + "."*int((totalnum-(num-minz))*progresscale)
                print(f"\r\033[K[\033[33m{bar}\033[39m] frame{(num-minz)/totalnum*100:.02f}%({minz}>{num}>{maxz}) Slits{slitscounter/totalslits*100:.02f}%({slitscounter}/{int(totalslits)}) : {round(Interval,2)}({round(interval_first,2)})sec/f",end="")
            else:
                print("\r")
                lbstime = time.time()
                Interval=lbstime-sstime
                if self.sepVideoOut == 1 or separate_num == 1:
                    print("video-preference",round(Interval,2))
                    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
                    self.out_videopath=videostr +'.mp4'
                    if self.scan_direction%2 == 1 :
                        video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if xy_trans_out == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
                    else :
                        video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if xy_trans_out == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
                else : print("file Writing",round(Interval,2))
                for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                    wstime = time.time()
                    if xy_trans_out :
                        if rotate_direction: exec('video.write(img%d.transpose(1,0,2)[:,::-1])' %(i))
                        else : exec('video.write(img%d.transpose(1,0,2)[::-1])' %(i))
                    else: exec('video.write(img%d)' %(i))
                    exec("del img%d"%(i))
                    gc.collect()
                    wetime = time.time()
                    knterval = round(wetime-wstime,2)
                    ci=i-int(s*wr_array.shape[0]/separate_num)
                    bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                    if self.sepVideoOut == 1 or separate_num == 1:
                        print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                    else :
                        print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                if self.sepVideoOut == 1  or separate_num == 1 : self._close_video_sink(_sink_kind, _sink_obj)
                break
        print("done:",s+1,"/",separate_num)
        print("\r")
        gc.collect()
        self.cap.release()
        runOverTime = time.time()
        lnterval = round(runOverTime-runFirstTime,2)
        print("All Done",lnterval,"sec")
        self.outfps=beforefps

    def overlay_tc_rate(self, output_suffix=None, divisions=5, hdr=None, text_nits=203,
                        text_only=False, font_scale=None, thickness=None,
                        font_scale_mult=1.0):
        """レンダリング済み映像にタイムコード・再生レートをオーバーレイして再書き出し。
        画面を横方向にdivisions等分し、各区画の中心スリットのTC・レートを表示する。
        hdr=None         : self.is_morethan_8bit で自動判定
        hdr=True         : H.265 10bit HDR (BT.2020 + PQ or HLG, 元の transfer を継承)
        hdr=False        : H.264 8bit SDR (従来動作)
        text_nits        : HDR時のオーバーレイ文字の明るさ (default 203nit ≈ HDR reference white)
        text_only        : True で元映像を捨て、テキストのみを透明背景の ProRes 4444 (with alpha)
                           で書き出す。AE 等での合成用。
        font_scale       : None で解像度に応じて自動 (FHD: 0.45, 4K: ≈0.9)
                           直接数値を渡すと font_scale_mult は無視される。
        thickness        : None で解像度に応じて自動 (FHD: 1, 4K: 2)
                           font_scale_mult が 1.0 以外の場合は thickness も同倍率で補正される
                           (ただし直接数値を渡した場合は尊重する)。
        font_scale_mult  : font_scale / thickness を自動算出する際の倍率 (default 1.0)。
                           例: 1.5 で auto値の1.5倍サイズ。
        output_suffix    : None で text_only なら "_tc_textonly"、そうでなければ "_tc"
        """
        import subprocess
        input_path = self.out_videopath

        if hdr is None:
            hdr = getattr(self, "is_morethan_8bit", False)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # --- 自動フォントスケール (4K で十分大きく見える様に調整) ---
        # FHD(1920) 基準 font_scale=0.45, thickness=1
        # 4K(3840) で font_scale≈0.9, thickness=2
        # font_scale_mult で更に倍率を掛けられる (例: 1.5 で 4K→≈1.35)
        auto_scale = max(1.0, w / 1920.0)
        if font_scale is None:
            font_scale = 0.45 * auto_scale * font_scale_mult
        if thickness is None:
            thickness = max(1, int(round(auto_scale * font_scale_mult)))
        # 文字下端からの余白 (従来 28 / 8 を解像度スケール、倍率も反映)
        # 3段構成: space(上) / tc(中) / rate(下)
        margin_scale = auto_scale * font_scale_mult
        rate_bottom_margin = max(3, int(round(8 * margin_scale)))   # 下端
        tc_bottom_margin = max(8, int(round(28 * margin_scale)))    # 中段 (rate の約20px上)
        space_bottom_margin = max(13, int(round(48 * margin_scale)))  # 上段 (tc の約20px上)

        # --- 出力パス ---
        if output_suffix is None:
            output_suffix = "_tc_textonly" if text_only else "_tc"
        if text_only:
            # ProRes 4444 (.mov) with alpha
            output_path = input_path.rsplit(".", 1)[0] + output_suffix + ".mov"
        else:
            output_path = input_path.rsplit(".", 1)[0] + output_suffix + ".mp4"

        slide_time = int(self.recfps / self.outfps)
        slit_count = self.data.shape[1]
        N = self.data.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX

        probe_slits = []
        for d in range(divisions):
            slit_idx = int((d + 0.5) * slit_count / divisions)
            pixel_x = int((d + 0.5) * w / divisions)
            probe_slits.append((slit_idx, pixel_x))

        # ---------------- text_only: テキストだけを透明背景で出力 ----------------
        if text_only:
            if hdr:
                in_trc = getattr(self, "input_transfer", None) or "smpte2084"
                if getattr(self, "force_hdr_mode", None) == "hlg" or in_trc == "arib-std-b67":
                    out_trc = "arib-std-b67"; trc_key = "hlg"
                    out_primaries = "bt2020"; out_colorspace = "bt2020nc"
                else:
                    out_trc = "smpte2084"; trc_key = "pq"
                    out_primaries = "bt2020"; out_colorspace = "bt2020nc"
                def encode_u16(r_lin, g_lin, b_lin):
                    lin = np.array([r_lin, g_lin, b_lin], dtype=np.float32) * text_nits
                    enc = oetf_from_scene_linear(lin, transfer=trc_key)
                    return tuple(int(np.clip(v, 0.0, 1.0) * 65535) for v in enc)
            else:
                out_trc = "bt709"
                out_primaries = "bt709"; out_colorspace = "bt709"
                def encode_u16(r_lin, g_lin, b_lin):
                    return (int(np.clip(r_lin, 0, 1) * 65535),
                            int(np.clip(g_lin, 0, 1) * 65535),
                            int(np.clip(b_lin, 0, 1) * 65535))
            white_u16 = encode_u16(1, 1, 1)

            # ProRes 4444 (yuva444p10le) writer. Input: rgba64le (16bit RGBA).
            out_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgba64le",
                "-s", f"{w}x{h}", "-r", str(fps),
                "-color_primaries", out_primaries,
                "-colorspace", out_colorspace,
                "-color_trc", out_trc,
                "-i", "-",
                "-c:v", "prores_ks", "-profile:v", "4444",
                "-pix_fmt", "yuva444p10le",
                "-vendor", "apl0",
                "-color_primaries", out_primaries,
                "-colorspace", out_colorspace,
                "-color_trc", out_trc,
                "-color_range", "tv",
                output_path,
            ]
            out_proc = subprocess.Popen(out_cmd, stdin=subprocess.PIPE)

            n_iter = min(N, total_in_frames) if total_in_frames > 0 else N
            for frame_idx in range(n_iter):
                rgb_u16 = np.zeros((h, w, 3), dtype=np.uint16)
                alpha_u8 = np.zeros((h, w), dtype=np.uint8)

                for slit_idx, px in probe_slits:
                    if frame_idx < N - 1:
                        rate = (self.data[frame_idx + 1, slit_idx, 1]
                                - self.data[frame_idx, slit_idx, 1]) / slide_time
                    else:
                        rate = 0.0
                    abs_rate = min(abs(rate), 1.0)
                    if rate > 0:
                        color = encode_u16(1, 1, 1 - abs_rate)
                    else:
                        color = encode_u16(1 - abs_rate, 1 - abs_rate * 0.5, 1)
                    z_raw = self.data[frame_idx, slit_idx, 1]
                    z_sec = int(z_raw // self.recfps)
                    z_frac = int(z_raw - z_sec * self.recfps)
                    tc_text = f"{z_sec}s{z_frac}f"
                    rate_text = f"{rate:+.2f}"
                    space_val = int(round(float(self.data[frame_idx, slit_idx, 0])))
                    space_text = f"{space_val}px"
                    # 空間座標のグラデーション色: 0→緑 (0,1,0), scan_nums→赤 (1,0,0)
                    t_sp = float(np.clip(space_val / max(1.0, float(self.scan_nums)), 0.0, 1.0))
                    space_color = encode_u16(t_sp, 1.0 - t_sp, 0.0)
                    (tw1, _), _ = cv2.getTextSize(tc_text, font, font_scale, thickness)
                    (tw2, _), _ = cv2.getTextSize(rate_text, font, font_scale, thickness)
                    (tw3, _), _ = cv2.getTextSize(space_text, font, font_scale, thickness)
                    x1 = max(0, min(px - tw1 // 2, w - tw1))
                    x2 = max(0, min(px - tw2 // 2, w - tw2))
                    x3 = max(0, min(px - tw3 // 2, w - tw3))
                    # RGB channels (color) and alpha mask are drawn separately so that
                    # anti-aliasing matches between them.
                    # 上: space (緑→赤グラデーション)
                    cv2.putText(rgb_u16, space_text, (x3, h - space_bottom_margin),
                                font, font_scale, space_color, thickness, cv2.LINE_AA)
                    cv2.putText(alpha_u8, space_text, (x3, h - space_bottom_margin),
                                font, font_scale, 255, thickness, cv2.LINE_AA)
                    # 中: tc (白)
                    cv2.putText(rgb_u16, tc_text, (x1, h - tc_bottom_margin),
                                font, font_scale, white_u16, thickness, cv2.LINE_AA)
                    cv2.putText(alpha_u8, tc_text, (x1, h - tc_bottom_margin),
                                font, font_scale, 255, thickness, cv2.LINE_AA)
                    # 下: rate (yellow/blue可変)
                    cv2.putText(rgb_u16, rate_text, (x2, h - rate_bottom_margin),
                                font, font_scale, color, thickness + 1, cv2.LINE_AA)
                    cv2.putText(alpha_u8, rate_text, (x2, h - rate_bottom_margin),
                                font, font_scale, 255, thickness + 1, cv2.LINE_AA)

                alpha_u16 = alpha_u8.astype(np.uint16) * 257  # 0..65535
                rgba = np.dstack([rgb_u16, alpha_u16])  # (h, w, 4) uint16, RGBA
                out_proc.stdin.write(rgba.tobytes())

            out_proc.stdin.close()
            out_proc.wait()
            print(f"TC overlay TEXT-ONLY ({divisions} div, {'HDR '+out_trc if hdr else 'SDR bt709'}, "
                  f"font_scale={font_scale:.2f}, thickness={thickness}): {output_path}")
            return

        if hdr:
            # transfer 判定 (PQ or HLG)
            in_trc = getattr(self, "input_transfer", None) or "smpte2084"
            if getattr(self, "force_hdr_mode", None) == "hlg" or in_trc == "arib-std-b67":
                out_trc = "arib-std-b67"
                trc_key = "hlg"
            else:
                out_trc = "smpte2084"
                trc_key = "pq"

            # 文字色を PQ/HLG エンコード済 uint16 (RGB順) に変換。
            # oetf_from_scene_linear は cd/m² (nit) を入力に取るので、text_nits を直接渡す。
            def encode_u16(r_lin, g_lin, b_lin):
                lin = np.array([r_lin, g_lin, b_lin], dtype=np.float32) * text_nits
                enc = oetf_from_scene_linear(lin, transfer=trc_key)
                return tuple(int(np.clip(v, 0.0, 1.0) * 65535) for v in enc)
            white_u16 = encode_u16(1, 1, 1)

            # ---- 入力: yuv420p10le → rgb48le で uint16 RGB を取得（PQ/HLGエンコードのまま保持） ----
            in_proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-i", input_path,
                 "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                stdout=subprocess.PIPE)

            # ---- 出力: libx265 10bit + HDR タグ (既存 _ENCODERS[OUT_H265] 準拠) ----
            enc = self._ENCODERS[self.OUT_H265]
            x265_params = enc["x265_params"]
            if out_trc == "arib-std-b67":
                x265_params = x265_params.replace("transfer=smpte2084", "transfer=arib-std-b67")
            out_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb48le",
                "-s", f"{w}x{h}", "-r", str(fps),
                "-color_primaries", "bt2020", "-colorspace", "bt2020nc", "-color_trc", out_trc,
                "-i", "-",
                "-c:v", enc["codec"], "-pix_fmt", enc["pix_fmt"],
                "-preset", "slow", "-crf", "14",
                "-x265-params", x265_params,
                "-color_primaries", "bt2020", "-colorspace", "bt2020nc", "-color_trc", out_trc,
                "-tag:v", enc["tagv"],
                output_path,
            ]
            out_proc = subprocess.Popen(out_cmd, stdin=subprocess.PIPE)

            fbytes = w * h * 3 * 2
            frame_idx = 0
            while True:
                buf = in_proc.stdout.read(fbytes)
                if len(buf) < fbytes:
                    break
                frame = np.frombuffer(buf, dtype=np.uint16).reshape(h, w, 3).copy()  # RGB
                if frame_idx < N:
                    for slit_idx, px in probe_slits:
                        if frame_idx < N - 1:
                            rate = (self.data[frame_idx + 1, slit_idx, 1]
                                    - self.data[frame_idx, slit_idx, 1]) / slide_time
                        else:
                            rate = 0.0
                        abs_rate = min(abs(rate), 1.0)
                        if rate > 0:
                            # yellow: R=1, G=1, B↓  (SDR pathと揃える)
                            color = encode_u16(1, 1, 1 - abs_rate)
                        else:
                            # blue: R↓, G=中間, B=1
                            color = encode_u16(1 - abs_rate, 1 - abs_rate * 0.5, 1)
                        z_raw = self.data[frame_idx, slit_idx, 1]
                        z_sec = int(z_raw // self.recfps)
                        z_frac = int(z_raw - z_sec * self.recfps)
                        tc_text = f"{z_sec}s{z_frac}f"
                        rate_text = f"{rate:+.2f}"
                        space_val = int(round(float(self.data[frame_idx, slit_idx, 0])))
                        space_text = f"{space_val}px"
                        # 空間座標のグラデーション色: 0→緑, scan_nums→赤
                        t_sp = float(np.clip(space_val / max(1.0, float(self.scan_nums)), 0.0, 1.0))
                        space_color = encode_u16(t_sp, 1.0 - t_sp, 0.0)
                        (tw1, _), _ = cv2.getTextSize(tc_text, font, font_scale, thickness)
                        (tw2, _), _ = cv2.getTextSize(rate_text, font, font_scale, thickness)
                        (tw3, _), _ = cv2.getTextSize(space_text, font, font_scale, thickness)
                        x1 = max(0, min(px - tw1 // 2, w - tw1))
                        x2 = max(0, min(px - tw2 // 2, w - tw2))
                        x3 = max(0, min(px - tw3 // 2, w - tw3))
                        # 上: space (緑→赤グラデーション)
                        cv2.putText(frame, space_text, (x3, h - space_bottom_margin),
                                    font, font_scale, space_color, thickness, cv2.LINE_AA)
                        # 中: tc (白)
                        cv2.putText(frame, tc_text, (x1, h - tc_bottom_margin),
                                    font, font_scale, white_u16, thickness, cv2.LINE_AA)
                        # 下: rate (yellow/blue)
                        cv2.putText(frame, rate_text, (x2, h - rate_bottom_margin),
                                    font, font_scale, color, thickness + 1, cv2.LINE_AA)
                out_proc.stdin.write(frame.tobytes())
                frame_idx += 1

            in_proc.stdout.close()
            in_proc.wait()
            out_proc.stdin.close()
            out_proc.wait()
            print(f"TC overlay HDR ({divisions} div, transfer={out_trc}): {output_path}")
            return

        # ---------------- SDR path (従来通り) ----------------
        cap = cv2.VideoCapture(input_path)
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s:v", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", output_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < N:
                for slit_idx, px in probe_slits:
                    if frame_idx < N - 1:
                        rate = (self.data[frame_idx + 1, slit_idx, 1]
                                - self.data[frame_idx, slit_idx, 1]) / slide_time
                    else:
                        rate = 0.0
                    abs_rate = min(abs(rate), 1.0)
                    if rate > 0:
                        color = (int(200 * (1 - abs_rate)),
                                 int(200 + 55 * abs_rate),
                                 int(200 + 55 * abs_rate))
                    else:
                        color = (int(200 + 55 * abs_rate),
                                 int(200 * (1 - abs_rate) + 100 * abs_rate),
                                 int(200 * (1 - abs_rate)))
                    z_raw = self.data[frame_idx, slit_idx, 1]
                    z_sec = int(z_raw // self.recfps)
                    z_frac = int(z_raw - z_sec * self.recfps)
                    tc_text = f"{z_sec}s{z_frac}f"
                    rate_text = f"{rate:+.2f}"
                    space_val = int(round(float(self.data[frame_idx, slit_idx, 0])))
                    space_text = f"{space_val}px"
                    # 空間座標のグラデーション色: 0→緑 (R=0, G=1, B=0), scan_nums→赤 (R=1, G=0, B=0)
                    t_sp = float(np.clip(space_val / max(1.0, float(self.scan_nums)), 0.0, 1.0))
                    # cv2 は BGR: (B, G, R)
                    space_color = (0, int(round((1.0 - t_sp) * 255)), int(round(t_sp * 255)))
                    (tw1, _), _ = cv2.getTextSize(tc_text, font, font_scale, thickness)
                    (tw2, _), _ = cv2.getTextSize(rate_text, font, font_scale, thickness)
                    (tw3, _), _ = cv2.getTextSize(space_text, font, font_scale, thickness)
                    x1 = max(0, min(px - tw1 // 2, w - tw1))
                    x2 = max(0, min(px - tw2 // 2, w - tw2))
                    x3 = max(0, min(px - tw3 // 2, w - tw3))
                    # 上: space (緑→赤グラデーション)
                    cv2.putText(frame, space_text, (x3, h - space_bottom_margin),
                                font, font_scale, space_color, thickness, cv2.LINE_AA)
                    # 中: tc (白)
                    cv2.putText(frame, tc_text, (x1, h - tc_bottom_margin),
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    # 下: rate (可変)
                    cv2.putText(frame, rate_text, (x2, h - rate_bottom_margin),
                                font, font_scale, color, thickness + 1, cv2.LINE_AA)
            proc.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
            frame_idx += 1

        cap.release()
        proc.stdin.close()
        proc.wait()
        print(f"TC overlay ({divisions} divisions): {output_path}")

    # レンダリング済みの映像ファイルと、軌道配列から、軌道の3D可視化アニメーションを作成する。
    # self.out_videopath と　self.dataが有効である必要がある
