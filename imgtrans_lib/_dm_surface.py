"""drawManeuver のサーフェスレンダリング (SurfaceRenderingMixin)

TimeFlowStudio (iOS) のサーフェスモードをデスクトップ側でフル解像度・
高ビット深度で再現するレンダラ。

スリット方式 (transprocess: 1次元スリット単位の走査) と異なり、
2次元 16bit グレースケールマップで「画素ごと」に参照時刻を指定する。
マップの XY は入力映像の XY と同じ座標系 (row 0 = 上端)。
解像度が異なる場合は入力映像サイズへ自動リサイズされる。

マップの解釈 (TimeWarpRenderer.swift のシェーダと同一の意味論):
- time : tf = frac(p + (g - 0.5) * amplitude)
         画素ごとの時間変位 (AE Time Displacement 相当)。
         amplitude は秒指定をクリップ実尺で正規化した値。
- rate : tf = frac(start_time + (p - anchor) * time_scale * rate(g))
         rate(g) = rate_black + (rate_white - rate_black) * g
         各画素が明度に応じたレートの独立した時計として回る。
         anchor (出力 0..1) の瞬間に全画素が start_time (入力 0..1) で一致。

  g  = マップ明度 0..1 / p = 出力位置 0..1 (frame / (N-1))
  tf = 入力の正規化時刻 → 参照フレーム z = tf * (count - 1)

レンダリング方式:
  出力フレームをセグメント分割し、セグメント内の全 (出力フレーム, 画素) の
  参照 z を基数ソートして「ソースフレーム → 書き込み先」の逆引きを構築。
  ソース映像を 1 パス順次デコードしながら散布書き込みする。
  ラップ (frac) により z が全域に散るため、原理的に
  「セグメント数 × ソース全域デコード」が読み込みコストの下限になる。
"""
import os
import gc
import math
import time

import psutil
import cv2
import numpy as np
import av

from ._utils import append_to_logfile


class SurfaceRenderingMixin:

    # TimeFlowStudio (GridAudioEngine) の既定グリッド = 7列 x 5行 = 35 ボイス
    SURFACE_AUDIO_COLS = 7
    SURFACE_AUDIO_ROWS = 5

    # ==================================================================
    # 音響用: グリッド縮約と時刻軌跡
    # (TimeWarpRenderer.reduceGrid / gridTimePositions と同じ意味論)
    # ==================================================================

    @staticmethod
    def _surface_grid_reduce(g2d, cols, rows):
        """マップをセル平均へ縮約する (アプリの reduceGrid と同じ帯分割)。

        明度→時刻の変換はどのモードでもアフィンなので、
        「セル平均明度を代入した時刻 = セル内の平均時刻」が厳密に成り立つ。

        returns: (rows*cols,) row-major のセル平均明度 0..1
        """
        h, w = g2d.shape
        out = np.empty((rows, cols), dtype=np.float64)
        for r in range(rows):
            y0 = r * h // rows
            y1 = min(h, max(y0 + 1, (r + 1) * h // rows))
            band = g2d[y0:y1]
            for c in range(cols):
                x0 = c * w // cols
                x1 = min(w, max(x0 + 1, (c + 1) * w // cols))
                out[r, c] = band[:, x0:x1].mean()
        return out.reshape(-1)

    def _surface_audio_state(self):
        """surfaceTransprocess が保存した音響用パラメータ (無ければ None)。"""
        return getattr(self, "surface_audio_state", None)

    def setup_surface_audio(self, surface_img_path,
                            interpretation="time", out_frame_num=None,
                            amplitude_sec=2.0,
                            rate_white=1.0, rate_black=-1.0,
                            anchor=0.5, start_time=0.5, wrap=True,
                            cols=None, rows=None,
                            render_clip_start=0, render_clip_end=None,
                            verbose=True):
        """サーフェス音響用のパラメータを準備する。

        surfaceTransprocess() が最後に自動で呼ぶため、通常は
        映像 → audio_render/audio_video_out と続けるだけでよい。
        映像を作らず音声だけ作りたい場合に直接呼ぶ。
        引数の意味は surfaceTransprocess と同じものを指定すること。

        グリッド縮約はアプリ (TimeWarpRenderer.reduceGrid) と同じく
        マップ本来の解像度に対して行う。
        """
        cols = int(cols or self.SURFACE_AUDIO_COLS)
        rows = int(rows or self.SURFACE_AUDIO_ROWS)

        img = cv2.imread(surface_img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"surface map を読み込めません: {surface_img_path}")
        if img.ndim == 3:
            img = img[:, :, 0]
        g2d = img.astype(np.float64) / (255.0 if img.dtype == np.uint8 else 65535.0)

        clip_dur = int(self.count) / self.recfps
        if out_frame_num is None:
            out_frame_num = int(round((30 if interpretation == "rate" else clip_dur)
                                      * self.outfps))
        out_frame_num = max(1, int(out_frame_num))
        if render_clip_end is None:
            render_clip_end = out_frame_num

        self.surface_audio_state = {
            "grid_g": self._surface_grid_reduce(g2d, cols, rows),
            "cols": cols, "rows": rows,
            "interpretation": interpretation,
            "amp_norm": amplitude_sec / clip_dur,
            "time_scale": (out_frame_num / self.outfps) / clip_dur,
            "rate_white": rate_white, "rate_black": rate_black,
            "anchor": anchor, "start_time": start_time,
            "wrap": wrap,
            "out_frame_num": out_frame_num,
            "p_denom": max(1, out_frame_num - 1),
            "render_clip_start": int(render_clip_start),
            "render_clip_end": int(render_clip_end),
            "map_path": surface_img_path,
        }
        if verbose:
            print(f"setup_surface_audio: {cols}x{rows}={cols*rows}voices "
                  f"[{interpretation}] frames="
                  f"{render_clip_end - render_clip_start}")
        return self.surface_audio_state

    def surface_grid_time_positions(self):
        """グリッド各セルの正規化入力時刻 (F, cols*rows) を返す。

        アプリの `gridTimePositions(progress:)` を全出力フレームぶん
        まとめて評価したもの。F はレンダリングした出力フレーム数
        (render_clip_start..render_clip_end) と一致する。
        """
        st = self._surface_audio_state()
        if st is None:
            raise ValueError(
                "サーフェスの音響パラメータがありません。"
                "先に surfaceTransprocess() を実行するか、"
                "audio_render(surface_map=...) でマップを指定してください。")
        g = st["grid_g"]                                   # (V,)
        k = np.arange(st["render_clip_start"], st["render_clip_end"],
                      dtype=np.float64)
        p = (k / st["p_denom"])[:, None]                   # (F, 1)

        if st["interpretation"] == "time":
            tf = p + (g - 0.5) * st["amp_norm"]
        else:
            rate = st["rate_black"] + (st["rate_white"] - st["rate_black"]) * g
            tf = st["start_time"] + (p - st["anchor"]) * st["time_scale"] * rate

        if st["wrap"]:
            tf = tf - np.floor(tf)
        else:
            tf = np.clip(tf, 0.0, 1.0)
        return tf

    def _load_surface_map(self, surface_img_path, eff_h, eff_w, map_interp="linear"):
        """16bit グレースケールマップを読み込み g (float32, 0..1) を返す。

        - カラー画像は先頭チャンネルを使用
        - 8bit 画像は /255, 16bit は /65535 で正規化
        - (eff_h, eff_w) と異なる場合はリサイズ

        リサイズはアスペクト比を保存せず出力解像度へ引き伸ばす。これは
        TimeFlowStudio のシェーダが正規化 UV でマップを画面全体に張る
        (`timeMap.sample(smp, in.uv)`) のと同じ挙動で、正方形マップを
        16:9 の映像に適用した場合もアプリの見た目と一致する。

        map_interp:
          "linear"  GPU のリニアサンプラと同じ = アプリのプレビュー忠実 (既定)
          "cubic" / "lanczos"
                    大きな拡大 (iPhone の低解像度マップ → 4K) で
                    時間場のファセット (階段状の折れ) を抑えて滑らかにする
        """
        img = cv2.imread(surface_img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"surface map を読み込めません: {surface_img_path}")
        if img.ndim == 3:
            img = img[:, :, 0]
        maxval = 255.0 if img.dtype == np.uint8 else 65535.0
        g = img.astype(np.float32) / maxval
        if g.shape != (eff_h, eff_w):
            flags = {"linear": cv2.INTER_LINEAR,
                     "cubic": cv2.INTER_CUBIC,
                     "lanczos": cv2.INTER_LANCZOS4}
            if map_interp not in flags:
                raise ValueError(f"map_interp は {list(flags)} のいずれか: {map_interp}")
            print(f"surface map resize [{map_interp}]: "
                  f"{g.shape[1]}x{g.shape[0]} -> {eff_w}x{eff_h}")
            g = cv2.resize(g, (eff_w, eff_h), interpolation=flags[map_interp])
            # cubic/lanczos はオーバーシュートするため 0..1 に戻す
            np.clip(g, 0.0, 1.0, out=g)
        return g

    def surfaceTransprocess(self, surface_img_path,
                            interpretation="time",
                            out_frame_num=None,
                            amplitude_sec=2.0,
                            rate_white=1.0, rate_black=-1.0,
                            anchor=0.5, start_time=0.5,
                            wrap=True,
                            time_interp="nearest",
                            map_interp="linear",
                            separate_num=None,
                            sep_start_num=0, sep_end_num=None,
                            out_type=1,
                            title_atr: str = None,
                            pixel_step=1,
                            use_pyav=False,
                            render_clip_start=0, render_clip_end=None):
        """2次元 16bit グレースケールマップで画素ごとに時間を指定してレンダリングする。

        Parameters
        ----------
        surface_img_path : str
            16bit グレースケールマップ。XY は入力映像と同じ座標系
            (異なる解像度は自動リサイズ)。
        interpretation : "time" | "rate"
            マップ明度の解釈 (TimeFlowStudio のサーフェスモードと同一)。
        out_frame_num : int, optional
            出力フレーム数。None なら
            time = 入力実尺を outfps 換算 / rate = 30 秒 × outfps
            (TimeFlowStudio のデフォルトループ長と同じ)。
        amplitude_sec : float
            time 解釈: 時間変位量 (秒)。AE Time Displacement と同じ単位。
        rate_white, rate_black : float
            rate 解釈: 白 (g=1) / 黒 (g=0) の再生レート。既定 +1 / -1。
        anchor : float
            rate 解釈: 全画素の時刻が一致する出力位置 (0=始点, 0.5=中間, 1=終点)。
        start_time : float
            rate 解釈: 同期の瞬間に映る入力クリップ上の位置 0..1。
        wrap : bool
            時間の範囲外参照を True=ラップ (アプリと同じ frac) / False=クランプ。
        time_interp : "nearest" | "linear"
            フレーム間補間。nearest が高速。linear はアプリのプレビューと同じ
            2 フレーム加重合成 (メモリ・時間コスト増)。
        map_interp : "linear" | "cubic" | "lanczos"
            マップを出力解像度へ拡大する際の補間。linear がアプリ忠実 (既定)。
            低解像度マップを 4K へ大きく拡大する場合は cubic が滑らか。
        separate_num : int, optional
            メモリ分割数。None なら memory_percent から自動計算。
        sep_start_num, sep_end_num : int
            処理するセグメント範囲 (部分レンダリング用)。
        out_type : int
            transprocess と同じ出力タイプ (OUT_H264 / OUT_H265 / OUT_PRORES_* / 0=連番)。
        pixel_step : int
            出力縮小 (1=等倍, 2=1/2, ...)。時間実測・プレビュー用。
        render_clip_start, render_clip_end : int
            出力フレームの部分範囲 (出力フレーム番号)。
        """
        # ============================================================
        # Phase 1: パラメータ・入力検証
        # ============================================================
        assert interpretation in ("time", "rate"), "interpretation は 'time' か 'rate'"
        assert time_interp in ("nearest", "linear"), "time_interp は 'nearest' か 'linear'"
        pixel_step = max(1, int(pixel_step))

        count = int(self.count)
        H, W = int(self.height), int(self.width)
        # スライス互換の実効サイズ (frame[::step] は ceil(H/step) 行になる)
        eff_h = (H + pixel_step - 1) // pixel_step
        eff_w = (W + pixel_step - 1) // pixel_step
        n_pix = eff_h * eff_w

        if self.cap is not None and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
            print("capの映像データが外れている。再読み込みします。")
            self.cap = cv2.VideoCapture(self.VIDEO_PATH)

        # ============================================================
        # Phase 2: マップ読み込みと出力長の決定
        # ============================================================
        g_2d = self._load_surface_map(surface_img_path, eff_h, eff_w,
                                      map_interp=map_interp)
        g_map = g_2d.reshape(-1)

        clip_dur = count / self.recfps
        if out_frame_num is None:
            if interpretation == "rate":
                out_frame_num = int(round(30 * self.outfps))
            else:
                out_frame_num = int(round(clip_dur * self.outfps))
        out_frame_num = max(1, int(out_frame_num))

        if render_clip_end is None:
            render_clip_end = out_frame_num
        render_range = render_clip_end - render_clip_start

        # 画素ごとの時刻計算に使う一次係数を先に作る:
        #   time : tf_i = p_i + off        (off = (g-0.5)*amp_norm)
        #   rate : tf_i = base + p_i*slope (base = start_time - anchor*slope_g,
        #                                   slope = time_scale*rate(g))
        if interpretation == "time":
            amp_norm = amplitude_sec / clip_dur
            tf_off = (g_map - 0.5) * amp_norm            # (n_pix,)
            tf_slope = None
            param_str = f"amp={amplitude_sec}s({amp_norm:.4f})"
        else:
            out_dur = out_frame_num / self.outfps
            time_scale = out_dur / clip_dur
            rates = rate_black + (rate_white - rate_black) * g_map
            tf_slope = time_scale * rates                # (n_pix,)
            tf_off = start_time - anchor * tf_slope      # (n_pix,)
            param_str = (f"rate={rate_black}..{rate_white} anchor={anchor} "
                         f"start={start_time} tscale={time_scale:.4f}")

        # 音響 (audio_render / audio_video_out) 用のパラメータを保存する。
        # 映像と同じ設定・同じ出力フレーム範囲を使うため、絵と音が必ず同期する。
        self.setup_surface_audio(
            surface_img_path, interpretation=interpretation,
            out_frame_num=out_frame_num, amplitude_sec=amplitude_sec,
            rate_white=rate_white, rate_black=rate_black,
            anchor=anchor, start_time=start_time, wrap=wrap,
            render_clip_start=render_clip_start, render_clip_end=render_clip_end,
            verbose=False)
        del g_map, g_2d

        # ============================================================
        # Phase 3: メモリ計算・分割数決定
        # ============================================================
        color_bytes = 2 if getattr(self, "input_bit_depth", 8) > 8 else 1
        # 出力フレーム 1 枚あたりの所要バイト数 (images + z/order の一時配列)
        if time_interp == "nearest":
            per_frame = n_pix * (3 * color_bytes + 4 + 8)       # img + z(i32) + order(i64)
        else:
            # linear: float32 蓄積 + (z0,z1,w) + 2 倍のエントリの order
            per_frame = n_pix * (3 * 4 + 3 * color_bytes + 12 + 2 * 8)
        total_mb = render_range * per_frame / (1024 * 1024)

        if separate_num is None:
            avail_mb = psutil.virtual_memory().available / (1024 ** 2)
            separate_num = max(1, math.ceil(total_mb / (avail_mb * (self.memory_percent / 100))))
        if sep_end_num is None:
            sep_end_num = separate_num

        print(f"surfaceTransprocess [{interpretation}/{time_interp}] {param_str}")
        print(f"out={eff_w}x{eff_h} frames={render_range} (src count={count}) "
              f"wrap={wrap} pixel_step={pixel_step}")
        print(f"separate_num={separate_num}, "
              f"active memory={psutil.virtual_memory().available/(1024**2):.0f}mb, "
              f"est. workset={total_mb/max(1,separate_num):.0f}mb/seg")
        append_to_logfile(f"surfaceTransprocess:{interpretation}/{time_interp} {param_str}")
        append_to_logfile(f"surface_map={os.path.basename(surface_img_path)}")
        append_to_logfile(f"out={eff_w}x{eff_h} frames={render_range} separate={separate_num}")

        # ============================================================
        # Phase 4: 出力名・ソースオープン
        # ============================================================
        runFirstTime = time.time()
        videostr = f"{self.ORG_NAME}_{self.out_name_attr}_surface-{interpretation}"
        if not self.embedHistory_intoName:
            videostr = f"{self.ORG_NAME}_process{self.log}_surface-{interpretation}"
        if title_atr:
            videostr += title_atr
        if sep_start_num != 0 or sep_end_num != separate_num:
            videostr += f"({sep_start_num}-{sep_end_num}sep)"
        append_to_logfile(videostr)

        is_seq_source = self.cap is None and getattr(self, "container", None) is None \
            and os.path.isdir(f"{self.ORG_PATH}/{self.ORG_NAME}")
        if not is_seq_source:
            if self.is_morethan_8bit or use_pyav:
                if out_type == 1 and self.is_morethan_8bit:
                    out_type = 2
                self.container = av.open(self.VIDEO_PATH)
                self.stream = self.container.streams.video[0]
                self.cap = None
            else:
                self.cap = cv2.VideoCapture(self.VIDEO_PATH)
                self.container = None
            self._active_rotation = getattr(self, "input_rotation", 0)

        image_dtype = np.uint16 if self.is_morethan_8bit else np.uint8
        if out_type == 0 and not os.path.isdir("img"):
            os.makedirs("img")

        # 単一シンクへストリーミング (tmp 退避なし)
        _sink_kind = _sink_obj = None
        if out_type != 0:
            ext = '.mov' if out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444,
                                         self.OUT_PRORES_422_SDR) else '.mp4'
            self.out_videopath = videostr + ext
            _sink_kind, _sink_obj = self._open_video_sink(
                self.out_videopath, eff_w, eff_h, self.outfps, out_type)

        # 進捗の分母 (nearest: 1 画素 1 書き込み / linear: 2 書き込み)
        entries_per_frame = n_pix * (1 if time_interp == "nearest" else 2)
        p_denom = max(1, out_frame_num - 1)   # p = i/(N-1): アプリの書き出しと同一
        img_written = 0
        bench = {"zmap": 0.0, "sort": 0.0, "decode_scatter": 0.0, "encode": 0.0}

        # ============================================================
        # Phase 5: メインループ (セグメント単位)
        # ============================================================
        for s in range(sep_start_num, sep_end_num):
            base_frames = render_range // separate_num
            seg_frames = base_frames + (render_range % separate_num
                                        if s == separate_num - 1 else 0)
            seg_start = render_clip_start + s * base_frames
            seg_end = seg_start + seg_frames
            print(f"seg {s+1}/{separate_num}: out-frame {seg_start} -> {seg_end}")

            # --- 5a: 画素ごとの参照フレーム z を構築 ---
            sstime = time.time()
            p_vals = np.arange(seg_start, seg_end, dtype=np.float32) / p_denom

            # numpy の argsort(kind='stable') は 16bit 以下の整数のみ基数ソート
            # (O(N)) になる。count が uint16 に収まる場合はキャストして
            # ソートを一桁高速化する (実測 int32 比で約 12 倍)。
            z_dtype = np.uint16 if count <= 65536 else np.int32

            if time_interp == "nearest":
                Z = np.empty((seg_frames, n_pix), dtype=z_dtype)
                for k in range(seg_frames):
                    if interpretation == "time":
                        tf = p_vals[k] + tf_off
                    else:
                        tf = tf_off + p_vals[k] * tf_slope
                    if wrap:
                        tf -= np.floor(tf)
                    else:
                        np.clip(tf, 0.0, 1.0, out=tf)
                    Z[k] = np.rint(tf * (count - 1)).astype(z_dtype)
                weights = None
            else:
                # linear: z0/z1 の 2 エントリと重みを作る
                Z = np.empty((2, seg_frames, n_pix), dtype=z_dtype)
                weights = np.empty((2, seg_frames, n_pix), dtype=np.float32)
                for k in range(seg_frames):
                    if interpretation == "time":
                        tf = p_vals[k] + tf_off
                    else:
                        tf = tf_off + p_vals[k] * tf_slope
                    if wrap:
                        tf -= np.floor(tf)
                    else:
                        np.clip(tf, 0.0, 1.0, out=tf)
                    t = tf * (count - 1)
                    z0 = np.floor(t)
                    w = t - z0
                    z0 = z0.astype(z_dtype)
                    z1 = np.minimum(z0.astype(np.int64) + 1, count - 1).astype(z_dtype)
                    Z[0, k], Z[1, k] = z0, z1
                    weights[0, k], weights[1, k] = 1.0 - w, w
                Z = Z.reshape(2 * seg_frames, n_pix)
                weights = weights.reshape(-1)
            bench["zmap"] += time.time() - sstime

            # --- 5b: 基数ソートで「ソースフレーム → 書き込み先」の逆引きを構築 ---
            sstime = time.time()
            z_flat = Z.reshape(-1)
            order = np.argsort(z_flat, kind='stable')   # int は基数ソート O(N)
            z_sorted = z_flat[order]
            uniq, starts = np.unique(z_sorted, return_index=True)
            ends = np.append(starts[1:], z_sorted.size)
            del z_flat, z_sorted, Z
            minz, maxz = int(uniq[0]), int(uniq[-1])
            if minz < 0 or maxz > count - 1:
                print(f"z 範囲エラー: {minz}..{maxz} (count={count})")
                return
            bench["sort"] += time.time() - sstime
            print(f"z range = {minz}..{maxz}, needed src frames = {len(uniq)}")

            # --- 5c: 出力バッファ ---
            if time_interp == "nearest":
                images = np.zeros((seg_frames, eff_h, eff_w, 3), dtype=image_dtype)
                acc = None
            else:
                acc = np.zeros((seg_frames * n_pix, 3), dtype=np.float32)
                images = None
            images_v = images.reshape(seg_frames * n_pix, 3) if images is not None else None

            # --- 5d: ソースを 1 パスで読みながら散布書き込み ---
            sstime = time.time()
            total_entries = seg_frames * entries_per_frame
            done_entries = 0
            next_update = max(1, total_entries // 100)

            def _scatter(fidx, frame):
                nonlocal done_entries
                if pixel_step != 1:
                    frame = frame[::pixel_step, ::pixel_step]
                frame_flat = frame.reshape(n_pix, 3)
                k = int(np.searchsorted(uniq, fidx))
                idx = order[starts[k]:ends[k]]
                if time_interp == "nearest":
                    flat_pos = idx           # = out_i * n_pix + pix
                    pix = idx % n_pix
                    images_v[flat_pos] = frame_flat[pix]
                else:
                    flat_pos = idx % (seg_frames * n_pix)
                    pix = flat_pos % n_pix
                    # 同一出力画素に z0/z1 が重複しうるため加算は np.add.at
                    np.add.at(acc, flat_pos,
                              frame_flat[pix].astype(np.float32) * weights[idx, None])
                done_entries += idx.size

            if is_seq_source:
                image_files = sorted([
                    f for f in os.listdir(f"{self.ORG_PATH}/{self.ORG_NAME}")
                    if f.endswith(('.png', '.jpg', '.tif', '.jpeg', '.bmp', '.npy'))
                ])
                for fidx in uniq:
                    p = os.path.join(f"{self.ORG_PATH}/{self.ORG_NAME}",
                                     image_files[int(fidx)])
                    frame = np.load(p) if p.endswith('.npy') else \
                        cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                    _scatter(int(fidx), frame)
                    if done_entries >= next_update:
                        self.print_progress(
                            current=done_entries, total=total_entries, color_code="33",
                            suffix=f"src={fidx} px={done_entries}/{total_entries}")
                        next_update += max(1, total_entries // 100)
            else:
                needed = set(int(v) for v in uniq)
                for fidx, fdata in self._iterate_frames(minz, maxz, needed):
                    _scatter(fidx, fdata)
                    if done_entries >= next_update:
                        self.print_progress(
                            current=done_entries, total=total_entries, color_code="33",
                            suffix=f"src={fidx} px={done_entries}/{total_entries}")
                        next_update += max(1, total_entries // 100)
            print()
            del order, uniq, starts, ends
            bench["decode_scatter"] += time.time() - sstime

            if time_interp == "linear":
                maxout = 65535 if image_dtype == np.uint16 else 255
                images = np.clip(np.rint(acc), 0, maxout).astype(image_dtype) \
                    .reshape(seg_frames, eff_h, eff_w, 3)
                del acc, weights

            # --- 5e: 出力 ---
            sstime = time.time()
            if out_type == 0:
                for i in range(images.shape[0]):
                    img_name = f"img/{videostr}_{seg_start + i}p{self.imgtype}"
                    self._save_image_with_profile(img_name, images[i])
                    self.print_progress(current=i, total=images.shape[0], color_code="32")
                print()
            else:
                self._render_images_to_sink(images, _sink_kind, _sink_obj, out_type,
                                            xy_trans_out=False, rotate_direction=False)
            bench["encode"] += time.time() - sstime

            img_written += seg_frames
            append_to_logfile(
                f"surface done:{s+1}/{separate_num}({seg_start}-{seg_end}) "
                f"z:{minz}->{maxz} mem={psutil.virtual_memory().percent}%")
            del images
            images_v = None
            gc.collect()
            print(f"done: {s+1}/{separate_num}")
            print()

        # ============================================================
        # Phase 6: クローズ・実測レポート
        # ============================================================
        if _sink_obj is not None:
            self._close_video_sink(_sink_kind, _sink_obj)
        if self.cap is not None:
            self.cap.release()

        total_time = time.time() - runFirstTime
        report = (f"All Done {total_time:.2f}sec "
                  f"({img_written} frames, {img_written/max(total_time,1e-6):.2f} f/s) | "
                  f"zmap={bench['zmap']:.1f}s sort={bench['sort']:.1f}s "
                  f"decode+scatter={bench['decode_scatter']:.1f}s "
                  f"encode={bench['encode']:.1f}s")
        print(report)
        append_to_logfile(report)
