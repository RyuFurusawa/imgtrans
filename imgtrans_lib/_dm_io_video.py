"""drawManeuver の動画 I/O (IOVideoMixin)

含むもの:
- _build_ffmpeg_cmd : 出力形式毎の ffmpeg コマンド生成 (H.264/H.265/ProRes/HDR/SDR)
- _open_video_sink  : ffmpeg/PyAV シンクをオープン
- _write_video_frame: シンクへ 1 フレームを書き込み (RGB/YUV, HDR/SDR 自動切替)
- _close_video_sink : シンクをクローズ
- yuv420_to_rgb     : YUV420 planar → RGB 静的変換ヘルパ
"""
import sys
import subprocess
import cv2
import numpy as np


class IOVideoMixin:
    def _build_ffmpeg_cmd(self, out_path: str, width: int, height: int, fps: float, out_type: int, use_yuv_native: bool = False):
        """
        出力タイプに応じて ffmpeg の Popen コマンド配列を作る。
        use_yuv_native=True の場合、入力は yuv422p10le (planar) を stdin で流す。
        それ以外は rgb48le の RAW を stdin で流し込む前提。
        """
        cmd = ["ffmpeg", "-y"]

        if out_type == self.OUT_H264:
            enc = self._ENCODERS[self.OUT_H264]
            cmd += [
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",            # stdinは8bit
                "-s:v", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",                      # stdin
                "-c:v", enc["codec"],
                "-preset", enc.get("preset", "fast"),
                "-crf", enc.get("crf", "23"),
                "-pix_fmt", enc["pix_fmt"],     # 出力はyuv420p
                "-tag:v", enc.get("tagv", "avc1"),
                out_path
            ]

        elif out_type == self.OUT_H265:
            enc = self._ENCODERS[self.OUT_H265]

            # HDRトランスファーを入力側にも明示（RGB→YUV変換マトリクスに影響）
            if self.force_hdr_mode == "hlg":
                in_trc = "arib-std-b67"
            elif self.force_hdr_mode == "pq":
                in_trc = "smpte2084"
            else:
                if getattr(self, "input_transfer", None) == "arib-std-b67":
                    in_trc = "arib-std-b67"
                else:
                    in_trc = "smpte2084"

            cmd += [
                "-f", "rawvideo",
                "-pix_fmt", "rgb48le",          # stdinは16bit
                "-color_primaries", "bt2020",    # 入力RGBの色域を明示
                "-color_trc", in_trc,            # 入力RGBのトランスファーを明示
                "-colorspace", "bt2020nc",       # 入力RGBの変換マトリクスを明示
                "-s:v", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",                      # stdin← ここまでが「入力側」の設定
                "-c:v", enc["codec"],
                "-pix_fmt", enc["pix_fmt"],  # 出力はyuv420p10le#← ここからが「出力側」の設定
                "-tag:v", enc["tagv"],
            ]
            # if self.is_morethan_8bit:
            #     cmd +=["-pix_fmt", enc["pix_fmt"],]  # 出力はyuv420p10le
            # else :
            #     cmd +=["-pix_fmt", "rgb24",]  # stdinは16bit
        
            # === HDRトランスファーの切り替え ===
            if self.force_hdr_mode == "hlg":
                cmd += [
                    "-x265-params",
                    (
                        "hdr-opt=1:repeat-headers=1:"
                        "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc"
                    ),
                ]
            elif self.force_hdr_mode == "pq":
                cmd += [
                    "-x265-params",
                    enc["x265_params"]  # 既存のPQ用パラメータをそのまま利用
                ]
            else:
                # 入力に従う
                if getattr(self, "input_transfer", None) == "arib-std-b67":
                    cmd += [
                    "-x265-params",
                    (
                        "hdr-opt=1:repeat-headers=1:"
                        "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc"
                    ),
                ]
                else:
                    cmd += [
                    # "-color_range", "tv",#HDR系 (PQ/HLG) は 基本的に TVレンジ前提
                    "-x265-params",
                    enc["x265_params"]  # 既存のPQ用パラメータをそのまま利用
                ]

            cmd.append(out_path)

        elif out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444):
            enc = self._ENCODERS[out_type]

            # HDRトランスファーを先に決定
            if self.force_hdr_mode == "hlg":
                trc = "arib-std-b67"
            elif self.force_hdr_mode == "pq":
                trc = "smpte2084"
            else:
                if getattr(self, "input_transfer", None) == "arib-std-b67":
                    trc = "arib-std-b67"
                else:
                    trc = "smpte2084"

            # prores_ks は -color_primaries/-color_trc を colr atom に書かない。
            # まず一時ファイルにエンコード → remux で colr を後付けする。
            self._prores_colr = {
                "primaries": "bt2020",
                "trc": trc,
                "space": "bt2020nc",
                "range": "tv",
            }
            if use_yuv_native:
                # YUV-native: RGB→YUV変換なし、色バイアスゼロ
                cmd += [
                    "-f", "rawvideo",
                    "-pix_fmt", "yuv422p10le",       # YUVプレーンを直接入力
                    "-s:v", f"{width}x{height}",
                    "-r", str(fps),
                    "-color_primaries", "bt2020",
                    "-color_trc", trc,
                    "-colorspace", "bt2020nc",
                    "-color_range", "tv",
                    "-i", "-",
                    "-c:v", enc["codec"],
                    "-pix_fmt", enc["pix_fmt"],
                    "-profile:v", enc["profile"],
                    "-vendor", "apl0",
                    "-color_primaries", "bt2020",
                    "-color_trc", trc,
                    "-colorspace", "bt2020nc",
                    "-color_range", "tv",
                    out_path,
                ]
            else:
                cmd += [
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb48le",
                    "-color_primaries", "bt2020",    # 入力RGBの色域を明示
                    "-color_trc", trc,               # 入力RGBのトランスファーを明示
                    "-colorspace", "bt2020nc",       # 入力RGBの変換マトリクスを明示
                    "-s:v", f"{width}x{height}",
                    "-r", str(fps),
                    "-i", "-",
                    "-c:v", enc["codec"],
                    "-pix_fmt", enc["pix_fmt"],
                    "-profile:v", enc["profile"],
                    "-vendor", "apl0",
                    "-color_primaries", "bt2020",    # 出力メタデータ
                    "-color_trc", trc,               # 出力メタデータ
                    "-colorspace", "bt2020nc",
                    "-color_range", "tv",
                    out_path,
                ]

        elif out_type == self.OUT_H265_SDR:
            cmd += [
                "-f", "rawvideo",
                "-pix_fmt", "rgb48le",       # stdinは16bit
                "-s:v", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",                   # stdin
                "-c:v", "libx265",
                "-pix_fmt", "yuv422p10le",   # SDR10bitを維持
                "-tag:v", "hvc1",
                "-x265-params", "repeat-headers=1:hdr-opt=0:colorprim=bt709:transfer=bt709:colormatrix=bt709",
                out_path
            ]

        elif out_type == self.OUT_PRORES_422_SDR:
            cmd += [
                "-f", "rawvideo",
                "-pix_fmt", "rgb48le",       # stdinは16bit
                "-s:v", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",                   # stdin
                "-c:v", "prores_ks",
                "-pix_fmt", "yuv422p10le",
                "-profile:v", "3",           # HQ
                "-color_primaries", "bt709",
                "-colorspace", "bt709",
                "-color_trc", "bt709",
                out_path
            ]
        elif out_type == self.OUT_H265_HW:
            # === H.265 VideoToolbox ハードウェアエンコード (HDR10 10bit) ===
            enc = self._ENCODERS[self.OUT_H265_HW]
            cmd += [
                "-f", "rawvideo",
                "-pix_fmt", "rgb48le",          # stdinは16bit
                "-s:v", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",                      # stdin
                "-c:v", enc["codec"],            # hevc_videotoolbox
                "-pix_fmt", enc["pix_fmt"],      # p010le (10bit)
                "-profile:v", enc["profile"],    # main10
                "-tag:v", enc["tagv"],           # hvc1
                "-color_primaries", "bt2020",
                "-colorspace", "bt2020nc",
            ]

            # === HDRトランスファーの切り替え ===
            if self.force_hdr_mode == "hlg":
                cmd += ["-color_trc", "arib-std-b67"]
            elif self.force_hdr_mode == "pq":
                cmd += ["-color_trc", "smpte2084"]
            else:
                if getattr(self, "input_transfer", None) == "arib-std-b67":
                    cmd += ["-color_trc", "arib-std-b67"]
                else:
                    cmd += ["-color_trc", "smpte2084"]

            cmd += [out_path]

        else:
            raise ValueError(f"Unsupported out_type: {out_type}")

        print("FFmpeg CMD:", " ".join(cmd))
        return cmd
    
    def _open_video_sink(self, out_path: str, width: int, height: int, fps: float, out_type: int, use_yuv_native: bool = False):
        print(sys._getframe().f_code.co_name)
        """
        書き出し先を用意。
        - HDR系(OUT_H265/PRORES系) or 入力が >8bit → ffmpeg stdin
        - それ以外（SDR 8bit H.264 等）→ OpenCV VideoWriter
        戻り値: ("ffmpeg", Popen) or ("cv2", cv2.VideoWriter)
        """
        #将来的には、ここにH264も加えてFFmpegに統一したい
        if out_type in (self.OUT_H265, self.OUT_PRORES_422, self.OUT_PRORES_4444, self.OUT_H265_SDR, self.OUT_PRORES_422_SDR, self.OUT_H265_HW):
            print(f"FFmpeg Out (yuv_native={use_yuv_native})")
            # FFmpeg パイプ
            cmd = self._build_ffmpeg_cmd(out_path, width, height, fps, out_type, use_yuv_native=use_yuv_native)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self._last_out_path = out_path   # colr remux 用に保存
            return "ffmpeg", proc
        else:
            # OpenCV VideoWriter（SDR 8bit）または、１０ビットだとしてもH264アウトの場合は、OpenCVで処理する
            print("OpenCV Out")
            if out_type == self.OUT_H264:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
                ext_ok = out_path.lower().endswith(".mp4")
                if not ext_ok:
                    out_path += ".mp4"
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_path, fourcc, fps, (int(width), int(height)), True)
            return "cv2", vw
        
    def _write_video_frame(self, sink_kind: str, sink_obj, frame, out_type: int,
                       xy_trans_out: bool=False, rotate_direction: bool=False,
                       use_yuv_native: bool=False):
        """
        1フレーム書き込み。
        - sink_kind == 'ffmpeg': RGB16 (uint16) またはYUV planes を stdin に流す
        - sink_kind == 'cv2'   : そのまま cv2.VideoWriter.write
        use_yuv_native=True の場合、frame は (Y, Cb, Cr) の tuple。
        """

        # === YUV-native パス ===
        if use_yuv_native and sink_kind == "ffmpeg":
            y, cb, cr = frame
            if xy_trans_out:
                if rotate_direction:
                    y  = y.transpose(1, 0)[:, ::-1]
                    cb = cb.transpose(1, 0)[:, ::-1]
                    cr = cr.transpose(1, 0)[:, ::-1]
                else:
                    y  = y.transpose(1, 0)[::-1]
                    cb = cb.transpose(1, 0)[::-1]
                    cr = cr.transpose(1, 0)[::-1]
            # planar yuv422p10le: Y全画素 → Cb半幅 → Cr半幅
            sink_obj.stdin.write(y.astype(np.uint16).tobytes())
            sink_obj.stdin.write(cb.astype(np.uint16).tobytes())
            sink_obj.stdin.write(cr.astype(np.uint16).tobytes())
            return

        # === 回転・XY変換（RGBパス）===
        if xy_trans_out:
            if rotate_direction:
                frame = frame.transpose(1,0,2)[:, ::-1]
            else:
                frame = frame.transpose(1,0,2)[::-1]

        if sink_kind == "ffmpeg":
            if out_type == self.OUT_H264 :
                # --- H.264 (8bit SDR) ---
                rgb8 = frame.astype(np.uint8)
                sink_obj.stdin.write(rgb8.tobytes())
            else:
                # --- H.265 / ProRes (HDR, 16bit) ---
                if not self.is_morethan_8bit:
                    # print("SDR (8bit) の保存")
                    # SDR (8bit) を16bitに引き上げ
                    rgb8 = frame[..., ::-1].astype(np.uint8)  # BGR→RGB OpenCvで処理していたため
                    # rgb16 = (rgb8.astype(np.uint16) * 257)
                    # rgb_lin = rgb16.astype(np.float32) / 65535.0
                    # # === OETF: 出力モード選択 ===
                    # rgb_out = self.pq_oetf(rgb_lin)
                    # # === 16bit整数に戻す ===
                    # rgb16 = np.clip(rgb_out * 65535.0, 0, 65535).astype(np.uint16)
                    sink_obj.stdin.write(rgb8.tobytes())
                else:
                    if self.force_hdr_mode != None : #入力のフォーマットから動画フォーマットを変更する場合。
                        # print("HDR入力 (HLG or PQ)")
                        # HDR入力 (HLG or PQ)
                       
                        if self.input_transfer == "arib-std-b67" and self.force_hdr_mode == "pq": # HLG入力 PQ out
                            # === EOTF: HLGならリニア化 ===
                            rgb = frame.astype(np.float32) / 65535.0
                            rgb_lin = self.hlg_eotf(rgb)
                            # === OETF: 出力モード選択 ===
                            rgb_out = self.pq_oetf(rgb_lin)
                            # === 16bit整数に戻す ===
                            rgb16 = np.clip(rgb_out * 65535.0, 0, 65535).astype(np.uint16)
                        elif self.input_transfer == "smpte2084" and self.force_hdr_mode == "hlg" :   # PQ入力 HLG out
                            rgb = frame.astype(np.float32) / 65535.0
                            rgb_lin = self.pq_eotf(rgb)   
                            rgb_out = self.hlg_oetf(rgb_lin)
                            # === 16bit整数に戻す ===
                            rgb16 = np.clip(rgb_out * 65535.0, 0, 65535).astype(np.uint16)
                        else: 
                            # print(" PyAV rgb48le → そのまま1")
                            rgb16 = frame.astype(np.uint16)
                        
                    else :
                        #sony A7s などの１０ビットはもともとリニアフォーマット
                        # PyAV rgb48le → そのまま
                        # print(" PyAV rgb48le → そのまま2")
                        rgb16 = frame.astype(np.uint16)
                    # print("ffmpeg CMD:", " ".join(cmd))
                    # print("frame dtype:", frame.dtype, "shape:", frame.shape)
                    sink_obj.stdin.write(rgb16.tobytes())
        else:
            # cv2.VideoWriter は BGR を期待 → パイプライン内部の RGB から変換
            sink_obj.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # def _write_video_frame(
    # self,
    # sink_kind: str,
    # sink_obj,
    # frame,
    # out_type: int,
    # xy_trans_out: bool = False,
    # rotate_direction: bool = False
    # ):
    #     # print(sys._getframe().f_code.co_name)
    #     """
    #     1フレーム書き込み。
    #     - sink_kind == 'ffmpeg' のとき: RGB16 (uint16) を stdin に流す
    #     - sink_kind == 'cv2' のとき: そのまま cv2.VideoWriter.write
    #     """

    #     if xy_trans_out:
    #         if rotate_direction:
    #             frame = frame.transpose(1, 0, 2)[:, ::-1]
    #         else:
    #             frame = frame.transpose(1, 0, 2)[::-1]

    #     if sink_kind == "ffmpeg":
    #         if out_type == self.OUT_H264:
    #             # H.264 は 8bit 前提
    #             rgb8 = frame.astype(np.uint8)  # すでにRGBなので変換不要
    #             sink_obj.stdin.write(rgb8.tobytes())
    #         else:
    #             # H.265 / ProRes は 16bit 前提
    #             if not self.is_morethan_8bit:
    #                 # 入力が8bitなら16bitに引き上げ
    #                 rgb8 = frame[..., ::-1].astype(np.uint8)
    #                 rgb16 = (rgb8.astype(np.uint16) * 257)  # 0–255 → 0–65535
    #             else:
    #                 rgb16 = frame.astype(np.uint16)
    #             # print("before:", self.input_transfer, 
    #             #     "min:", frame.min(), "max:", frame.max(), 
    #             #     "mean:", frame.mean(axis=(0,1)))
    #             # print("after:", self.input_transfer, 
    #             #     "min:", rgb16.min(), "max:", rgb16.max(), 
    #             #     "mean:", rgb16.mean(axis=(0,1)))
    #             sink_obj.stdin.write(rgb16.tobytes())
    #     else:
    #         sink_obj.write(frame)


    def _close_video_sink(self, sink_kind: str, sink_obj):
        print(sys._getframe().f_code.co_name)
        """ シンクのクローズ """
        if sink_kind == "ffmpeg":
            try:
                if sink_obj.stdin:
                    sink_obj.stdin.flush()
                    sink_obj.stdin.close()
            except Exception as e:
                print(f"⚠️ Failed to close FFmpeg stdin: {e}")
            finally:
                try:
                    sink_obj.wait()
                    print("✅ FFmpeg process finished.")
                except Exception as e:
                    print(f"⚠️ FFmpeg wait() failed: {e}")

            # ProRes colr atom 後付け remux
            # 注: _build_ffmpeg_cmd で入力側・出力側の両方に
            # -color_primaries / -color_trc / -colorspace を直接指定済みのため
            # remux は不要。スキップする。
            if getattr(self, "_prores_colr", None):
                print("ℹ️ colr atom はエンコード時に直接付与済み（remuxスキップ）")
                self._prores_colr = None
        else:
            sink_obj.release()

    # added 20251001
    def yuv420_to_rgb(yuv_planes, width, height):
        Y, U, V = yuv_planes

        # 10bit → float 正規化 (0..1)
        Y = Y.astype(np.float32) / 1023.0
        U = U.astype(np.float32) / 1023.0
        V = V.astype(np.float32) / 1023.0

        # U, V をYのサイズに拡大（最近傍 or バイリニア）
        U_up = cv2.resize(U, (width, height), interpolation=cv2.INTER_LINEAR)
        V_up = cv2.resize(V, (width, height), interpolation=cv2.INTER_LINEAR)

        # オフセット補正（YUV→YCbCr）
        U_up = U_up - 0.5
        V_up = V_up - 0.5

        # === Rec.2020 (non-constant luminance) の変換行列 ===
        # R = Y + 1.4746 * Cr
        # G = Y - 0.1645 * Cb - 0.5714 * Cr
        # B = Y + 1.8814 * Cb

        R = Y + 1.4746 * V_up
        G = Y - 0.1645 * U_up - 0.5714 * V_up
        B = Y + 1.8814 * U_up

        # クリップして16bitに戻す
        rgb = np.stack([R, G, B], axis=-1)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb16 = (rgb * 65535).astype(np.uint16)

        return rgb16
        
    # added 20250921
