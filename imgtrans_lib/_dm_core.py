"""drawManeuver の基盤 (CoreMixin)

含むもの:
- クラス属性 (imgtype, OUT_STILL/OUT_H264/..., _ENCODERS テーブル, _CICP_*)
- 静的 HDR EOTF/OETF (hlg_eotf, pq_eotf, hlg_oetf, pq_oetf)
- extract_params_from_filename
- __init__ : 動画/画像シーケンス読込、出力ディレクトリ作成
- _detect_input_format : ffprobe による pix_fmt / transfer 検出
- _inject_cicp_png : PNG への cICP chunk 注入
- _save_image_with_profile : HDR/SDR 静止画保存
"""
import os
import re
import json
import subprocess
from datetime import datetime
import cv2
import numpy as np

from ._utils import addmovfile, returnfps, append_to_logfile


class CoreMixin:
    imgtype = ".png" #.bmp or .jpg .tif なら 16bit OK、.png も 16bit OK です。.jpg だと 8bit に切り捨てられます。
    img_size_type = 0 #0:hw->hw 1:hw->w,w*2 2:総フレーム数分 3: square 
    outfps = 30
    recfps = 120
    progressbarsize=50
    sepVideoOut = 0 # セパレートしない場合、rawでnpアレイファイルをテンポファイルとしてハードディスクに貯めておき、全てのアレイが準備できてからレンダリングする。そのためHD容量を100GBとか普通に食う。
    memory_percent = 60 #%　映像のレンダリングの際に、確保するメモリーの許容容量。単位は％。アクティブメモリに対しての比率となる。（デフォルトは 60％）
    auto_visualize_out = True
    default_debugmode = False
    audio_form_out = False
    embedHistory_intoName = True
    some_recfps_array=[]
    plot_w_inc=5
    plot_h_inc=9
    xyt_boxel_scale=1
    
    # ===== 出力タイプ（out_type） =====
    OUT_STILL        = 0  # 静止画（SDR: 既存のまま）
    OUT_H264         = 1  # H.264（SDR 8bit 想定）
    OUT_H265         = 2  # H.265/HEVC（HDR10 10bit）
    OUT_PRORES_422   = 3  # ProRes 422 HQ（10bit 4:2:2）
    OUT_PRORES_4444  = 4  # ProRes 4444（10bit 4:4:4）
    OUT_H265_SDR = 5  # 追加
    OUT_PRORES_422_SDR  = 6
    OUT_H265_HW  = 7  # H.265 VideoToolbox HWエンコード（HDR10 10bit, 高速）


    # ===== 出力エンコーダ定義 =====
    _ENCODERS = {
        OUT_H264: {
        "codec":   "libx264",
        "pix_fmt": "yuv420p",   # H.264 は基本8bit 4:2:0
        "ext":     "mp4",
        "preset":  "fast",      # 書き出し速度 (ultrafast, superfast, fast, medium, slow...)
        "crf":     "23",        # 品質 (0=lossless, 18-23 高品質, 28以降は低ビットレート)
        "tagv":    "avc1",      # MP4/MOV 再生互換性用
        },
        OUT_H265: {
            "codec":   "libx265",
            "pix_fmt": "yuv420p10le",
            "ext":     "mp4",
            "x265_params": (
                "hdr-opt=1:repeat-headers=1:"
                "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                "master-display=G(13250,34500)B(7500,3000)"
                "R(34000,16000)WP(15635,16450)L(10000000,50):" #20251101updated L value, under 1 -> 50
                "max-cll=1000,400"
            ),
            "tagv": "hvc1",
        },
        # OUT_H265: {
        #     "codec":   "libx265",
        #     "pix_fmt": "yuv420p10le",
        #     "ext":     "mp4",
        #     "x265_params": (
        #         "hdr-opt=1:repeat-headers=1:"
        #         "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
        #         "master-display=G(13250,34500)B(7500,3000)"
        #         "R(34000,16000)WP(15635,16450)L(10000000,1):"
        #         "max-cll=1000,400"
        #     ),
        #     "tagv": "hvc1",
        # },
        OUT_PRORES_422: {
            "codec":   "prores_ks",
            "pix_fmt": "yuv422p10le",
            "ext":     "mov",
            "profile": "3",  # HQ
        },
        OUT_PRORES_4444: {
            "codec":   "prores_ks",
            "pix_fmt": "yuv444p10le",
            "ext":     "mov",
            "profile": "4",  # 4444
        },
        OUT_H265_SDR: {
        "codec":   "libx265",
        "pix_fmt": "yuv422p10le",   # SDRでも10bit維持
        "ext":     "mp4",
        "x265_params": (
            "hdr-opt=0:repeat-headers=1:"   # HDRタグはつけない
            "colorprim=bt709:transfer=bt709:colormatrix=bt709"
        ),
        "tagv": "hvc1",
        },
        OUT_PRORES_422_SDR: {
        "codec":   "prores_ks",
        "pix_fmt": "yuv422p10le",  # SDR 10bit
        "ext":     "mov",
        "profile": "3",            # HQ
        "color_primaries": "bt709",
        "color_trc": "bt709",
        "colorspace": "bt709",
        },
        OUT_H265_HW: {
        "codec":   "hevc_videotoolbox",
        "pix_fmt": "p010le",       # 10bit 4:2:0 (VideoToolbox用)
        "ext":     "mp4",
        "profile": "main10",
        "tagv":    "hvc1",
        },
    }

    # === HLG EOTF (HLG → Linear) ===
    @staticmethod
    def hlg_eotf(E):
        """
        入力: E = 0..1 の正規化信号 (HLG)
        出力: L = リニア光量
        """
        a = 0.17883277
        b = 1 - 4 * a
        c = 0.5 - a * np.log(4 * a)

        L = np.where(E <= 0.5,
                    (E ** 2) / 3,
                    (np.exp((E - c) / a) + b) / 12)
        return L
    @staticmethod
    def pq_eotf(E):
        # ST.2084 (PQ) 定数
        m1 = 2610.0 / 16384.0
        m2 = 2523.0 / 32.0
        c1 = 3424.0 / 4096.0
        c2 = 2413.0 / 128.0
        c3 = 2392.0 / 128.0
        
        E = np.maximum(E, 1e-8)  # ゼロ割り防止
        L = ((np.maximum(E**(1/m2) - c1, 0)) / (c2 - c3 * E**(1/m2)))**(1/m1)
        return L

    # === HLG OETF (Linear → HLG) ===
    @staticmethod
    def hlg_oetf(L):
        """
        入力: L = リニア光量 (0..1)
        出力: E = HLG信号 (0..1)
        """
        a = 0.17883277
        b = 1 - 4 * a
        c = 0.5 - a * np.log(4 * a)

        E = np.where(L <= 1/12,
                    np.sqrt(3 * L),
                    a * np.log(12 * L - b) + c)
        return E

    # === PQ OETF (Linear → PQ) ===
    @staticmethod
    def pq_oetf(L):
        """
        Linear → PQ (ST2084)
        L = 0..1 (リニア光量正規化)
        """
        m1 = 0.1593017578125
        m2 = 78.84375
        c1 = 0.8359375
        c2 = 18.8515625
        c3 = 18.6875

        Lm = np.power(L, m1)
        num = c1 + c2 * Lm
        den = 1 + c3 * Lm
        E = np.power(num / den, m2)
        return E
    
    #png>data変換に必要
    @staticmethod
    def extract_params_from_filename(filename: str):
        """ファイル名から t種別 と 付属パラメータを抽出する"""
        name = os.path.basename(filename)

        # --- space ---
        m = re.search(r"space_(\d+)\.png$", name)
        if m:
            scan_range = int(m.group(1))
            return {"type": "space", "range": scan_range}

        # --- time ---
        # m = re.search(r"outimage_time_(\d+)-(\d+)\.png$", name)
        m = re.search(r"time_([-+]?\d+)-([-+]?\d+)\.png$", name)
        if m:
            vmin, vmax = int(m.group(1)), int(m.group(2))
            return {"type": "time", "vmin": vmin, "vmax": vmax}

        # --- rate ---
        m = re.search(r"rate_([0-9.]+)\.png$", name)
        if m:
            max_dev = float(m.group(1))
            print({"type": "rate", "max_dev": max_dev})
            return {"type": "rate", "max_dev": max_dev}

        raise ValueError(f"ファイル名から情報を抽出できません: {filename}")

    def __init__(self,videopath:str,sd:bool,outdir:str=None,datapath:str=None,foldername_attr:str=None,another_fps_dir=None,recfps:float=None,outfps:float=None):
        # recfps : 入力映像の「実際の」fps。Noneなら動画メタデータのfpsを採用。
        #          ハイスピード収録で動画メタデータと実fpsが乖離する場合(例:480fps収録/30fps格納)に明示指定する。
        # outfps : 出力fps。Noneならクラスデフォルト(30)を維持。
        self.VIDEO_PATH = videopath
        self.ORG_NAME= videopath.split(".")[0].rsplit("/",1)[-1]
        self.ORG_PATH=videopath.split(".")[0].rsplit("/",1)[0] if outdir==None else outdir
        # videopathがディレクトリ（画像シーケンス）かファイル（ビデオ）かチェック
        if os.path.isdir(videopath):
            # 画像ファイルリストの取得
            image_files = [f for f in os.listdir(videopath) if f.endswith(('.png', '.jpg','.tif','.jpeg', '.bmp','.npy'))]
            image_files.sort()  # 必要に応じてソート
            # 画像ファイルがある場合、最初の画像を読み込み
            if image_files:
                first_image_path = os.path.join(videopath, image_files[0])
                first_image = np.load(first_image_path) if first_image_path.endswith('.npy') else cv2.imread(first_image_path)
                self.cap = None  # VideoCaptureは不要
                self.width = first_image.shape[1]
                self.height = first_image.shape[0]
                self.count = len(image_files)
                # self.recfps = 120  # FPSは画像シーケンスでは不要
            else:
                raise ValueError("No image files found in the specified folder.")
        else:
            # ビデオファイルの場合
            self.cap = cv2.VideoCapture(videopath)
            self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)# 幅
            self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)# 高さ
            self.count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)# 総フレーム数
            self.inputmovfps = self.cap.get(cv2.CAP_PROP_FPS)# fps
            self.recfps = self.inputmovfps # HDR対応のため急造、
        # recfps/outfps の明示上書き (動画メタデータfps ≠ 実fps の場合に利用)
        if recfps is not None:
            self.recfps = recfps
        if outfps is not None:
            self.outfps = outfps
        self.data = [] if datapath == None else  np.load(datapath)
        self.scan_direction=sd 
        self.scan_nums = int(self.width) if sd % 2 == 1 else int(self.height)
        self.slit_length = int(self.height) if sd % 2 == 1 else int(self.width)
        self.sc_resetPositionMap=[]
        self.sc_rateMap = []
        self.sc_inPanMap = []
        self.sc_now_depth = []
        self.sc_SDRateMap = []
        self.sc_movementRateMap = []
        self.sc_parallel_component_Map = []
        self.sc_perpendicular_component_Map = []
        self.depth_to_sel_recfps = []
        self.out_videopath = ""
        sd_attr = "Hslit" if sd==0 else "Vslit" #Vertical or horizontal
        self.out_name_attr =datetime.now().strftime('%Y_%m%d')+"_"+sd_attr 
        if foldername_attr != None : self.out_name_attr+="_"+foldername_attr
        self.log = 0
        self.infolog = 0
        self.sc_FNAME =self.ORG_NAME + ".AIFF"
        self.cycle_axis=[]
        self.another_videos = []
        if another_fps_dir != None:
            self.another_videos = addmovfile(another_fps_dir)
            self.another_videos.append(videopath)
            self.some_recfps_array=returnfps(self.another_videos)
            # self.some_recfps_array.append(self.recfps)
            # ソートする際のインデックスを取得
            sorted_indices = np.argsort(np.array(self.some_recfps_array))
            self.some_recfps_array = np.sort(np.array(self.some_recfps_array)).tolist()
            self.another_videos = np.array(self.another_videos)[sorted_indices]
            print("self.another_videos=",self.another_videos)
            self.renderfps_scales= np.array(self.some_recfps_array)/self.recfps
            print("self.renderfps_scales=",self.renderfps_scales)

        # ==== HDR対応 ====
        # === 入力のフォーマット情報（初期値） ===
        self.input_pix_fmt     = None
        self.input_bit_depth   = 8
        self.input_primaries   = None
        self.input_transfer    = None
        self.input_colorspace  = None
        self.input_rotation    = 0   # Display Matrix 回転角 (iPhone縦位置撮影 = -90 等)
        self.is_morethan_8bit  = False #HDRの場合およそ2.6倍の時間がかかる。
        self.force_hdr_mode = None  # ← 入力に従う（HLGならHLG, PQならPQ）
        # self.force_hdr_mode = "hlg" # ← 強制的にHLG出力
        # self.force_hdr_mode = "pq"  # ← 強制的にPQ出力

        # 入力動画のピクセルフォーマット/ビット深度/HDR 推定
        self._detect_input_format()
        # ==== HDR対応 ====
        
        #ディレクトリ作成、そのディデクトリに移動
        # NPATH = self.ORG_PATH+"/"+self.ORG_NAME+"_"+self.out_name_attr 
        # if os.path.isdir(NPATH)==False:
        #     os.makedirs(NPATH)
        NPATH_BASE = self.ORG_PATH + "/" + self.ORG_NAME + "_" + self.out_name_attr
        NPATH = NPATH_BASE
        counter = 1

        while os.path.isdir(NPATH):
            NPATH = f"{NPATH_BASE}({counter})"
            counter += 1

        os.makedirs(NPATH)
        os.chdir(NPATH)
        # 現在の日時を取得
        now = datetime.now()
        # 日、時間、分の形式で文字列に変換
        log_entry = now.strftime('%Y-%m-%d %H:%M')
        append_to_logfile("-")
        append_to_logfile(self.ORG_NAME+"_"+ sd_attr)
        append_to_logfile(log_entry)

    #added 2025 9/19 for appling HDR out
    def _detect_input_format(self):
        """
        ffprobe を使って入力ストリームの pix_fmt / color_primaries / color_transfer / colorspace を取得し、
        self.input_bit_depth / self.is_morethan_8bit などを推定する。
        ffprobe 不在や解析不能時は 8bit SDR とみなす。
        """
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=pix_fmt,color_primaries,color_transfer,colorspace:stream_side_data_list",
                "-of", "json",
                self.VIDEO_PATH
            ]
            out = subprocess.check_output(cmd).decode("utf-8", errors="ignore")
            info = json.loads(out)
            st = (info.get("streams") or [{}])[0]

            self.input_pix_fmt    = st.get("pix_fmt")
            self.input_primaries  = st.get("color_primaries")
            self.input_transfer   = st.get("color_transfer")
            self.input_colorspace = st.get("colorspace")

            # Display Matrix の回転角 (縦位置撮影のスマホ動画等)。
            # OpenCV は自動適用するが PyAV は適用しないため、PyAV デコード時に
            # frame_to_ndarray(rotation=...) で同じ向きに揃える必要がある。
            for sd in st.get("side_data_list") or []:
                if sd.get("rotation") is not None:
                    self.input_rotation = int(sd["rotation"])
                    break

            # ビット深度推定（例: yuv420p10le → 10 / rgb48le → 16）
            bit_depth = 8
            if self.input_pix_fmt:
                m = re.search(r'(\d+)(le|be)?$', self.input_pix_fmt)
                if m:
                    bit_depth = int(m.group(1))
                elif self.input_pix_fmt.startswith("rgb48"):
                    bit_depth = 16
                elif self.input_pix_fmt.startswith("rgb30") or self.input_pix_fmt.endswith("xyz12"):
                    bit_depth = 10  # だいたいの目安

            self.input_bit_depth = bit_depth
            # HDR推定（PQ or HLG or 10bit超）
            self.is_morethan_8bit = (bit_depth > 8) or (self.input_transfer in ("smpte2084", "arib-std-b67"))
            # SDR 10bit 判定
            self.is_sdr10bit = (self.input_transfer == "bt709" and bit_depth > 8)
        except Exception:
            # 失敗時は既定のまま（SDR 8bit）
            self.input_bit_depth = 8
            self.is_morethan_8bit = False

    # cICP code mappings (ITU-T H.273)
    _CICP_PRIMARIES = {
        "bt709": 1, "bt2020": 9, "smpte432": 12,  # Display P3
    }
    _CICP_TRANSFER = {
        "bt709": 1, "smpte2084": 16, "arib-std-b67": 18,
    }

    def _inject_cicp_png(self, png_path: str, primaries_code: int, transfer_code: int,
                         matrix_code: int = 0, full_range: int = 1):
        """PNG ファイルに cICP chunk を注入する (IHDR直後)"""
        import struct, zlib
        cicp_data = struct.pack('BBBB', primaries_code, transfer_code, matrix_code, full_range)
        chunk_type = b'cICP'
        chunk_len = struct.pack('>I', len(cicp_data))
        crc = struct.pack('>I', zlib.crc32(chunk_type + cicp_data) & 0xffffffff)
        cicp_chunk = chunk_len + chunk_type + cicp_data + crc

        with open(png_path, 'rb') as f:
            png = f.read()
        # IHDR chunk: 4(len) + 4(type) + 13(data) + 4(crc) = 25 bytes, after 8-byte signature
        ihdr_end = 8 + 25
        with open(png_path, 'wb') as f:
            f.write(png[:ihdr_end] + cicp_chunk + png[ihdr_end:])

    def _save_image_with_profile(self, img_name: str, rgb_array):
        """
        RGB配列を画像として保存。HDRの場合はカラープロファイルを埋め込む。
        対応形式: PNG (cICP), TIFF (sips ICC), その他 (cv2のみ)
        rgb_array: np.uint8 or np.uint16, shape (H, W, 3), RGB順
        """
        transfer = getattr(self, "input_transfer", None)
        is_hdr = transfer in ("smpte2084", "arib-std-b67")
        lower_name = img_name.lower()

        # HDR PNG: cICP chunkでPQ/HLGプロファイルを完全に埋め込める
        if is_hdr and lower_name.endswith(".png"):
            cv2.imwrite(img_name, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
            primaries_str = getattr(self, "input_primaries", None) or "bt2020"
            pri_code = self._CICP_PRIMARIES.get(primaries_str, 9)
            trc_code = self._CICP_TRANSFER.get(transfer, 16)
            self._inject_cicp_png(img_name, pri_code, trc_code)

        # HDR TIFF: sipsでICCプロファイル埋め込み（BT.2020 primariesのみ、TRCはガンマ）
        elif is_hdr and (lower_name.endswith(".tif") or lower_name.endswith(".tiff")):
            cv2.imwrite(img_name, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
            icc_path = "/System/Library/ColorSync/Profiles/ITU-2020.icc"
            if os.path.isfile(icc_path):
                subprocess.run(["sips", "--embedProfile", icc_path, img_name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        else:
            cv2.imwrite(img_name, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

