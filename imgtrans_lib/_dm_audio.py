"""drawManeuver の音声出力 (AudioMixin)

Maneuver Data (座標列) を音声処理用の滑らかな軌跡に変換し、
- Python 内でのオフライン音声レンダリング (SC 不要)
- SuperCollider 用の改良版 scd 出力 (コントロールトラック方式)
- フーリエ成分 (周波数の重ね合わせ) としての CSV 出力
を行う。

背景:
  旧 scd_out は 30Hz(outfps) の .set + Lag2UD による区分線形補間と、
  rate 積分 + resetp 再トリガ方式のため、
    1. 補間の角 (C1 不連続) による歪み
    2. 再トリガ時のクリックノイズ
    3. rate 積分のドリフトで CycleTrans が周期終端で完全同期しない
  という問題があった。

  本モジュールは「レート駆動」ではなく「位置駆動」で再生する。
  座標列をフレーム軸方向にフーリエ変換 (帯域制限アップサンプル) または
  キュービックスプラインで補間し、オーディオレートの滑らかな
  再生位置信号 pos(t) を直接生成する。位置を直接参照するため
  ドリフトが原理的に発生せず、周期データは周期終端で厳密に再同期する。

単位系:
  self.data[k, x, 1] : ソース映像のフレーム番号 (float) → 秒 = /recfps
  出力タイムライン    : フレーム k の時刻 = k/outfps 秒
"""
import os
import sys
import math
import subprocess
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import wavfile


class AudioMixin:

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    def _audio_source_extent(self):
        """inPan (空間座標) の正規化に使うソース映像の空間方向ピクセル数。"""
        return int(self.width) if self.scan_direction % 2 == 1 else int(self.height)

    def _audio_voice_indices(self, n_voices):
        """scan_nums 軸から n_voices 本の走査位置を等間隔に選ぶ。"""
        cols = self.data.shape[1]
        n = int(min(max(2, n_voices), cols))
        return np.unique(np.round(np.linspace(0, cols - 1, n)).astype(int))

    def _audio_trajectories(self, n_voices):
        """ボイスごとの軌跡を返す。

        returns:
          traj_sec   : (F, n) ソース内の再生位置 [秒]
          inpan_traj : (F, n) 正規化した空間座標 (0..1)。
                       0=ソース左端 (Lチャンネル寄り) / 1=右端 (Rチャンネル寄り)
        """
        if len(self.data) == 0:
            raise ValueError("maneuver data がありません。先に add 系の変換を実行してください。")
        idx = self._audio_voice_indices(n_voices)
        traj_sec = self.data[:, idx, 1] / self.recfps
        ext = max(self._audio_source_extent() - 1, 1)
        inpan_traj = np.clip(self.data[:, idx, 0] / ext, 0.0, 1.0)
        return traj_sec.astype(np.float64), inpan_traj.astype(np.float64)

    @staticmethod
    def _traj_segments(traj_1d, outfps, jump_thresh_sec):
        """軌跡の不連続点 (ジャンプ) でセグメント分割する。

        returns: [(f0, f1), ...] フレームindexの閉区間リスト
        """
        d = np.abs(np.diff(traj_1d))
        jumps = np.where(d > jump_thresh_sec)[0]
        bounds = [0] + [int(j) + 1 for j in jumps] + [len(traj_1d)]
        return [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)
                if bounds[i + 1] - 1 >= bounds[i]]

    @staticmethod
    def _fourier_upsample(seg_vals, n_up, n_harmonics=None):
        """1セグメントを帯域制限 (フーリエ) アップサンプルする。

        端点を結ぶ直線 (トレンド) を除去した残差を周期信号とみなして
        rfft → (任意で上位 n_harmonics 成分のみ残す) → ゼロ詰め irfft。
        残差の両端が 0 になるため周期接続が連続になり、Gibbs を抑える。

        返り値は n_up サンプルで、サンプル j は元データのフレーム位置
        j * (n-1) / n_up に対応する (周期グリッド、終端は含まない)。
        """
        n = len(seg_vals)
        m = n - 1  # 周期 (最終点はトレンド上なので落とす)
        t = np.arange(n)
        if n < 4 or n_up <= n:
            # 短すぎるセグメントは線形補間
            return np.interp(np.arange(n_up) * m / n_up, t, seg_vals)
        trend = seg_vals[0] + (seg_vals[-1] - seg_vals[0]) * t / m
        resid = seg_vals - trend
        spec = np.fft.rfft(resid[:m])
        if n_harmonics is not None and n_harmonics < len(spec) - 1:
            mags = np.abs(spec[1:])
            keep = np.argsort(mags)[::-1][:n_harmonics] + 1
            mask = np.zeros(len(spec), dtype=bool)
            mask[0] = True
            mask[keep] = True
            spec = np.where(mask, spec, 0)
        out_spec = np.zeros(n_up // 2 + 1, dtype=complex)
        k = min(len(spec), len(out_spec))
        out_spec[:k] = spec[:k]
        resid_up = np.fft.irfft(out_spec, n_up) * (n_up / m)
        trend_up = np.interp(np.arange(n_up) * m / n_up, t, trend)
        return trend_up + resid_up

    @staticmethod
    def _spline_upsample(seg_vals, n_up):
        n = len(seg_vals)
        if n < 4:
            return np.interp(np.linspace(0, n - 1, n_up), np.arange(n), seg_vals)
        cs = CubicSpline(np.arange(n), seg_vals)
        return cs(np.linspace(0, n - 1, n_up))

    def _upsample_traj(self, traj_1d, sr_out, smooth="fourier", n_harmonics=None,
                       jump_thresh_sec=0.25):
        """軌跡 (outfps サンプル) をオーディオ/コントロールレートへ滑らかに補間する。

        returns:
          up        : (n_out,) 補間済み軌跡
          seg_edges : セグメント境界のサンプル index リスト (クロスフェード用)
        """
        F = len(traj_1d)
        n_out = int(round(F / self.outfps * sr_out))
        up = np.zeros(n_out, dtype=np.float64)
        seg_edges = []
        segments = self._traj_segments(traj_1d, self.outfps, jump_thresh_sec)
        for (f0, f1) in segments:
            # フレーム k がサンプル round(k/outfps*sr_out) に正確に乗るよう
            # [s0, sL] にセグメント本体を配置し、残り (ジャンプ直前の
            # 1フレーム分 / 末尾) は終端値をホールドする。
            s0 = int(round(f0 / self.outfps * sr_out))
            sL = int(round(f1 / self.outfps * sr_out))
            s1 = int(round((f1 + 1) / self.outfps * sr_out)) if f1 < F - 1 else n_out
            s1 = min(s1, n_out)
            sL = min(sL, n_out - 1)
            if s1 <= s0:
                continue
            vals = traj_1d[f0:f1 + 1]
            if f1 == f0 or sL <= s0:
                up[s0:s1] = vals[-1]
            else:
                core_len = sL - s0 + 1
                if smooth == "fourier":
                    core = self._fourier_upsample(vals, core_len - 1, n_harmonics)
                    up[s0:sL] = core
                    up[sL] = vals[-1]
                else:
                    up[s0:sL + 1] = self._spline_upsample(vals, core_len)
                up[sL + 1:s1] = vals[-1]
            if s0 > 0:
                seg_edges.append(s0)
        return up, seg_edges

    @staticmethod
    def _audio_decode(path, sr):
        """ffmpeg で音声ファイルをデコードして float32 (2, N) を返す。"""
        cmd = ["ffmpeg", "-v", "error", "-i", path,
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
        raw = subprocess.check_output(cmd)
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.reshape(-1, 2).T.copy()

    @staticmethod
    def _audio_duration(path):
        """ffprobe で音声ファイルの長さ [秒] を返す。"""
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        return float(subprocess.check_output(cmd).decode().strip())

    _AUDIO_EXTS = (".wav", ".aiff", ".aif", ".aifc", ".flac", ".mp3",
                   ".m4a", ".ogg", ".caf")

    def _has_audio_stream(self, path):
        """ffprobe でファイルに音声ストリームがあるか確認する。"""
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
                   "-show_entries", "stream=codec_type",
                   "-of", "default=noprint_wrappers=1:nokey=1", path]
            return subprocess.check_output(cmd).decode().strip() == "audio"
        except Exception:
            return False

    def _resolve_audio_path(self, audio_path, for_sc=False):
        """音声入力の解決。

        探索順: 明示指定 → 映像と同名の音声ファイル (旧来の 同名.AIFF 規約)
                → 元映像ファイルの音声トラック
        for_sc=True の場合、SC の Buffer.read は動画コンテナを読めないため
        動画から音声を WAV に抽出してそのパスを返す。
        """
        if audio_path is None:
            # 旧来の規約: 映像と同名の音声ファイルを探す (.AIFF 優先)
            base = self.ORG_PATH + "/" + self.ORG_NAME
            for ext in (".AIFF", ".aiff", ".aif", ".wav", ".WAV"):
                if os.path.isfile(base + ext):
                    audio_path = base + ext
                    break
            if audio_path is None:
                # 無ければ元映像の音声トラックを直接使う
                if os.path.isfile(self.VIDEO_PATH) and self._has_audio_stream(self.VIDEO_PATH):
                    print("同名の音声ファイルが無いため元映像の音声を使用:", self.VIDEO_PATH)
                    audio_path = self.VIDEO_PATH
                else:
                    raise ValueError(
                        "音声入力が見つかりません。" + base + ".AIFF を用意するか、"
                        "音声トラック付きの映像を使うか、audio_path を指定してください。")
        if for_sc and not audio_path.lower().endswith(self._AUDIO_EXTS):
            # SC 用: 動画コンテナから音声を WAV に抽出する (カレント=出力ディレクトリ)
            wav_name = self.ORG_NAME + "_scsrc.wav"
            if not os.path.isfile(wav_name):
                print("SC 用に音声を抽出:", wav_name)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio_path,
                                "-vn", "-acodec", "pcm_f32le", "-ac", "2", wav_name],
                               check=True)
            audio_path = os.path.abspath(wav_name)
        return audio_path

    def _audio_out_basename(self):
        base = self.ORG_NAME + "_" + self.out_name_attr
        if self.embedHistory_intoName == False:
            base = self.ORG_NAME + "_process" + str(self.log)
        return base

    # ------------------------------------------------------------------
    # Python 内オフラインレンダリング (SuperCollider 不要)
    # ------------------------------------------------------------------
    def audio_render(self, thread_num=20, audio_path=None, mode="play",
                     smooth="fourier", n_harmonics=None, sr=48000,
                     grain_dur=0.1, grain_rate=None, inpan_mode="balance",
                     jump_thresh_sec=0.25, fade_sec=0.01, normalize=True,
                     out_name=None):
        """Maneuver Data を音声に適用してステレオ WAV を書き出す。

        mode:
          "play"  : 可変速再生 (ピッチは速度に追従。旧 SC_Play 相当)
          "grain" : グラニュラー合成 (ピッチ保存。旧 SC_Grain 相当)
        smooth:
          "fourier" : 帯域制限補間 (周波数成分の重ね合わせによる再構成)
          "spline"  : キュービックスプライン補間
        n_harmonics : fourier 時に残す上位周波数成分数 (None=全成分)
        inpan_mode  : inPan (読み取り位置の空間座標 0..1) の適用方法
          "balance" : ステレオバランス。読み取り位置が左端(0)に近いほど
                      ソースLチャンネルが大きく、右端(1)に近いほど
                      Rチャンネルが大きくなる (本来の設計意図)
          "gain"    : 旧 SC コード互換。inPan を両チャンネル共通の
                      ゲインとして適用 (左端で無音に近づく)
          "none"    : inPan を使わずボイス位置の固定パンのみ
        jump_thresh_sec : この値 [秒] を超える軌跡の飛びを不連続点として
                          セグメント分割し、クロスフェードでクリックを防ぐ
        """
        print(sys._getframe().f_code.co_name)
        audio_path = self._resolve_audio_path(audio_path)
        src = self._audio_decode(audio_path, sr)  # (2, N)

        traj_sec, inpan_traj = self._audio_trajectories(thread_num)
        F, n_voices = traj_sec.shape
        n_out = int(round(F / self.outfps * sr))
        out = np.zeros((2, n_out), dtype=np.float64)

        if grain_rate is None:
            grain_rate = float(self.outfps)
        fade_n = max(int(fade_sec * sr), 8)

        for v in range(n_voices):
            pos_sec, seg_edges = self._upsample_traj(
                traj_sec[:, v], sr, smooth, n_harmonics, jump_thresh_sec)
            pos = pos_sec * sr  # ソース内サンプル位置

            if mode == "grain":
                sig = self._render_grain(src, pos, sr, grain_dur, grain_rate)
            else:
                sig = self._render_play(src, pos)

            # セグメント境界のクロスフェード (ジャンプによるクリック防止)
            env = np.ones(n_out)
            for e in seg_edges:
                a0, a1 = max(e - fade_n, 0), e
                b0, b1 = e, min(e + fade_n, n_out)
                env[a0:a1] = np.minimum(env[a0:a1], np.linspace(1, 0, a1 - a0))
                env[b0:b1] = np.minimum(env[b0:b1], np.linspace(0, 1, b1 - b0))
            # 全体の頭とお尻にも短いフェード
            env[:fade_n] *= np.linspace(0, 1, fade_n)
            env[-fade_n:] *= np.linspace(1, 0, fade_n)
            sig *= env

            pan = v / (n_voices - 1) if n_voices > 1 else 0.5
            amp = 1.0 / n_voices
            if inpan_mode == "none":
                out[0] += (sig[0] * pan + sig[1] * (1 - pan)) * amp
                out[1] += (sig[0] * (1 - pan) + sig[1] * pan) * amp
            else:
                p = np.interp(np.arange(n_out) / sr * self.outfps,
                              np.arange(F), inpan_traj[:, v])
                if inpan_mode == "gain":
                    # 旧 SC コード互換: inPan を共通ゲインとして適用
                    sig *= p
                    out[0] += (sig[0] * pan + sig[1] * (1 - pan)) * amp
                    out[1] += (sig[0] * (1 - pan) + sig[1] * pan) * amp
                else:
                    # balance: 読み取り位置 0=L / 1=R のステレオバランス (等パワー)
                    out[0] += sig[0] * np.cos(p * np.pi / 2) * amp
                    out[1] += sig[1] * np.sin(p * np.pi / 2) * amp
            print(f"voice {v + 1}/{n_voices} rendered")

        if normalize:
            peak = np.max(np.abs(out))
            if peak > 0:
                out *= 0.891 / peak  # -1 dBFS

        base = out_name if out_name else self._audio_out_basename()
        harm_attr = f"-h{n_harmonics}" if n_harmonics else ""
        fname = f"{base}_PyAudio-{mode}-{smooth}{harm_attr}-{n_voices}voices.wav"
        wavfile.write(fname, sr, out.T.astype(np.float32))
        print("audio_render out:", fname)
        return fname

    @staticmethod
    def _render_play(src, pos):
        """位置駆動の可変速再生 (線形補間読み出し)。src:(2,N), pos:(n_out,)"""
        n_src = src.shape[1]
        p = np.clip(pos, 0, n_src - 1.000001)
        i0 = p.astype(np.int64)
        frac = p - i0
        sig = np.empty((2, len(pos)), dtype=np.float64)
        for ch in range(2):
            sig[ch] = src[ch, i0] * (1 - frac) + src[ch, i0 + 1] * frac
        # ソース範囲外は無音
        outside = (pos < 0) | (pos > n_src - 1)
        sig[:, outside] = 0.0
        return sig

    @staticmethod
    def _render_grain(src, pos, sr, grain_dur, grain_rate):
        """位置駆動のグラニュラー合成 (ピッチ保存)。"""
        n_src = src.shape[1]
        n_out = len(pos)
        gl = max(int(grain_dur * sr), 32)
        win = np.hanning(gl)
        hop = sr / grain_rate
        sig = np.zeros((2, n_out + gl), dtype=np.float64)
        # 窓の重なりによるゲイン補正
        overlap = gl / hop
        norm = 1.0 / max(overlap * 0.5, 1.0)
        t = 0.0
        while t < n_out:
            s = int(t)
            p0 = int(pos[s])
            if 0 <= p0 <= n_src - gl:
                sig[:, s:s + gl] += src[:, p0:p0 + gl] * win
            t += hop
        return sig[:, :n_out] * norm

    # ------------------------------------------------------------------
    # 音声つき映像の統合書き出し (Python 内で完結)
    # ------------------------------------------------------------------
    def audio_video_out(self, thread_num=20, videopath=None, rendered_audio=None,
                        **audio_kwargs):
        """レンダリング済み映像と音声を統合し、音声つき映像として書き出す。

        videopath      : 結合する映像。None なら直近にレンダリングした
                         self.out_videopath (transprocess 系の出力) を使う
        rendered_audio : レンダリング済み音声 WAV のパス。None なら
                         audio_render() をここで実行する
        audio_kwargs   : audio_render に渡す引数
                         (mode="play"/"grain", smooth, n_harmonics,
                          inpan_mode, audio_path など)

        映像ストリームは再エンコードせずコピーするため画質劣化はない。
        音声は .mov には PCM 24bit、それ以外 (.mp4 等) には AAC 320k で載せる。
        """
        print(sys._getframe().f_code.co_name)
        if videopath is None:
            videopath = self.out_videopath
        if not videopath or not os.path.isfile(videopath):
            raise ValueError(
                "結合する映像が見つかりません。transprocess 等で映像を書き出してから"
                "呼ぶか、videopath を指定してください。")
        if rendered_audio is None:
            rendered_audio = self.audio_render(thread_num=thread_num, **audio_kwargs)

        base, ext = os.path.splitext(videopath)
        out_path = base + "_wAudio" + ext
        if ext.lower() == ".mov":
            acodec = ["-c:a", "pcm_s24le"]
        else:
            acodec = ["-c:a", "aac", "-b:a", "320k"]
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-i", videopath, "-i", rendered_audio,
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"] + acodec
        # HEVC を mp4 へ再多重化すると hev1 タグに変わり QuickTime 非互換に
        # なることがあるため hvc1 を明示する
        if ext.lower() == ".mp4":
            try:
                codec = subprocess.check_output(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_name",
                     "-of", "default=noprint_wrappers=1:nokey=1", videopath]
                ).decode().strip()
                if codec == "hevc":
                    cmd += ["-tag:v", "hvc1"]
            except Exception:
                pass
        # -shortest はストリームコピー時に不正確なため、短い方の尺で明示的に切る
        try:
            t = min(self._audio_duration(videopath),
                    self._audio_duration(rendered_audio))
            cmd += ["-t", f"{t:.6f}"]
        except Exception:
            cmd += ["-shortest"]
        cmd += [out_path]
        subprocess.run(cmd, check=True)
        print("audio_video_out:", out_path)
        return out_path

    # ------------------------------------------------------------------
    # SuperCollider 用 改良版出力 (コントロールトラック方式)
    # ------------------------------------------------------------------
    def scd_out_v2(self, thread_num=20, audio_path=None, ctl_rate=2400,
                   smooth="fourier", n_harmonics=None, inpan_mode="balance",
                   jump_thresh_sec=0.25, grain_dur=0.1, amp=None):
        """位置駆動方式の SuperCollider コードを書き出す (クリック/非同期対策版)。

        旧 scd_out との違い:
          - CSV + 30Hz の .set ループを廃止し、滑らかに補間済みの再生位置を
            多チャンネル float32 WAV (コントロールトラック) として書き出す
          - SC 側は BufRd.ar でコントロールトラックを読み、その位置で
            ソースバッファを直接参照する (位置駆動)。rate 積分やリセット
            トリガが無いためクリックが出ず、周期データは厳密に再同期する

        inpan_mode: "balance" (読み取り位置 0..1 でソース L/R バランス、設計意図)
                    / "gain" (旧 SC 互換の共通ゲイン) / "none" (固定パンのみ)
        """
        print(sys._getframe().f_code.co_name)
        audio_path = self._resolve_audio_path(audio_path, for_sc=True)
        src_dur = self._audio_duration(audio_path)

        traj_sec, inpan_traj = self._audio_trajectories(thread_num)
        F, n = traj_sec.shape
        total_dur = F / self.outfps
        n_ctl = int(round(F / self.outfps * ctl_rate))

        # ctl track: ch 0..n-1 = 再生位置 (0..1) / ch n..2n-1 = inPan (0..1)
        ctl = np.zeros((n_ctl, n * 2), dtype=np.float32)
        for v in range(n):
            pos_sec, _ = self._upsample_traj(
                traj_sec[:, v], ctl_rate, smooth, n_harmonics, jump_thresh_sec)
            ctl[:, v] = np.clip(pos_sec / src_dur, 0.0, 1.0)
            ctl[:, n + v] = np.interp(
                np.arange(n_ctl) / ctl_rate * self.outfps,
                np.arange(F), inpan_traj[:, v])

        base = self._audio_out_basename()
        ctl_name = f"{base}_SCv2ctl-{n}voices.wav"
        wavfile.write(ctl_name, int(ctl_rate), ctl)

        now_dir = os.getcwd()
        ctl_path = now_dir + "/" + ctl_name
        if amp is None:
            amp = 1.0 / n

        pan_exprs = ",".join(
            str(v / (n - 1)) if n > 1 else "0.5" for v in range(n))

        common = {
            "n": n, "nch": n * 2, "amp": amp,
            "audio": audio_path, "ctl": ctl_path,
            "dur": total_dur, "pans": pan_exprs,
            "grain_dur": grain_dur, "grain_rate": float(self.outfps),
            "rec_dir": now_dir, "base": base,
        }

        play_scd = f"{base}_SCv2_Play-{n}voices.scd"
        with open(play_scd, "w") as f:
            f.write(self._scd_v2_template("play", common, inpan_mode))
        grain_scd = f"{base}_SCv2_Grain-{n}voices.scd"
        with open(grain_scd, "w") as f:
            f.write(self._scd_v2_template("grain", common, inpan_mode))
        print("scd_out_v2 out:", ctl_name, play_scd, grain_scd)
        return [ctl_name, play_scd, grain_scd]

    @staticmethod
    def _scd_v2_template(mode, p, inpan_mode="balance"):
        """位置駆動 SC コードの生成。mode: 'play' | 'grain'"""
        # inPan の適用方法ごとのミックス行 (sL/sR = 各ボイスの L/R 信号)
        def mix_lines(sL, sR):
            if inpan_mode == "balance":
                return (
                    '        mixL = mixL + (' + sL + ' * (g * 0.5pi).cos);\n'
                    '        mixR = mixR + (' + sR + ' * (g * 0.5pi).sin);\n'
                )
            elif inpan_mode == "gain":
                return (
                    '        mixL = mixL + (((' + sL + ' * pans[i]) + (' + sR + ' * (1 - pans[i]))) * g);\n'
                    '        mixR = mixR + (((' + sL + ' * (1 - pans[i])) + (' + sR + ' * pans[i])) * g);\n'
                )
            else:
                return (
                    '        mixL = mixL + ((' + sL + ' * pans[i]) + (' + sR + ' * (1 - pans[i])));\n'
                    '        mixR = mixR + ((' + sL + ' * (1 - pans[i])) + (' + sR + ' * pans[i]));\n'
                )

        if mode == "play":
            synth = (
                'SynthDef(\\imgtransPlayV2, {{ arg out=0, snd, ctl, amp={amp};\n'
                '    var phase = Phasor.ar(0, BufRateScale.kr(ctl), 0, BufFrames.kr(ctl));\n'
                '    var c = BufRd.ar({nch}, ctl, phase, 0, 4);\n'
                '    var pans = [{pans}];\n'
                '    var mixL = 0, mixR = 0;\n'
                '    {n}.do {{ arg i;\n'
                '        var pos = c[i] * (BufFrames.kr(snd) - 1);\n'
                '        var g = c[{n} + i];\n'
                '        var v = BufRd.ar(2, snd, pos, 0, 4);\n'
                + mix_lines('v[0]', 'v[1]') +
                '    }};\n'
                '    Line.kr(0, 1, {dur}, doneAction: 2);\n'
                '    Out.ar(out, [mixL, mixR] * amp);\n'
                '}}).add;\n'
            ).format(**p)
            boot_bufs = (
                '    ~snd = Buffer.read(s, "{audio}");\n'
                '    ~ctl = Buffer.read(s, "{ctl}");\n'
            ).format(**p)
            synth_args = '[\\snd, ~snd, \\ctl, ~ctl]'
            synth_name = '\\imgtransPlayV2'
            rec_name = '{rec_dir}/{base}_SCv2_Play.aiff'.format(**p)
        else:
            synth = (
                'SynthDef(\\imgtransGrainV2, {{ arg out=0, sndL, sndR, ctl, amp={amp};\n'
                '    var phase = Phasor.ar(0, BufRateScale.kr(ctl), 0, BufFrames.kr(ctl));\n'
                '    var c = BufRd.ar({nch}, ctl, phase, 0, 4);\n'
                '    var pans = [{pans}];\n'
                '    var mixL = 0, mixR = 0;\n'
                '    {n}.do {{ arg i;\n'
                '        var pos = c[i];\n'
                '        var g = c[{n} + i];\n'
                '        var trig = Impulse.ar({grain_rate});\n'
                '        var vL = GrainBuf.ar(1, trig, {grain_dur}, sndL, 1, pos, 2, 0, -1, 512);\n'
                '        var vR = GrainBuf.ar(1, trig, {grain_dur}, sndR, 1, pos, 2, 0, -1, 512);\n'
                + mix_lines('vL', 'vR') +
                '    }};\n'
                '    Line.kr(0, 1, {dur}, doneAction: 2);\n'
                '    Out.ar(out, [mixL, mixR] * amp);\n'
                '}}).add;\n'
            ).format(**p)
            boot_bufs = (
                '    ~sndL = Buffer.readChannel(s, "{audio}", channels: [0]);\n'
                '    ~sndR = Buffer.readChannel(s, "{audio}", channels: [1]);\n'
                '    ~ctl = Buffer.read(s, "{ctl}");\n'
            ).format(**p)
            synth_args = '[\\sndL, ~sndL, \\sndR, ~sndR, \\ctl, ~ctl]'
            synth_name = '\\imgtransGrainV2'
            rec_name = '{rec_dir}/{base}_SCv2_Grain.aiff'.format(**p)

        return (
            '// imgtrans scd_out_v2 (位置駆動・コントロールトラック方式)\n'
            '// 旧版と異なり CSV/setループ/再トリガを使わないため、\n'
            '// クリックノイズが出ず周期マニューバは厳密に再同期します。\n'
            '(\n'
            '// 必要に応じて出力デバイスを設定:\n'
            '// d = ServerOptions.devices[4]; Server.default.options.outDevice_(d);\n'
            's.options.numOutputBusChannels = 2;\n'
            's.waitForBoot({\n'
            + boot_bufs +
            '    s.sync;\n'
            + '    ' + synth.replace('\n', '\n    ').rstrip() + '\n'
            '    s.sync;\n'
            '    s.record("' + rec_name + '", duration: ' + str(p["dur"] + 1) + ');\n'
            '    Synth(' + synth_name + ', ' + synth_args + ');\n'
            '});\n'
            ')\n'
        )

    # ------------------------------------------------------------------
    # フーリエ成分 (周波数の重ね合わせ) としての書き出し
    # ------------------------------------------------------------------
    def maneuver_fourier_out(self, thread_num=20, n_harmonics=64):
        """各ボイスの時間軌跡をフーリエ成分の集合として CSV 出力する。

        行形式: voice, kind, a, b, c
          kind=trend    : a=切片[秒], b=傾き[秒/秒] (出力タイムライン基準)
          kind=harmonic : a=周波数[Hz], b=振幅[秒], c=位相[rad]
        pos(t) = trend + Σ amp * cos(2π f t + phase) で再構成できる。
        SuperCollider / Swift 等でオシレータバンクとして利用可能。
        """
        print(sys._getframe().f_code.co_name)
        traj_sec, _ = self._audio_trajectories(thread_num)
        F, n = traj_sec.shape
        rows = []
        for v in range(n):
            vals = traj_sec[:, v]
            t = np.arange(F)
            trend0 = vals[0]
            slope = (vals[-1] - vals[0]) / max(F - 1, 1)
            resid = vals - (trend0 + slope * t)
            m = max(F - 1, 1)
            spec = np.fft.rfft(resid[:m])
            mags = np.abs(spec[1:])
            order = np.argsort(mags)[::-1][:n_harmonics] + 1
            # trend: 出力タイムライン t[秒] 基準に変換
            rows.append([v, 0, trend0 + np.real(spec[0]) / m, slope * self.outfps, 0.0])
            period_sec = m / self.outfps
            for k in sorted(order):
                amp = 2.0 * np.abs(spec[k]) / m
                if amp <= 1e-12:
                    continue
                freq = k / period_sec
                phase = np.angle(spec[k])
                rows.append([v, k, freq, amp, phase])
        base = self._audio_out_basename()
        fname = f"{base}_Fourier-{n}voices.csv"
        np.savetxt(fname, np.array(rows, dtype=np.float64), delimiter=",",
                   header="voice,kind(0=trend/k=harmonic),freqHz_or_intercept,amp_or_slope,phase",
                   comments="")
        print("maneuver_fourier_out out:", fname)
        return fname
