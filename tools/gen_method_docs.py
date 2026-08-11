"""README のメソッド解説に 3D プロット GIF を自動生成して追記するツール。

DrawBeautifulManeuver.py の ⓘ (Info) ボタンは README の
「## `メソッド名`」セクションをそのまま引用して表示する。図があると
理解が早いため、図の無いメソッドについて 3D プロットを生成し、
README / README_JA の該当セクション末尾へ画像参照を追記する。

使い方:
    python tools/gen_method_docs.py            # 図の無いメソッドすべて
    python tools/gen_method_docs.py addFreeze  # 指定メソッドのみ (再生成)
    python tools/gen_method_docs.py --list     # 対象一覧の確認のみ

生成物: images/doc_<メソッド名>_3dplot.gif
図はソース映像の内容ではなく「軌道の形」を示すものなので、
入力にはテストパターン映像を使う (無ければ ffmpeg で生成する)。
"""

import os
import re
import sys
import glob
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from imgtrans import drawManeuver  # noqa: E402

IMAGES = REPO / "images"
READMES = [REPO / "README_JA.md", REPO / "README.md"]
SAMPLE = REPO / "images" / "_doc_source.mp4"

# 3D プロットの設定 (README 用。既存の doc_*.gif と見た目を揃える)
PLOT_FRAMES = 36
PLOT_FPS = 12
PLOT_DPI = 90
GIF_WIDTH = 480

# --- 各メソッドのデモ手順 ---
# (前段のチェーン, 対象メソッドの引数)。前段は「その効果が分かる下地」。
BASE_FLAT = [("addFlat", dict(frame_nums=60))]
BASE_TRANS = [("addFlat", dict(frame_nums=30)),
              ("addTrans", dict(frame_nums=120))]
BASE_CYCLE = [("addFlat", dict(frame_nums=30)),
              ("addCycleTrans", dict(frame_nums=150, cycle_degree=360))]

DEMOS = {
    # ---- Add 系 ----
    "addSlicePlane": ([], dict(frame_nums=1)),
    "addFreeze": (BASE_TRANS, dict(frame_nums=90)),
    "addExtend": (BASE_TRANS, dict(addframe=90)),
    "preExtend": (BASE_TRANS, dict(addframe=90)),
    # 速度を引き継ぐので、下地に動きのあるチェーンが要る
    "addKeepSpeedTrans": (BASE_TRANS, dict(frame_nums=120)),
    "addBoxUnfoldCut": ([], dict(center_time=100)),
    "addWideKeyframeTrans": ([], dict(frame_nums=150,
                                      key_array=[[0, 0], [150, 300]],
                                      wide_scale=3)),
    "addFixWideCycleTrans": ([], dict(frame_nums=150, cycle_degree=360,
                                      wide_scale=3)),
    "rootingB_interporation": ([], dict(FRAME_NUMS=200, loop_num=1)),
    "rootingA_interporation_RANDOM": ([], dict(FRAME_NUMS_range=(200, 260),
                                               loop_num=2, seed=3)),
    # ---- Apply 系 ----
    "applyTimeBlur": (BASE_CYCLE, dict(bl_time=60)),
    "applySpaceBlur": (BASE_CYCLE, dict(bl_time=60)),
    "applyLoopBlur": (BASE_CYCLE, dict(sblur=40, tblur=40)),
    "applyConnectLoopBlur": (BASE_CYCLE, dict(sblur=40, tblur=40,
                                              connect_frame=60)),
    "applyPointBlur": (BASE_CYCLE, dict(point_frame=90, sblur=40, tblur=40,
                                        range_frame=80)),
    "applyCustomeBlur": (BASE_CYCLE, dict(s_frame=40, e_frame=140, bl_time=60)),
    "applyTimeSlide": (BASE_CYCLE, dict(settime=200)),
    "applyInOutGapFix": (BASE_CYCLE, dict()),
    "applySpaceFlip": (BASE_CYCLE, dict()),
    "applySpaceFlat": (BASE_CYCLE, dict()),
    "applyTimeFlowKeepingExtend": (BASE_TRANS, dict(frame_nums=90)),
    # ---- データ操作 ----
    "arrayReflection": (BASE_TRANS, dict()),
    "arrayExtract": (BASE_CYCLE, dict(start=30, end=140)),
    "zCenterArange": (BASE_CYCLE, dict(center_time_frame=150)),
    "zArange": (BASE_CYCLE, dict(target_frame=0, center_time_frame=150)),
    "zStartArange": (BASE_CYCLE, dict()),
    "zPointCheck": ([("addFlat", dict(frame_nums=30)),
                     ("addCycleTrans", dict(frame_nums=150, cycle_degree=360,
                                            zslide=-400))], dict()),
    "wide_expandB": (BASE_CYCLE, dict(add_size=240)),
    # ---- 第2弾: 軌道を作る/整える残りのメソッド ----
    "addInterpolation": ([("addFlat", dict(frame_nums=30))],
                         dict(frame_nums=150, i_direction=0, z_direction=0,
                              axis_position=0)),
    "interpolation": ([("addFlat", dict(frame_nums=30))],
                      dict(frame_nums=150, i_direction=1, z_direction=1,
                           axis_position=0.5)),
    "rooting4C_interporation": ([], dict(FRAME_NUMS=240)),
    "timeFlowKeepingExtend": (BASE_TRANS, dict(frame_nums=90)),
    "zPointCheckandReflect": ([("addFlat", dict(frame_nums=30)),
                               ("addCycleTrans", dict(frame_nums=150,
                                                      cycle_degree=360,
                                                      zslide=-400))], dict()),
    "applyTimeFlowKeepingExtend_CoodinateBase_Intro": (
        BASE_TRANS, dict(target_z=0, num_frames=90)),
    "applyTimeFlowKeepingExtend_CoodinateBase_Outtro": (
        BASE_TRANS, dict(target_z=0, num_frames=90)),
    "dataCheck": (BASE_CYCLE, dict()),
}


def ensure_sample_video():
    """図生成用のテストパターン映像 (無ければ ffmpeg で作る)。"""
    if SAMPLE.exists():
        return str(SAMPLE)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg が必要です")
    IMAGES.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=640x360:rate=30:duration=20",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         str(SAMPLE)], check=True)
    return str(SAMPLE)


def has_image(method):
    """README (JA) の該当セクションに画像参照があるか。"""
    txt = READMES[0].read_text(encoding="utf-8")
    m = re.search(r"^## `" + re.escape(method) + r"`$(.*?)(?=^## |\Z)",
                  txt, re.M | re.S)
    return bool(m and "![" in m.group(1))


def section_exists(path, method):
    txt = path.read_text(encoding="utf-8")
    return re.search(r"^## `" + re.escape(method) + r"`$", txt, re.M) is not None


def generate_gif(method, chain, kwargs, video):
    """デモチェーンを実行して 3D プロット GIF を作り、images/ へ置く。"""
    out_gif = IMAGES / f"doc_{method}_3dplot.gif"
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="docgen_")
    try:
        os.chdir(tmp)
        dm = drawManeuver(videopath=video, sd=False)   # 横スリット
        dm.auto_visualize_out = False
        dm.outfps = 30
        for name, kw in chain:
            getattr(dm, name)(**kw)
        getattr(dm, method)(**kwargs)
        if not len(getattr(dm, "data", [])):
            raise RuntimeError("data が空")
        dm.zPointCheck()
        ts = time.time() - 1
        dm.maneuver_3dplot(out_framenums=PLOT_FRAMES, out_fps=PLOT_FPS,
                           dpi=PLOT_DPI)
        IMAGES.mkdir(exist_ok=True)
        mp4s = [p for p in glob.glob(os.path.join(os.getcwd(), "**", "*.mp4"),
                                     recursive=True)
                if os.path.getmtime(p) >= ts]
        if mp4s:
            mp4 = max(mp4s, key=os.path.getmtime)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", mp4,
                 "-vf", f"fps={PLOT_FPS},scale={GIF_WIDTH}:-1:flags=lanczos",
                 "-loop", "0", str(out_gif)], check=True)
            return out_gif
        # 1フレームしか無いメソッドは maneuver_3dplot が PNG を出力する
        pngs = [p for p in glob.glob(os.path.join(os.getcwd(), "*.png"))
                if os.path.getmtime(p) >= ts and "3dPlot" in os.path.basename(p)]
        if not pngs:
            raise RuntimeError("3dplot の出力が見つからない")
        out_png = IMAGES / f"doc_{method}_3dplot.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", max(pngs, key=os.path.getmtime),
             "-vf", f"scale={GIF_WIDTH}:-1:flags=lanczos", str(out_png)],
            check=True)
        return out_png
    finally:
        plt.close("all")
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def append_image_to_readme(path, method, gif_name, caption):
    """該当セクションの末尾へ画像参照を追記する (既にあれば何もしない)。"""
    txt = path.read_text(encoding="utf-8")
    pat = re.compile(r"(^## `" + re.escape(method) + r"`$.*?)(?=^## |\Z)",
                     re.M | re.S)
    m = pat.search(txt)
    if not m:
        return False
    body = m.group(1)
    if f"images/{gif_name}" in body:
        return False
    new_body = body.rstrip() + f"\n\n![{caption}](images/{gif_name})\n\n"
    path.write_text(txt[:m.start(1)] + new_body + txt[m.end(1):],
                    encoding="utf-8")
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only_list = "--list" in sys.argv

    targets = args or [m for m in DEMOS if not has_image(m)]
    targets = [m for m in targets if m in DEMOS]

    print(f"対象 {len(targets)} メソッド: {', '.join(targets)}")
    if only_list:
        return

    video = ensure_sample_video()
    ok, ng = [], []
    for i, method in enumerate(targets, 1):
        chain, kwargs = DEMOS[method]
        print(f"[{i}/{len(targets)}] {method} …", flush=True)
        try:
            gif = generate_gif(method, chain, kwargs, video)
            for path, cap in ((READMES[0], f"{method}（3dプロット例）"),
                              (READMES[1], f"{method} (3D plot example)")):
                if section_exists(path, method):
                    append_image_to_readme(path, method, gif.name, cap)
            ok.append(method)
            print(f"    → images/{gif.name}")
        except Exception as e:
            ng.append((method, f"{type(e).__name__}: {e}"))
            print(f"    ✗ {type(e).__name__}: {e}")

    print(f"\n完了: 成功 {len(ok)} / 失敗 {len(ng)}")
    for m, e in ng:
        print(f"  ✗ {m}: {e}")


if __name__ == "__main__":
    main()
