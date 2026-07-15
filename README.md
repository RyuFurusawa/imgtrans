# imgtrans

A programming tool for manipulating time and space in video data. It's primarily a Python library centered around the `drawManeuver` class.

## Table of Contents

- [Project Overview](#project-overview)
- [Background](#background)
- [Installation](#installation)
- [Usage Flow](#usage-flow)
  - [1. Library Import](#1-library-import)
  - [2. Initialization of drawManeuver Class](#2-initialization-of-drawmaneuver-class)
    - [Direction of the Slit](#slit-direction)
  - [3. Maneuver Design](#3-maneuver-design)
    - [1. Main Methods to Add Spatiotemporal Movement](#1-main-methods-to-add-spatiotemporal-movement)
    - [2. Main Methods to Adapt Flow of Time](#2-main-methods-adapting-the-flow-of-time)
    - [Examples of Combinations](#examples-of-combinations)
    - [Saving and Loading Maneuver Data](#saving-and-loading-maneuver-data)
  - [4. Visualization](#4-visualization)
    - [Slit color](#slit-color)
    - [2D Plot](#2d-plot)
    - [3D Plot](#3d-plot)
  - [5. Rendering](#5-rendering)
    - [Structure of data](#structure-of-data)
    - [Video Rendering](#video-rendering)
    - [Color Pipeline in Video Rendering](#color-pipeline-in-video-rendering)
    - [Audio Rendering](#audio-rendering)
        - [How the audio input is resolved](#how-the-audio-input-is-resolved)
        - [Python-only audio rendering: audio_render](#python-only-audio-rendering-audio_render)
        - [Integrated audio+video output: audio_video_out](#integrated-audiovideo-output-audio_video_out)
        - [Improved SuperCollider output: scd_out_v2](#improved-supercollider-output-scd_out_v2)
        - [Exporting as Fourier components: maneuver_fourier_out](#exporting-as-fourier-components-maneuver_fourier_out)
        - [Legacy scd_out](#legacy-scd_out)
        - [4 CSV of Maneuver Data](#4-csv-data-of-maneuver)
        - [4 SCD Files](#4-scd-files)
        - [Running the scd File](#running-the-scd-file)
          - [1. Audio Settings and Data Loading](#1-audio-settings-and-data-loading)
          - [2. Playing `Synth` and Recording](#2-playing-synth-and-recording)
    - [Combining Audio and Video](#combining-audio-and-video)
- [drawManeuver Class](#drawmaneuver-class)
  - [Class Variables](#class-variables)
  - [initialization](#initialization)
  - [List of All Class Methods](#list-of-all-class-methods)
  - [addTrans](#addtrans)
  - [addBlowupTrans](#addblowuptrans)
  - [addInterpolation](#addinterpolation)
  - [addCycleTrans](#addcycletrans)
  - [addcustomCycleTrans](#addcustomcycletrans)
  - [addWaveTrans](#addwavetrans)
  - [transprocess](#transprocess)
  - [animationout](#animationout)
- [Contribute](#contribute)
- [License](#license)

## Project Overview
By interpreting the time dimension of the image as depth axis of three dimentions, the image data becomes a cube composed of voxels.  
This video data cube stores color information over the entire space and has the plasticity to be retrieved as an image with an array of colors, even when cut from various angles and directions.  
By manipulating the cross-sectional behavior of the video data cube, one can create images ordered in a new time and space, different from the captured time and space.  
In addition, working with this tool will provide new insights into "movement" as defined by time and space, and will encourage us to think about our perception and awareness of time and space.
<div style="text-align:center;">
<img src="images/ost-illustrate20220106_RFS1459.gif" alt="illustrate moving image manipulation by time and space" style="max-width:100%; height:auto;">
</div><br>

## Background
This tool began its development in 2020 for the production of works by the video artist Ryu Furusawa, and several works were produced.
<ul><li><a href="https://ryufurusawa.com/post/711685011289554944/mid-tide-ryu-furusawa-multi-channel-video">Mid Tide,2023 </a></li>
<li><a href="https://ryufurusawa.com/post/661228499174113280/wavesetude">Waves Etude,2020-2022</a></li>
</ul>

## Installation
Before installing this library, please install the following external libraries:
```bash
pip install opencv-python numpy scipy psutil matplotlib av numba
```
- `scipy`: used for trajectory interpolation (interpolate) and for writing WAV files in the audio output (io.wavfile).
- `av` (PyAV): required for decoding HDR / 10bit+ input video and for the YUV-native encode path.
- `numba`: optional but recommended — JIT-accelerates the YUV-native slit-scan kernels. If absent, a pure-NumPy fallback is used.

In addition, the **FFmpeg** suite (both `ffmpeg` and `ffprobe`) must be installed and available on your `PATH`. It is used for input color-metadata detection (`ffprobe`) and for encoding the output video (`ffmpeg`). On macOS: `brew install ffmpeg`. The optional Ultra HDR still export additionally requires the `ultrahdr_app` binary (macOS: `brew install libultrahdr`).

### To install this library:
```bash
pip install git+https://github.com/ryufurusawa/imgtrans.git
```

## Usage Flow
### 1. Library Import
First, import the module.
```
python
import imgtrans
```

### 2. Initialization of drawManeuver Class
Specify the path of the input video and the direction of the slit that will be the basis for calculations to create an instance of the drawManeuver class.
```
# Path of the input video
videopath= '/Users/Movies/20230917_RFS3108_mod-HD720p.mov'

# Create an instance of the drawManeuver class.
# The second variable specifies whether the slit is horizontal or vertical. 0 indicates horizontal slit and 1 indicates vertical slit.
bm=imgtrans.drawManeuver(videopath,1)
```

#### Slit Direction  
In this tool, the slit direction is limited to vertical and horizontal, and it's specified initially.  
The results from future maneuver operations will vary greatly depending on the direction of the slit.
![slit-direction illustration](images/slit-direction.png)
![Alt text](images/slit-direction-transpostion-3dplotanimation.gif)
![Alt text](images/slit-direction-transpostion-rendering.gif)

### 3. Maneuver Design
Design the maneuver of the playback section by combining several class methods.  
Class methods can be broadly divided into **1. Functions that add spatiotemporal movements** and **2. Functions that adapt the flow of time**.  
By running these, the maneuver data is stored in the instance variable `data`.  
Each function internally performs operations on the instance variable `data`, such as adding a new array or multiplying some value to the entire data.
The maneuver data being edited here describes which slit (spatial position, time position) of the output video corresponds to the input video in terms of coordinate transformation.  
For more details, please refer to the structure of [`data`](#structure-of-data).

#### 1. Main Methods to Add Spatiotemporal Movement
- [`addTrans`](#addtrans): Simple replacement of spatial and temporal dimensions.
- [`addBlowupTrans`](#addblowuptrans): Inherits addTrans while operating on the scaling of the temporal dimension.
- [`addInterpolation`](#addinterpolation): Transition of spatiotemporal dimensions.
- [`addCycleTrans`](#addcycletrans): Rotating the playback section around the center line of the screen.
- [`addWaveTrans`](#addwavetrans): Creating a playback section using dynamic wave shapes.
- [`addEventHorizonTrans`](#addeventhorizontrans): The progression speed of time varies between the center and periphery of the screen.

![Alt text](images/Maneuver-examples-3dplot.gif)

#### 2. Main Methods Adapting the Flow of Time
- [`applyTimeForward`](#applytimeforward): Provides forward flow of time to the entire array.
- [`applyTimeOblique`](#applytimeoblique): Shifts time slightly for each slit.
- [`applyTimeForwardAutoSlow`](#applytimeforwardautoslow): Playback starts at rate 1, slows down, and then returns to rate 1.
- [`applyTimeLoop`](#applytimeloop): Imparts a seamless loop structure.
- [`applyTimeClip`](#applytimeclip): Fixes the flow of time for a specified slit to a specified time.
- [`applyTimeBlur`](#applytimeblur): Applies a temporal blur.
![Alt text](images/timeManeuver-examples-3dplot.gif)

#### Examples of Combinations
```python
# Maneuver design
bm.rootingA_interporation(270)
bm.applyTimeLoop(1)
```
![Alt text](images/mixManeuver-examples-3dplot_harf.gif)

#### Saving and Loading Maneuver Data
There are times when you might want to save just the maneuver data to edit in another software, to render the video later on, or to render multiple videos using the same maneuver data.

##### Saving Maneuver Data
It will be saved in the designated output directory.
```python
bm.data_save()
```

##### Loading Maneuver Data
If initializing.
```python
import numpy as np
bm=imgtrans.drawManeuver(videopath="path/to/video.mp4", sd=1,datapath="path/to/data.npy" )
# Checking the maneuver data
print(bm.data.shape)
```
For only replacing the `data`.
```python
import numpy as np
bm.data=np.read("path/to/data.npy" )
```
In either case, the loaded video data must be within a coordinate range defined by the size and frame count of the video. For example, if the input video has a resolution of Full HD (1920x1080), and the reference vertical slit's horizontal position is 2000, an error will occur.

### 4. Visualization
This function provides a visual representation of the instance variable `data` for easier understanding.
The `data` is  described the movement of slits for the resolution of the output video. For example, the number of vertical slits at 4k resolution is 3840, but in the graph it is reduced to 20 for better clarity.
The data can be visualized in two ways: a 2D plot and a 3D plot.  
The 3D graph provides an intuitive view of the overall movement of the trajectory. On the other hand, the details of time flow can be more clearly understood by looking at a 2D graph.  
By combining these two methods, it is possible to design detailed time movements, for example, time moves backward on the left side of the screen and time moves forward on the right side.  
The image data of the visualization is stored in an export directory generated on the same path as the input video.

#### Slit Color
In some cases, the spatial direction of the video image may be inverted, and the slits in the visualization are drawn using a green-red gradient to clearly indicate the direction of such spatial dimension.
1. For vertical slits, green corresponds to the leftmost (0px) and red to the rightmost (3839px for 4k) output position.
1. In case of horizontal slit, green corresponds to the top (0px) and red corresponds to the bottom (2159px for 4k).
! [slit-direction illustration](images/slit-direction.png)

#### 2D Plot
The 2D maneuver graph is output sequentially every time a maneuver design operation is performed.
This graph displays the following three elements in a single figure, with the output video's time on the x-axis.
1. Movement in the spatial direction
2. Movement in the time direction
3. Playback rate of movement in the time direction

```python
bm.maneuver_2dplot()
```
By default, 20 slits are generated. By changing the first argument, thread_num, of the maneuver_2dplot method, you can adjust the number of slits displayed.
```python
# Drawing 50 slits
bm.maneuver_2dplot(50)
```

Combining the maneuver design code and the 2D plot of that maneuver data.
```python
# 1. Sequentially connecting the modules for spatiotemporal movement design.
# Add 100 frames of normal state
bm.addFlat(100)
# Rotate 90 degrees around the left edge of the video frame from the normal state.
bm.addInterpolation(100,0,1)
# Add 100 frames of a maneuver that swaps time and space.
bm.addTrans(100)

# 2. Combine modules related to time behavior.
# Move the entire maneuver forward one frame at a time.
bm.applyTimeForward(1)
# Apply blur to the entire maneuver's movement in the time direction to smooth the transition.
bm.applyTimeblur(50)

#### Outputting the 2D plot
bm.maneuver_2dplot()
```

![visualized 2dplot image](images/GX010161_2023_0616_Vslit_Flat100+Interpolation300(ID1-ZD0-AP0-REV0)+Freeze30+Transposition300+CycleTrans_addExtend_TimeForward1_TimeBlur30_TimeBlur100_SpaceBlur100_20thread.png)

If you don't need sequential output, please change the class variable's setting.
```python
bm.auto_visualize_out = False
```
#### 3D Plot
To output an animation of the maneuver plot to a 3D graph, you need to specify explicitly.
```python
bm.maneuver_3dplot()
```

![visualized 3dplot gifimage](images/GX010148_2023_0617_Vslit_Flat100+Interpolation300(ID1-ZD0-AP0-REV0)+Freeze30+Transposition300+CycleTrans_addExtend_TimeForward1_TimeBlur30_TimeBlur100_SpaceBlur100_3dPlot.gif)


### 5. Rendering
This section will demonstrate how to reconfigure the spatiotemporal layout of the input video data based on the instance variable `data`, and then render the video.

#### Structure of `data`
The instance variable `data` stores the maneuver data. This section describes the structure of this `data`.  
In this module, we first define whether the slit direction is horizontal or vertical, and access the video data not by pixel but by slit.  
By doing this, access to each slit, which is the smallest unit of the video data, is possible by specifying the two-dimensional coordinates (one-dimensional position (horizontal for vertical slits) and time).  
The maneuver data stored in `data` indicates from which coordinates (one-dimensional position, time position) of the input video's slits each slit of the output video is taken, and it's a map of coordinate conversion.  
Therefore, the color data of each pixel is not stored. It only describes the correspondence of coordinate conversion.  
The data is stored as a three-dimensional NUMPY array that has two channels for each data in the two dimensions of the number of frames of the output video and the number of slits that make up the output video.
1. Slit position of the referenced input video
2. Time position of the input video

Here are some samples to examine `data`:
```python
print("Number of frames in the output video", bm.data.shape[0])
print("Scan count, in the case of vertical slits, pixel width of the output video", bm.data.shape[1])
print("From what time of the input video was the slit on the far right of the first frame of the output video referenced?", bm.data[0,-1,1])
print("Maximum time position to reference from the input video", np.max(bm.data[:,:,1]))

# Plotting the output position transition of the first slit.
plt.plot(bm.data[:,0,0])
# Plotting the input time position transition of the first slit.
plt.plot(bm.data[:,0,2])
```

#### Video Rendering
The rendered video data is stored in the export directory created in the same path as the input video.
```python
bm.transprocess()
```
If you want to export high-resolution and longer videos, you can handle it by splitting the export. Please set it according to your terminal specs. 
Intermediate files temporarily create a tmp directory and save the numpy array data (two-dimensional image data) there.
```python
bm.transprocess(10) # Split and export in 10 parts.
```
You can also set up methods for splitting rendering as an option.  
There must be a tmp directory in the export directory, and the intermediate data must be stored there. Otherwise, an error will occur in the final data integration process.  
In the example below, rendering is performed from the 5th step of 10 steps.
```python
bm.transprocess(10, sep_start_num=5, sep_end_num=10)
```
The `out_type` variable selects the output **codec / format**, not a still-vs-video toggle. `0` exports a series of still images; the other values choose a video encoder:

```python
bm.transprocess(out_type=0)   # still image sequence
bm.transprocess(out_type=1)   # H.264 SDR 8bit .mp4 (default)
bm.transprocess(out_type=3)   # ProRes 422 HQ 10bit .mov (recommended for HDR archival)
```

| `out_type` | Constant | Output |
|---|---|---|
| `0` | `OUT_STILL` | Still image sequence (format set by `imgtype`) |
| `1` | `OUT_H264` | H.264 SDR 8bit `.mp4` (**default**) |
| `2` | `OUT_H265` | H.265/HEVC HDR10 10bit `.mp4` |
| `3` | `OUT_PRORES_422` | ProRes 422 HQ 10bit `.mov` |
| `4` | `OUT_PRORES_4444` | ProRes 4444 10bit `.mov` |
| `5` | `OUT_H265_SDR` | H.265 SDR 10bit `.mp4` |
| `6` | `OUT_PRORES_422_SDR` | ProRes 422 SDR 10bit `.mov` |
| `7` | `OUT_H265_HW` | HEVC hardware encode (VideoToolbox), HDR10 10bit `.mp4` |

> Note: when `out_type=1` (H.264) is requested for HDR/10bit+ input, it is automatically promoted to `out_type=2` (H.265) to preserve the higher bit depth.

For details, refer to [`transprocess`](#transprocess) and the [Output Type Summary](#output-type-summary) table.

#### Color Pipeline in Video Rendering

This section describes the complete color conversion pipeline from source video to rendered output. Understanding this flow is critical for maintaining color accuracy, especially with HDR (PQ/HLG) content in BT.2020 color space.

![Color Pipeline Diagram](images/color_pipeline.svg)

##### Overview

The rendering pipeline has two modes: the **RGB pipeline** (legacy, used for H.265 and HDR transfer conversion) and the **YUV-native pipeline** (default for ProRes 10bit+, bypasses RGB roundtrip for maximum color accuracy). See [YUV-Native Pipeline](#yuv-native-pipeline-rgb-roundtrip-bypass) for details.

The RGB pipeline involves two separate color space conversions:

```
Source Video (YUV 4:2:2 10bit)
    │
    ▼  [PyAV decode]
    │  frame.to_ndarray(format="rgb48le")
    │  YUV → RGB conversion using source video's color matrix
    ▼
RGB 48bit Linear Buffer (numpy uint16 array)
    │
    ▼  [Slit-scan processing]
    │  Pixel remapping based on maneuver data (z-coordinates → source frames)
    │  No color space conversion occurs here — raw RGB values are copied
    ▼
Composited RGB Frame (numpy uint16 array)
    │
    ▼  [Optional: HDR Transfer Conversion]
    │  HLG→PQ or PQ→HLG via EOTF/OETF if force_hdr_mode differs from input
    │  If same transfer: passthrough (no conversion)
    ▼
FFmpeg stdin (rawvideo rgb48le)
    │
    ▼  [FFmpeg encode]
    │  RGB → YUV conversion using specified color matrix
    │  Encode to ProRes / H.265 / etc.
    ▼
Output Video File
```

##### Step 1: Source Frame Reading (YUV → RGB)

For HDR/10bit+ content, frames are read via **PyAV** (not OpenCV):

```python
# imgtrans_lib/_dm_frame_proc.py — PyAV decode path
pyav_fmt = "rgb48le"  # 16bit per channel RGB
img = frame.to_ndarray(format=pyav_fmt)
```

PyAV internally uses libswscale for the YUV→RGB conversion. The conversion matrix is determined by the source video's embedded color metadata (`color_primaries`, `color_trc`, `colorspace`).

For SDR/8bit content, OpenCV is used instead:
```python
# OpenCV decode path
self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(minz))
ret, img = self.cap.read()  # Returns BGR uint8
```

##### Step 2: Slit-Scan Composition

The maneuver `data` array maps each output pixel to a source frame position (z-coordinate). The rendering engine seeks to the corresponding source frame, extracts the slit (column or row), and places it into the output image buffer.

**No color conversion occurs at this stage** — raw RGB/BGR values are directly copied from source frame to output buffer.

##### Step 3: HDR Transfer Function Conversion (Optional)

If `force_hdr_mode` is set and differs from the input transfer function, an explicit EOTF→OETF conversion is applied:

| Input | Output (`force_hdr_mode`) | Conversion |
|-------|---------------------------|------------|
| HLG (`arib-std-b67`) | PQ (`smpte2084`) | `hlg_eotf()` → linear → `pq_oetf()` |
| PQ (`smpte2084`) | HLG (`arib-std-b67`) | `pq_eotf()` → linear → `hlg_oetf()` |
| Same as input | Same | Passthrough (no conversion) |

```python
# Example: HLG input → PQ output
rgb = frame.astype(np.float32) / 65535.0
rgb_lin = self.hlg_eotf(rgb)      # HLG → Linear
rgb_out = self.pq_oetf(rgb_lin)   # Linear → PQ
rgb16 = np.clip(rgb_out * 65535.0, 0, 65535).astype(np.uint16)
```

##### Step 4: FFmpeg Output (RGB → YUV Encode)

The composited RGB frame is piped to FFmpeg via stdin as `rawvideo rgb48le`. FFmpeg performs the final RGB→YUV conversion and encodes to the target codec.

**Critical: input color space metadata must be declared** on the rawvideo input so that FFmpeg uses the correct RGB→YUV conversion matrix. Without this, FFmpeg defaults to BT.709, which produces incorrect colors (especially greens) for BT.2020 content.

The `_build_ffmpeg_cmd` method constructs the FFmpeg command with color metadata on **both input and output sides**:

**ProRes 422/4444 output:**
```
ffmpeg -y
  # --- Input side (declares what the RGB data is) ---
  -f rawvideo
  -pix_fmt rgb48le
  -color_primaries bt2020        ← Tells FFmpeg the RGB is BT.2020
  -color_trc smpte2084           ← Transfer function (PQ or HLG)
  -colorspace bt2020nc           ← YUV conversion matrix to use
  -s:v {width}x{height}
  -r {fps}
  -i -
  # --- Output side (codec + metadata tags) ---
  -c:v prores_ks
  -pix_fmt yuv422p10le
  -profile:v 3                   ← 422 HQ
  -vendor apl0
  -color_primaries bt2020        ← Output metadata tag
  -color_trc smpte2084           ← Output metadata tag
  -colorspace bt2020nc
  -color_range tv
  output.mov
```

**H.265 (HEVC) HDR output:**
```
ffmpeg -y
  # --- Input side ---
  -f rawvideo
  -pix_fmt rgb48le
  -color_primaries bt2020
  -color_trc smpte2084
  -colorspace bt2020nc
  -s:v {width}x{height}
  -r {fps}
  -i -
  # --- Output side ---
  -c:v libx265
  -pix_fmt yuv420p10le
  -tag:v hvc1
  -x265-params hdr-opt=1:repeat-headers=1:
    colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:
    master-display=G(13250,34500)B(7500,3000)R(34000,16000)
    WP(15635,16450)L(10000000,50):max-cll=1000,400
  output.mp4
```

##### Color Metadata Detection

On initialization, `drawManeuver` automatically detects the source video's color metadata via `ffprobe`:

```python
self.input_color_primaries  # e.g., "bt2020"
self.input_transfer         # e.g., "smpte2084" (PQ) or "arib-std-b67" (HLG)
self.input_colorspace       # e.g., "bt2020nc"
self.inputmovfps            # e.g., 59.94005994 (from cv2.CAP_PROP_FPS)
```

These values determine:
1. Which EOTF/OETF to use for transfer function conversion
2. Which color metadata flags to pass to FFmpeg for output encoding
3. Which ICC profile to embed in still image output (PNG/TIFF)

##### Output Type Summary

| `out_type` | Codec | Pixel Format | Color Depth | Chroma | HDR Metadata |
|------------|-------|-------------|-------------|--------|-------------|
| `OUT_H264` (1) | libx264 | yuv420p | 8bit | 4:2:0 | None (SDR) |
| `OUT_H265` (2) | libx265 | yuv420p10le | 10bit | 4:2:0 | x265-params (PQ/HLG) |
| `OUT_PRORES_422` (3) | prores_ks | yuv422p10le | 10bit | 4:2:2 | colr atom (PQ/HLG) |
| `OUT_PRORES_4444` (4) | prores_ks | yuv444p10le | 10bit | 4:4:4 | colr atom (PQ/HLG) |
| `OUT_H265_SDR` (5) | libx265 | yuv422p10le | 10bit | 4:2:2 | BT.709 (SDR) |
| `OUT_PRORES_422_SDR` (6) | prores_ks | yuv422p10le | 10bit | 4:2:2 | BT.709 (SDR) |
| `OUT_H265_HW` (7) | hevc_videotoolbox | p010le | 10bit | 4:2:0 | PQ/HLG |

##### Notes on Color Accuracy

- **ProRes 422 (`OUT_PRORES_422`)** is recommended for archival HDR rendering as it preserves 4:2:2 chroma subsampling, matching typical HDR camera sources.
- **H.265 (`OUT_H265`)** uses 4:2:0 chroma subsampling, which discards half the chroma resolution compared to 4:2:2. This can cause subtle color differences, especially in saturated greens and reds.
- **SDR 8bit (`OUT_H264`)** input goes through OpenCV (BGR uint8), not PyAV. The `_write_video_frame` method handles BGR→RGB channel reorder before piping to FFmpeg.
- When the source and output share the same transfer function (e.g., both PQ), **no EOTF/OETF conversion is applied** — the 16bit RGB values pass through unchanged, preserving maximum precision.

##### YUV-Native Pipeline (RGB Roundtrip Bypass)

従来のレンダリングパイプラインでは、ソース動画の YUV フレームを PyAV で RGB に変換し、スリットスキャン処理後に FFmpeg で再度 YUV にエンコードしていました。この YUV→RGB→YUV ラウンドトリップにおいて、libswscale の BT.2020nc 色行列変換に**系統的な Cr チャネルバイアス（10bit で約 -1.0）** が存在し、出力映像の緑色が不自然に鮮やかになる問題がありました。

YUV-native パイプラインでは、この RGB 変換を完全にスキップします:

```
Source Video (YUV 4:2:2 10bit)
    │
    ▼  [PyAV decode — プレーン直接取得]
    │  frame.planes[0] → Y  (full width)
    │  frame.planes[1] → Cb (half width, 4:2:2)
    │  frame.planes[2] → Cr (half width, 4:2:2)
    │  ※ RGB変換なし — YUVデータをそのまま取得
    ▼
Y / Cb / Cr Buffers (numpy uint16 arrays, separate)
    │
    ▼  [Slit-scan processing in YUV space]
    │  Y: フル解像度でスリット合成
    │  Cb/Cr: 半幅でスリット合成（column_index // 2）
    │  Numba JIT で高速化
    ▼
Composited Y / Cb / Cr Frames
    │
    ▼  [FFmpeg encode — YUV直接入力]
    │  planar yuv422p10le として stdin にパイプ
    │  ※ FFmpeg側の RGB→YUV 変換も不要
    ▼
Output Video File (ProRes 422/4444)
```

**有効条件（自動判定）:**
- 入力が 10bit 以上（`is_morethan_8bit == True`）
- 出力が ProRes 422 または ProRes 4444
- トランスファー関数の変換が不要（入力と出力が同一、または `force_hdr_mode` 未指定）
- PyAV コンテナが利用可能

```python
# 判定ロジック（new_transprocess 内）
use_yuv_native = (
    self.is_morethan_8bit
    and out_type in (self.OUT_PRORES_422, self.OUT_PRORES_4444)
    and _hdr_mode_matches
    and self.container is not None
)
```

**改善効果:**
| 指標 | RGB パス (従来) | YUV-native パス |
|------|----------------|-----------------|
| Cr bias (10bit) | -1.0 (系統的) | -0.05 (無視可能) |
| 緑色シフト | 目視で明確 | 検出不能 |
| 処理速度 | 基準 | 約10-20%高速 |

**制限事項:**
- `force_hdr_mode` で HLG↔PQ 変換を行う場合、EOTF/OETF 処理に RGB 空間が必要なため、従来の RGB パイプラインにフォールバックします。
- H.265 出力（`OUT_H265`）は現在 RGB パスのみ対応です。H.265 では入力側の色メタデータ指定で正しい色行列が使用されるため、実用上の色精度は十分です。
- SDR 8bit 入力（OpenCV パス）には適用されません。

**関連メソッド:**
- `_process_frame_yuv()` — YUV空間でのスリットスキャン処理
- `_process_frame_vertical_yuv_jit()` / `_process_frame_horizontal_yuv_jit()` — Numba JIT 高速化カーネル
- `_render_images_to_sink_yuv()` — YUV バッファの動画出力
- `_build_ffmpeg_cmd(use_yuv_native=True)` — yuv422p10le 入力の FFmpeg コマンド構築
- `_write_video_frame(use_yuv_native=True)` — Y/Cb/Cr プレーンの書き出し

#### Audio Rendering
There are two ways to apply the same maneuver data (`data`) to sound:

1. **Python-only workflow (recommended)** — render the audio with `audio_render()` and produce an integrated video-with-audio file with `audio_video_out()`. SuperCollider is not required.
2. **Real-time processing in SuperCollider** — export a .scd file with `scd_out_v2()` (improved, recommended) or `scd_out()` (legacy) and run it in SuperCollider (https://supercollider.github.io).

In both cases the slits are thinned out to `thread_num` voices (default 20), and each voice plays the source sound at a playback position derived from its smoothly interpolated time trajectory.  
If you increase the number of voices too much, the volume may drop drastically due to frequency cancellation caused by slight time differences. Please choose an appropriate number based on the maneuver editing content and the acoustic characteristics of the source sound.

##### How the audio input is resolved
The audio input is resolved automatically in the following order:
1. A file explicitly given via the `audio_path` argument
2. An audio file with the same name as the input video (`videoname.AIFF` / `.aiff` / `.aif` / `.wav`, the legacy convention)
3. **The audio track of the input video file itself** (mp4/mov works as-is; you no longer need to prepare a separate AIFF)

##### Python-only audio rendering: audio_render
The trajectory is smoothly interpolated up to audio rate (by default with band-limited Fourier interpolation, i.e. reconstruction as a superposition of frequency components), and the source sound is read directly at that playback position, producing a stereo WAV. Because playback is position-driven, no click noise occurs and periodic maneuvers such as CycleTrans re-synchronize exactly at the end of each cycle.

```python
bm.audio_render(thread_num=20, mode="play")   # variable-speed playback (pitch follows the rate)
bm.audio_render(thread_num=20, mode="grain")  # granular synthesis (pitch preserved)
```

Main parameters:
- `mode`: `"play"` (variable-speed playback) / `"grain"` (granular synthesis, pitch preserved)
- `smooth`: `"fourier"` (band-limited interpolation, default) / `"spline"` (cubic spline)
- `n_harmonics`: number of strongest frequency components kept in fourier mode. Setting this smooths the trajectory further (None = all components)
- `inpan_mode`: how the spatial coordinate (reading position) of each slit is reflected in the sound
  - `"balance"` (default): stereo balance — reading near the left edge emphasizes the source L channel, near the right edge the R channel
  - `"gain"`: legacy SC-code compatible (spatial coordinate applied as a common gain)
  - `"none"`: fixed per-voice panning only
- `grain_dur` / `grain_rate`: grain length [sec] and trigger rate [Hz] for grain mode
- `jump_thresh_sec`: threshold [sec] for detecting discontinuous jumps in the trajectory. Jumps are split into segments and cross-faded to prevent clicks

##### Integrated audio+video output: audio_video_out
Call this after rendering the video with `transprocess` etc.; it renders the audio and muxes it into the video in one step, producing a **video with audio**. The video stream is copied without re-encoding, so there is no quality loss (audio: PCM 24bit for .mov, AAC 320k for .mp4).

```python
bm.transprocess()                                  # render the video
bm.audio_video_out(thread_num=20, mode="grain")    # render audio + mux in one call
# → produces *_wAudio.mp4 (.mov) next to the latest video output
```

- `videopath`: video to mux with (default: the most recently rendered `out_videopath`)
- `rendered_audio`: reuse an already rendered WAV (if None, `audio_render` is executed internally)
- All other arguments are passed through to `audio_render`

##### Improved SuperCollider output: scd_out_v2
For real-time / live use, use the improved SuperCollider export. Instead of the legacy CSV + 30Hz control loop, it writes the interpolated playback positions and balance information as a multi-channel float32 WAV (control track), and the SC side plays back position-driven via `BufRd`. Click noise and the re-synchronization drift of periodic maneuvers are eliminated.

```python
bm.scd_out_v2(thread_num=20)
# → *_SCv2ctl-20voices.wav (control track)
#    *_SCv2_Play-20voices.scd / *_SCv2_Grain-20voices.scd
```

Each generated .scd plays and records in a single block execution (the legacy two-step execution is not needed). If you need to configure the output device, edit the comment line at the top of the file.

##### Exporting as Fourier components: maneuver_fourier_out
Exports each voice's time trajectory as a CSV of "trend line + a set of frequency components (frequency, amplitude, phase)". It can be reconstructed as `pos(t) = trend + Σ amp·cos(2πft + φ)`, which is useful for driving an oscillator bank in SuperCollider or for using the trajectory data from native apps written in Swift etc.

```python
bm.maneuver_fourier_out(thread_num=20, n_harmonics=64)
```

##### Legacy scd_out
The legacy `scd_out` is kept unchanged for compatibility, but it has the following known issues. For new projects, `audio_render` / `scd_out_v2` are recommended.
- 30Hz control updates + piecewise-linear interpolation cause distortion at the interpolation corners
- The playback position is tracked by integrating the rate, so it drifts and CycleTrans does not re-synchronize exactly at the end of a cycle
- Periodic re-triggering causes click noise
- The spatial coordinate (inPan) is applied as a common gain instead of the intended stereo balance

The audio file name can be specified with the instance variable `sc_FNAME`. By default, it is set as [input video file name.AIFF]. Please make sure it is saved in the same directory as the input video.  
The number of voices can be specified as the first argument of `scd_out`. If omitted, it defaults to 20.

```python
bm.sc_FNAME="GX010230-t-AIFF.aiff"
bm.scd_out(7)
```
Upon running the above, four CSV data and four SuperCollider program .scd files will be output.

##### 4 CSV Data of Maneuver
1. *_7threads.csv : Time position of the slit.
1. *_Rate_7threads.csv : Playback rate of the slit.
1. *_inPanMap_7threads.csv : Spatial position of the slit.
1. *_nowDepth_7threads.csv : Time offset within a single frame.

##### 4 SCD files
1. *_SC_Play-7voices.scd : Multi-play, pitch changes according to the playback rate.
1. *_SC_Grain-7voices.scd : Multi-play using granular synthesis. The pitch does not change regardless of the playback rate.
1. *_SC_Rev_Play-7voices.scd : Multi-play. Apply reverb according to the time offset.
1. *_SC_Rev_Grain-7voices.scd : In addition to multi-play using granular synthesis, add reverb according to the time offset.

Please check the sample video file to refer to its effects and characteristics.

##### Running the scd file
Load one of the scd files into SuperCollider.  
All of them perform sound processing in real-time and save it as an audio file on a virtual server. 
The saved audio file will be stored in the same directory as the video rendering data.  
Since it takes time to read the audio data and CSV data, we avoid running them all at once and divide them into two processes.  
Please execute the contents enclosed in `()` in order.

![Alt text](images/scd_sample.png)
 
###### 1. Audio Settings and Data Loading
Set up the audio, define the synth with `SynthDef`, load the audio data, and load the CSV data.  
Please customize the audio output device setting according to your environment.  
By default, it is set as follows:

```supercollier
Server.default.options.outDevice_("MacBook Pro's speaker");
```
By running the below, a list of available devices will be output to the console window.
```supercollier
ServerOptions.devices; 
```

###### 2. Playing `Synth` and Recording
Perform real-time playback of the defined `Synth` with loop processing and record.  
At the same time as recording, play the rendering video file in QuickTime Player with a Unix command. 
There will be a slight time gap, but you can play the video and audio in a pseudo-synchronized state. 
This can only be executed on a Mac with QuickTime Player, so if you are in a different environment, please comment out this part and adapt accordingly.
```unixcmd
"open -a 'QuickTime Player' '/Users/Movies/sample-raw-mov/sample_Vslit.mp4' ".unixCmd;
```
#### Combining Audio and Video
For the Python-only workflow, [`audio_video_out`](#integrated-audiovideo-output-audio_video_out) handles everything from audio rendering to muxing automatically, so no manual work is needed.

If you use audio recorded in SuperCollider (.aiff), either synchronize the video and audio in a video editing software and re-export, or pass the rendered WAV/AIFF as `rendered_audio` to `audio_video_out`.

```python
bm.audio_video_out(rendered_audio="rec_from_supercollider.aiff")
```

## drawManeuver Class

This class is the main component of the Imgtrans library.

### Class Variables:
- `imgtype`: Format of the still image in rendering. `.png` and `.tif` support 16-bit; `.jpg` is 8-bit only (default is `".png"`)
- `img_size_type`: Setting for the output image size. Given the height as h and width as w of the input video, `0`:h,w `1`:w,w*2 `2`: Total number of frames `3`: square (default is `0`)
- `outfps`: Frame rate for the output (default is `30`)
- `recfps`: Recording frame rate, typically overwritten by input video FPS on init (default is `120`)
- `progressbarsize`: Width of the console progress bar in characters (default is `50`)
- `sepVideoOut`: Separate rendering mode. `0` = accumulate all npy temp files on disk before rendering (can consume 100GB+); non-zero = render segments directly (default is `0`)
- `memory_percent`: Memory usage limit as a percentage of active memory during rendering (default is `60`)
- `auto_visualize_out`: Setting for automatic visualization (default is `True`)
- `default_debugmode`: Default debug mode setting (default is `False`)
- `audio_form_out`: Setting for audio format output (default is `False`)
- `embedHistory_intoName`: Setting for embedding history into the name (default is `True`)
- `some_recfps_array`: List of recording FPS values for multi-FPS rendering. Set externally before calling rendering methods (default is `[]`)
- `plot_w_inc`: Width in inches for the 2D plot output (default is `5`)
- `plot_h_inc`: Height in inches for the 2D plot output (default is `9`)
- `xyt_boxel_scale`: Aspect ratio scale for the XYT spacetime cube. When testing with lower resolution video, set this to compensate for resolution differences (default is `1`)
- `OUT_STILL`, `OUT_H264`, `OUT_H265`, `OUT_PRORES_422`, `OUT_PRORES_4444`, `OUT_H265_SDR`, `OUT_PRORES_422_SDR`, `OUT_H265_HW`: Output type constants (`0`-`7`) used with the `out_type` parameter in rendering methods. `OUT_H265_HW` (7) uses the macOS VideoToolbox hardware HEVC encoder for faster HDR10 output.

### initialization
The class initialization method takes the attributes of the video path, scan direction, data, and folder name as arguments. This method initializes the instance variables below, creates an output directory at the same level as the video path, and moves to that directory. All output files will be saved within this directory.

#### Parameters
- `videopath` (str): Path to the input. This can be **either a single video file** (any container ffmpeg/OpenCV can read — `.mov`, `.mp4`, etc.) **or a directory containing an image sequence**. When a directory is given, all `.png` / `.jpg` / `.jpeg` / `.tif` / `.bmp` / `.npy` files inside it are read in sorted order as consecutive frames (frame count = number of files; the FPS is taken from `recfps`/class default since a sequence carries no FPS metadata).
- `sd` (bool): Direction of the slit. `True` for vertical slit, `False` for horizontal slit.
- `outdir` (str, optional): Directory indication of the output folder. Default is the same as the input video data path.
- `datapath` (str, optional): Optional path to previously saved maneuver data, saved as a multi-dimensional array in npy format.
- `foldername_attr` (str, optional): Optionally appends the specified name to the output directory's name.
- `another_fps_dir` (str, optional): Directory of additional source videos with differing FPS, enabling multi-FPS rendering (see `some_recfps_array`).
- `recfps` (float, optional): The **actual** capture FPS of the input. Defaults to the FPS read from the video metadata (`cv2.CAP_PROP_FPS`). Specify this explicitly when the stored metadata FPS differs from the real capture rate (e.g. footage shot at 480fps but stored as 30fps), or to set the frame rate for an image-sequence input.
- `outfps` (float, optional): Output frame rate. `None` keeps the class default (`30`).

#### Instance Variables
1. **data**: Maneuver data of the playback section with the slit as the minimum unit. Defaults to an empty list.
1. **width**: Width of the video. Reflects the video information read from `videopath`.
1. **height**: Height of the video.
1. **count**: Total number of video frames.
1. **recfps**: Frame rate (fps) of the video. The output frame rate is set in the [Class Variables](#class-variables).
1. **inputmovfps**: Original FPS of the input video file (set from `cv2.CAP_PROP_FPS`).
1. **scan_direction**: Defines the slit orientation and scan direction. The `sd` argument from the initialization method is applied directly.
1. **scan_nums**: Number of scans. 3840 for vertical slit at 4k resolution.
1. **slit_length**: Number of pixels in one slit. 2160 for vertical slit at 4k resolution.
1. **out_name_attr**: The `foldername_attr` argument from the initialization method is applied directly.
1. **out_videopath**: Holds the path of the output video. Initially empty. Referenced by [animationout](#animationout) and [audio_video_out](#integrated-audiovideo-output-audio_video_out).
1. **sc_FNAME**: Initially set to automatically accept the input video's filename with ".AIFF" added. Used when outputting code for audio processing in super collider.
1. **sc_resetPositionMap**, **sc_rateMap**, **sc_inPanMap**, **sc_now_depth**: Arrays optimized for audio processing by reducing the number of slit divisions from the maneuver array.
1. **cycle_axis**: Array storing rotation center axis positions. Referenced by `applyTimebySpace` (mode=1).
1. **another_videos**: List of additional video paths for multi-FPS rendering. Set via `another_fps_dir` in init.
1. **input_pix_fmt**: Pixel format of the input video (e.g., `yuv420p10le`). Detected automatically.
1. **input_bit_depth**: Bit depth of the input video (default: `8`).
1. **input_primaries**: Color primaries of the input video (e.g., `bt2020`, `bt709`). Detected automatically via ffprobe.
1. **input_transfer**: Transfer characteristics of the input video (e.g., `smpte2084` for PQ, `arib-std-b67` for HLG). Detected automatically.
1. **input_colorspace**: Color space / matrix of the input video (e.g., `bt2020nc`). Detected automatically.
1. **is_morethan_8bit**: Boolean flag indicating HDR/high-bit-depth input (bit depth > 8, or a PQ/HLG transfer function). When True, rendering takes approximately 2.6x longer. Detected automatically.
1. **is_sdr10bit**: Boolean flag indicating SDR 10bit input (BT.709 transfer with bit depth > 8). Detected automatically.
1. **force_hdr_mode**: Force specific HDR output mode. `None` = follow input, `"hlg"` = force HLG, `"pq"` = force PQ.
1. **log**: Accumulated maneuver operation log string. Built up by each method call for output filename generation.
1. **infolog**: Accumulated info log string. Used by `info_setting`.
1. **depth_to_sel_recfps**: Array mapping depth to selected recording FPS. Used in multi-FPS rendering.
1. **renderfps_scales**: Array of FPS scale ratios (`some_recfps_array / recfps`). Used in multi-FPS rendering.

#### Example
```python
bm=imgtrans.drawManeuver(videopath="path/to/video.mp4", sd=1)
```
To inherit previously saved maneuver data, refer to the example below:
```python
import numpy as np
bm=imgtrans.drawManeuver(videopath="path/to/video.mp4", sd=1, datapath="path/to/data.npy")
# Check the maneuver data
print(bm.data.shape)
```


### List of All Class Methods:
- [`__init__`](#initialization): Initializes by receiving the video path. Params: `videopath`(str), `sd`(bool: slit direction), `outdir`(str), `datapath`(str: npy path), `foldername_attr`(str), `another_fps_dir`(str). See [Initialization](#initialization) for details.
- [`append`](#append): Appends the maneuver data created separately to the end of `data`. Params: `maneuver`(ndarray), `auto_zslide`(bool: auto time-adjust, default True), `zslide`(int: manual offset).
- [`prepend`](#prepend): Adds a maneuver to the beginning of `data`. Params: `maneuver`(ndarray).
- [`arrayExtract`](#arrayextract): Extracts a range from `data` and rewrites it. Params: `start`(int), `end`(int).
- [`arrayReflection`](#arrayreflection): Mirror-reflects `data` by appending it in reverse order (time-reversed copy). No params.
- [`wide_expandB`](#wide_expandb): Expands the spatial dimension by extrapolating edge slit data outward. Params: `add_size`(int, default 3840), `sclip`(bool), `zclip`(bool), `spacedirection`(bool), `z_offset`(int: time offset per added slit).
- [`interpolation_append`](#interpolation_append): Smoothly connects the current `data` to another maneuver array. Params: `maneuver`(ndarray), `connection_num`(int: transition frames), `speed_round`(bool), `add_maneuver`(bool).
- [`interpolation_append_byspeed`](#interpolation_append_byspeed): Connects to another maneuver at a specified speed with auto blur. Params: `maneuver`(ndarray), `frame_speed`(float), `speed_round`(bool), `add_maneuver`(bool), `sblur`(bool), `tblur`(bool), `blur_range`(int).
- **Classes related to maneuver design**
    - **Functions that add space-time integrated motion**
        - [`addFlat`](#addflat): Adds a flat array. Params: `frame_nums`(int), `z_pos`(int), `z_autofit`(bool), `prepend`(bool), `flip`(bool).
        - [`addFreeze`](#addfreeze): Creates and adds an array of the final row for the number of frames specified. Params: `frame_nums`(int).
        - [`addSlicePlane`](#addsliceplane): Adds a cross-sectional frame sliced along the time axis. Params: `frame_nums`(int), `xypoint`(float: spatial position 0-1), `full_range`(bool), `z_start`(float), `z_end`(float).
        - **3D Geometric Surface Cuts** — Cut XYT space with geometric surfaces to produce a single closed-loop frame.
            - [`addCylinderCut`](#addcylindercut): Cylinder cut — a circular/elliptic loop on the (space, time) plane. Params: `center_time`(float: time center in frames), `center_pos`(float: spatial center 0-1, default 0.5), `time_scale`(float: time-axis scale, default 1.0), `phase`(float: start angle in radians, default 0.0), `output_width`(int: number of output slits, None=auto).
            - [`addBoxUnfoldCut`](#addboxunfoldcut): Unfolded-box loop cut — normal frame → right-edge slit scan → flipped frame → left-edge slit scan. Params: `center_time`(float), `center_pos`(float, default 0.5), `time_scale`(float, default 1.0), `output_width`(int, None=auto).
        - **Transposition of space and time dimensions**
            - [`addTrans`](#addtrans): Simple transposition of space and time dimensions. Params: `frame_nums`(int), `start_line`(float), `end_line`(float), `speed_round`(bool), `zd`(bool), `zscale`(float).
            - [`addKeepSpeedTrans`](#addkeepspeedtrans): Creates frames maintaining existing speed until a spatial area is reached. Params: `frame_nums`(int), `under_xyp`(float), `over_xyp`(float), `rendertype`(int).
            - [`addInsertKeepSpeedTrans`](#addinsertkeepspeedtrans): Advanced version of `addKeepSpeedTrans`. Params: `frame_nums`(int), `under_xyp`(float), `over_xyp`(float), `after_array`(list), `rendertype`(int).
            - [`addWideKeyframeTrans`](#addwidekeyframetrans): Wide-output version for outputs larger than input. Params: `frame_nums`(int), `key_array`(list), `wide_scale`(int, default 3), `start_frame`(list), `speed_round`(bool).
            - [`addBlowupTrans`](#addblowuptrans): Blowup motion with keyframe control. Params: `frame_nums`(int), `deg`(int, default 360), `speed_round`(bool), `connect_round`(list), `timevalues`(list), `timepoints`(list: 0-1 ratios), `timecenter`(list), `extra_degree`(int), `wave_type`(int), `zslide`(int).
       - **Transition of space-time dimensions**
            - [`addInterpolation`](#addinterpolation): Interpolates and adds to the data. Params: `frame_nums`(int), `i_direction`(bool), `z_direction`(bool), `axis_position`(bool), `s_reversal`(bool), `z_reversal`(bool), `cycle_degree`(int, default 90), `extra_degree`(int), `zslide`(int), `speed_round`(bool), `rrange`(list), `zscale`(float).
            - [`rootingA_interporation`](#rootinga_interporation): Combines multiple addInterpolations in zigzag motion. Params: `FRAME_NUMS`(int), `loop_num`(int, default 2), `axis_first_p`(int), `speed_round`(bool), `interval_nums`(int: freeze frames between segments), `loopinterval_nums`(int).
            - [`rootingA_interporation_single`](#rootinga_interporation_single): Single-segment rootingA. Params: `FRAME_NUMS`(int), `seg_type`(int: 0=fwd→rev, 1=rev→fwd), `speed_round`(bool), `interval_nums`(int), `panorama_nums`(int), `flip_axis`(bool), `junction_mode`(int: 0=normal, 1=smooth), `blur_rate`(int, default 90).
            - [`rootingA_interporation_trans_single`](#rootinga_interporation_trans_single): Single-segment combining interpolation with transposition. Params: `FRAME_NUMS`(int), `seg_type`(int), `speed_round`(bool), `interval_nums`(int), `trans_nums`(int), `trans_end_line`(float), `flip_axis`(bool), `junction_mode`(int), `blur_rate`(int), `time_flip`(bool).
            - [`rootingA_interporation_RANDOM`](#rootinga_interporation_random): Randomized rootingA with random axis and direction selection. Params: `FRAME_NUMS`(int), `loop_num`(int), `seed`(int), and various range params.
            - [`rootingAA_interporation`](#rootingaa_interporation): Variant of rootingA keeping same spatial start and end positions. Params: `FRAME_NUMS`(int), `loop_num`(int), `axis_first_p`(int), `speed_round`(bool).
            - [`rootingB_interporation`](#rootingb_interporation): Domino-rolling-down-a-slope motion. Params: `FRAME_NUMS`(int), `loop_num`(int), `axis_fix_p`(int).
            - [`rooting8_interporation`](#rooting8_interporation): 8-pattern (figure-eight) interpolation. Params: `FRAME_NUMS`(int).
            - [`rooting8B_interporation`](#rooting8b_interporation): Variant of rooting8. Params: `FRAME_NUMS`(int).
            - [`rooting4C_interporation`](#rooting4c_interporation): 4C-pattern interpolation. Params: `FRAME_NUMS`(int).
            - [`rooting4D_interporation`](#rooting4d_interporation): 4D-pattern interpolation. Params: `FRAME_NUMS`(int).
            - [`addCycleTrans`](#addcycletrans): Rotates the playback cross-section around the screen centerline. Params: `frame_nums`(int), `cycle_degree`(int, default 360), `t_auto_scaling`(bool), `zslide`(int), `extra_degree`(int), `speed_round`(bool), `spaceflow`(bool), `zscale`(float).
            - [`addCustomCycleTrans`](#addcustomcycletrans): Movable center axis version of addCycleTrans. Params: `frame_nums`(int), `cycle_degree`(int), `start_center`(float, default 0.5), `end_center`(float, default 0.5), `t_auto_scaling`(bool), `extra_degree`(int), `speed_round`(bool), `zslide`(int), `auto_zslide`(bool), `t_auto_scaling_num`(float), `zscale`(float), `spaceflow`(bool).
            - [`addWideCustomCycleTrans`](#addwidecustomcycletrans): Wide-output version of addCustomCycleTrans. Params: `frame_nums`(int), `cycle_degree`(int), `start_center`(float), `end_center`(float), `maxz_range`(int), `wide_scale`(int, default 3), `t_auto_scaling`(bool), `extra_degree`(int), `speed_round`(bool).
            - [`addFixWideCycleTrans`](#addfixwidecycletrans): Fixed-width wide cycle trans with auto time-axis scaling. Params: `frame_nums`(int), `cycle_degree`(int), `wide_scale`(int, default 3), `t_auto_scaling`(bool), `extra_degree`(int), `speed_round`(bool).
        - **Wavy playback cross-section**
            - [`addWaveTrans`](#addwavetrans): Dynamic wave-shaped playback cross-section. Params: `frame_nums`(int), `cycle_degree`(float: wavelength), `zdepth`(float: wave amplitude), `flow`(bool: move spatial axis), `zslide`(float), `speed_round`(bool).
            - [`addEventHorizonTrans`](#addeventhorizontrans): Time progression varies between center and periphery. Params: `frame_nums`(int), `zdepth`(float), `z_osc`(int), `cycle_degree`(int, default 180), `flow`(bool), `zslide`(int).
    - **Time-focused maneuver**
        - [`applyTimeForward`](#applytimeforward): Apply forward time flow to the entire array. Params: `slide_time`(int: frames per output frame, default=recfps/outfps), `start_frame`(int, default 0), `end_frame`(int).
        - [`applyTimeOblique`](#applytimeoblique): Apply oblique time effect — shifts each slit progressively. Params: `maxgap`(int: maximum time shift).
        - [`applyTimeForwardAutoSlow`](#applytimeforwardautoslow): Adds normal playback intro/outro with ease processing. Params: `slide_time`(int), `defaultAddTime`(int, default 100), `addTimeEasing`(bool), `easeRatio`(float, default 0.3).
        - [`applyTimeFlowKeepingExtend`](#applytimeflowkeepingextend): Prepends/appends extended frames maintaining time flow rate. Params: `frame_nums`(int), `fade`(bool: ease to speed 0), `intro`(bool), `outro`(bool), `fade_speed`(int), `fade_type`(str: "inout"), `space_apply`(bool).
        - [`applyTimeFlowKeepingExtend_CoodinateBase_Intro`](#applytimeflowkeepingextend_coodinatebase_intro): Coordinate-based intro extension with zero accumulated error. Params: `target_z`(float: exact destination timecode for all slits), `num_frames`(int: number of extension frames). Per-slit step = `(data[0,slit,1] - target_z) / num_frames`.
        - [`applyTimeFlowKeepingExtend_CoodinateBase_Outtro`](#applytimeflowkeepingextend_coodinatebase_outtro): Coordinate-based outro extension with zero accumulated error. Params: `target_z`(float: exact destination timecode), `num_frames`(int). Per-slit step = `(target_z - data[-1,slit,1]) / num_frames`.
        - [`applyTimeLoop`](#applytimeloop): Time loop — forward, reverse, forward — creating a seamless loop. Params: `slide_time`(int), `freq`(int, default 2), `stay_time`(int, default 30), `intepolation_min`(int, default 300), `stay_time_min`(int, default 30).
        - [`applyTimeLoopB`](#applytimeloopb): Per-slit variant of applyTimeLoop with individual stay time adjustment. Params: `slide_time`(int), `freq`(int), `stay_time`(int, default 90), `intepolation_min`(int), `stay_time_min`(int).
        - [`applyTimeChoppyLoop`](#applytimechoppyloop): Triangle-wave time loop pattern. Params: `slide_time`(int), `frequency`(int), `phase_shift`(int), `rise`(float, default 0.5), `fall`(float, default 0.5).
        - [`applyTimeChoppyLoopB`](#applytimechoppyloopb): Extended choppy loop with sine wave option. Params: `slide_time`(int), `frequency`(int), `phase_shift`(int), `rise`(float), `fall`(float), `wave_type`(str: 'triangle'|'sine'), `blur`(int).
        - [`applyTimeClip`](#applytimeclip): Fix specified slits' time to a specific value. Params: `trackslit`(int: slit index), `cliptime`(float).
        - [`applyTimebySpace`](#applytimebyspace): Shift slits in time based on spatial position. Params: `v`(int: max frame shift), `mode`(int: 0=linear, 1=cycle_axis, 2=mean).
        - [`applyTimebyKeyframetoSpace`](#applytimebykeyframetospace): Shift slits in time by keyframe values. Params: `keyframes`(list: a list of shift amounts [frames], spread evenly across the spatial range and spline-interpolated), `mode`(int).
        - [`applyTimeSlide`](#applytimeslide): Set reference time of central slit in first frame. Params: `settime`(int: target time in frames), `baseframe`(int, default 0: -1 for last frame).
        - [`applyInOutGapFix`](#applyinoutgapfix): Seamless loop helper — linearly adjusts all frames to match first/last difference. No params.
        - [`applyInFix`](#applyinfix): Adjusts first frame to target, linear blend over all frames. Params: `target_z_array`(ndarray: target time values per slit).
        - [`applyOutFix`](#applyoutfix): Adjusts last frame to target with ease-in-out blend. Params: `target_z_array`(ndarray), `ease`(bool, default True).
        - [`applyInPartFix`](#applyinpartfix): Partial fix from frame 0 to `b_frame`. Params: `target_z`(float), `a_frame`(int: target frame), `b_frame`(int: blend end frame).
        - [`applyOutPartFix`](#applyoutpartfix): Partial fix from `a_frame` to end. Params: `target_z`(float), `a_frame`(int: blend start), `b_frame`(int: target frame), `b_frame_s_point`(int).
        - [`applyOutPartFixB`](#applyoutpartfixb): Array version — per-slit adjustment. Params: `target_z_array`(ndarray), `a_frame`(int), `b_frame`(int), `base_z_array`(ndarray).
        - [`applySpaceBlur`](#applyspaceblur): Apply spatial blur. Params: `bl_time`(int: blur kernel size in frames).
        - [`applyTimeBlur`](#applytimeblur): Apply temporal blur. Params: `bl_time`(int: blur kernel size in frames).
        - [`applyCustomeBlur`](#applycustomeblur): Apply custom range blur with weighted mean. Params: `s_frame`(int: start), `e_frame`(int: end), `bl_time`(int: kernel size), `dim_num`(int: 1=time, 0=space).
        - [`applyLoopBlur`](#applyloopblur): Blur for loop continuity — triples data, blurs, extracts center. Params: `sblur`(int: space blur), `tblur`(int: time blur).
        - [`applyConnectLoopBlur`](#applyconnectloopblur): Loop blur at connection points only. Params: `sblur`(int), `tblur`(int), `connect_frame`(int, default 100: blur range at connection).
        - [`applyPointBlur`](#applypointblur): Blur centered at a specific frame. Params: `point_frame`(int), `sblur`(int), `tblur`(int), `range_frame`(int, default 100).
    - **Spatial operations**
        - [`applySpaceFlip`](#applyspaceflip): Flips the spatial dimension of `data` (mirror reversal). No params.
        - [`applySpaceFlat`](#applyspaceflat): Resets spatial component to initial sequential values (0 to scan_nums-1). No params.
    - **Other maneuver functions**
        - [`addFreeze`](#addfreeze): Generate and add frames based on the final column's array. Params: `frame_nums`(int).
        - [`preExtend`](#preextend): Extend the first frame forward. Params: `addframe`(int: number of frames to prepend).
        - [`addExtend`](#addextend): Extend the final frame. Z-rate becomes 0. Params: `addframe`(int), `flip`(bool: mirror spatial axis).
        - [`zCenterArange`](#zcenterarange): Shifts time dimension to center at `count/2`. NaN-safe. Params: `center_time_frame`(int, optional: custom center).
        - [`zArange`](#zarange): Slides the whole data so that the mean time of a given frame lands on a target time. Params: `target_frame`(int: reference frame), `center_time_frame`(float, None=count/2).
        - [`zStartArange`](#zstartarange): Shifts time dimension so minimum starts at 0. No params.
        - [`zPointCheck`](#zpointcheck): Checks time coordinates are within valid range. Params: `subtract_count`(int, default 0: margin).
        - [`zPointCheckandReflect`](#zpointcheckandreflect): Checks and reflects out-of-range time values. Params: `subtract_count`(int, default 0).
        - [`spline_interpolate`](#spline_interpolate): Spline or linear interpolation of keyframes. Params: `x`(ndarray: positions), `keyframes`(list: [(pos, val),...]), `method`(str: 'spline'|'linear').
- **Methods for maneuver data output**
    - [`dataCheck`](#datacheck): Output `data` shape and min/max to the console. No params.
    - [`info_setting`](#info_setting): Configure data for audio/analysis output. Params: `thread_num`(int, default 20: number of slit divisions), `raw`(bool: output raw arrays).
    - [`maneuver_CSV_out`](#maneuver_csv_out): Output maneuver data to CSV. Params: `thread_num`(int), `time_map`(bool), `space_map`(bool), `time_rate_map`(bool), `now_depth_map`(bool), `space_rate_map`(bool), `movement_rate_map`(bool).
    - [`scd_out`](#legacy-scd_out): Output SuperCollider code and CSV data (legacy). Params: `thread_num`(int), `audio_path`(str). See [Legacy scd_out](#legacy-scd_out) for details.
- Audio output methods (see [Audio Rendering](#audio-rendering) for details)
    - [`audio_render`](#python-only-audio-rendering-audio_render): Render audio inside Python and write a stereo WAV. Params: `thread_num`(int), `audio_path`(str), `mode`(str: "play"|"grain"), `smooth`(str: "fourier"|"spline"), `n_harmonics`(int), `inpan_mode`(str: "balance"|"gain"|"none"), `grain_dur`(float), `grain_rate`(float), `jump_thresh_sec`(float), `normalize`(bool).
    - [`audio_video_out`](#integrated-audiovideo-output-audio_video_out): Render audio and mux it into the rendered video in one step, producing a video with audio. Params: `thread_num`(int), `videopath`(str), `rendered_audio`(str); other kwargs are passed to `audio_render`.
    - [`scd_out_v2`](#improved-supercollider-output-scd_out_v2): Output position-driven SuperCollider code + a control-track WAV (improved). Params: `thread_num`(int), `audio_path`(str), `ctl_rate`(int), `smooth`(str), `n_harmonics`(int), `inpan_mode`(str), `grain_dur`(float), `amp`(float).
    - [`maneuver_fourier_out`](#exporting-as-fourier-components-maneuver_fourier_out): Export each voice's time trajectory as a CSV of Fourier components (frequency, amplitude, phase). Params: `thread_num`(int), `n_harmonics`(int).
    - [`data_save`](#data_save): Save maneuver data as npy. Params: `attr`(str: filename suffix), `sep`(int: split count, 0=no split).
    - [`split_3_npySave`](#split_3_npysave): Split data into 3 parts (L/C/R) and save as npy. No params.
    - [`split_3_npysavereturn`](#split_3_npysavereturn): Same as `split_3_npySave` but returns file paths as array. No params.
    - [`vsizeReturn`](#vsizereturn): Returns output image dimensions (width, height). No params.
- **Methods for maneuver visualization file output**
    - [`maneuver_2dplot`](#maneuver_2dplot): 2D plot with optional seekbar video. Params: `thread_num`(int), `thread_through`(bool), `debugmode`(bool), `normal_line_draw`(bool), `w_inc`(float), `h_inc`(float), `video_out`(bool), `video_alpha`(bool).
    - [`maneuver_3dplot`](#maneuver_3dplot): 3D plot animation. Params: `thread_num`(int), `thread_through`(bool), `zRangeFix`(bool), `out_framenums`(int), `out_fps`(int), `colormode`(str), `line_width`(float), `aspect_ratio`(tuple), `elev`(float), `azim`(float), `dpi`(int), `xticks`(bool), `zticks`(bool), `yticks_normal`(bool), `only_seq_img`(bool), `lineplot`(bool), `vectorplot`(bool), `gridplot`(bool), `vector_def_frame`(int), `velocity`(float), `vector_color_amp`(float), `s_frame`(int), `zRangeMin`(float), `zRangeMax`(float).
    - [`maneuver_3dplot_midtide`](#maneuver_3dplot_midtide): 3D plot optimized for mid-tide wide output. Params: `thread_num`(int), `thread_through`(bool), `zRangeFix`(bool), `out_framenums`(int), `out_fps`(int), `colormode`(str), `aspect_ratio`(tuple), `elev`(float), `azim`(float), `dpi`(int).
    - [`maneuver_imgplot`](#maneuver_imgplot): Static image plots (space/time/rate). Params: `plot_mode`(str: "space"|"time"|"rate"|"all"), `colormode`(str), `nticks_x`(int), `nticks_y`(int), `save_png`(bool), `time_axis`(str: 'auto'|'frame'|'sec').
    - [`img_to_maneuver`](#img_to_maneuver): Reconstruct `data` from space and time images (16-bit PNG). Params: `space_img_path`(str), `time_img_path`(str), `space_set`(float), `vrange`(float).
    - [`img_to_maneuver_rate_based`](#img_to_maneuver_rate_based): Reconstruct `data` from rate image by integrating. Params: `time_rate_path`(str), `space_img_path`(str), `space_set`(float), `start_time`(float), `rate_range`(float), `rate_baseline`(float), `rate_startpoint`(float).
- **Methods for video rendering**
    - [`new_transprocess`](#new_transprocess): Primary HDR rendering method. Params: `separate_num`(int), `sep_start_num`(int), `sep_end_num`(int), `out_type`(int: 0-7), `xy_trans_out`(bool), `render_mode`(int: 0-3), `title_atr`(str), `del_data`(bool), `render_clip_start`(int), `render_clip_end`(int), `slit_step`(int: downscale slit), `scan_step`(int: downscale scan), `use_pyav`(bool).
    - [`transprocess`](#transprocess): Legacy OpenCV rendering. Params: `separate_num`(int), `sep_start_num`(int), `sep_end_num`(int), `out_type`(int), `XY_TransOut`(bool), `render_mode`(int), `seqrender`(bool), `title_atr`(str).
    - [`pretransprocess`](#pretransprocess): Fast preview rendering with thinned frames. Params: `outnums`(int, default 100: output frame count), `xy_trans_out`(bool).
- **Analysis**
    - [`movement_intensity_analyze`](#movement_intensity_analyze): Analyzes motion intensity frame by frame and plots. No params.
- **Post-rendering methods**
    - [`overlay_tc_rate`](#overlay_tc_rate): Overlays timecode and playback rate on rendered video. Params: `output_suffix`(str, default "_tc"), `divisions`(int, default 5: number of probe points across frame width). Rate color: yellow(+1), blue(-1), gray(0). Timecode displayed as `{sec}sec---{frac}f`.
    - [`animationout`](#animationout): Plot pixel colors from rendered video onto 3D graph as animation. Params: `out_framenums`(int, default 100), `drawLineNum`(int, default 250), `dpi`(int, default 200), `out_fps`(int, default 10).
    - [`animationout_custome`](#animationout_custome): Customizable animationout. Params: `zRangeFix`(bool), `out_framenums`(int), `drawLineNum`(int), `dpi`(int), `out_fps`(int), `aspect_ratio`(tuple), `elev`(float), `azim`(float), `colormode`(str), `transparent`(bool), `gridplot`(bool), `vectorplot`(bool), `vector_def_frame`(int), `velocity`(float), `vector_color_amp`(float), `s_frame`(int).

### Standalone Functions (Module-level)

These are not class methods but standalone functions available at the module level.

- [`export_segments`](#export_segments): Exports A/B segments from a source video with frame number overlay and real-time speed correction. A segments are forward playback; B segments are hflip + reverse.
- [`rendered_npys_to_mov`](#rendered_npys_to_mov): Combines split-rendered npy files into a single video file. Supports all `out_type` formats (H.264, H.265, ProRes 422, ProRes 4444, etc.).
- [`rearrange_wide_video`](#rearrange_wide_video): Rearranges a wide panoramic video by interleaving left-half and right-half columns. Can read from npy files or an existing video file.
- [`rendered_mov_to_seq`](#rendered_mov_to_seq): Extracts frames from a rendered video and saves as an image sequence. Params: `video_path`(str), `divide_num`(int: split into subfolders), `img_format`(str: `'jpg'`|`'png'`|`'ultrahdr'`|`'avif'`|`'npy'`), `frame_array`(array: selective extraction), `color_mode`(str: `'source'`|`'sdr'`|`'hlg'`).
- [`convert_npy_to_jpg`](#convert_npy_to_jpg): Converts a single npy file (saved frame array) to JPEG images.
- [`custom_blur`](#custom_blur): Applies weighted-mean blur to a data array over a specified frame range and dimension. Used internally by `applyCustomeBlur`.
- [`custom_onedimention_blur`](#custom_onedimention_blur): 1D weighted-mean blur for a single time array over a specified frame range.
- [`double_first_dimension_with_interpolation`](#double_first_dimension_with_interpolation): Doubles the first dimension of a 3D array by inserting interpolated frames between existing ones.


## `append`
Appends a separately created maneuver array to the end of `data`. By default the time gap at the junction is compensated automatically so that time stays continuous.

### Parameters
- `maneuver`(ndarray): Maneuver array to append. shape=(frames, slits, 2).
- `auto_zslide`(bool, optional, default: `True`): If `True`, cancels the time gap between the head of `maneuver` and the tail of `data`.
- `zslide`(int, optional, default: `0`): Manual time offset. A non-zero value takes precedence over `auto_zslide`.

### Usage
```python
array = bm.data.copy()
bm.append(array)      # doubled length with continuous time
```

## `prepend`
Prepends a maneuver array to the head of `data`. Unlike `append`, no automatic time-gap compensation is applied.

### Parameters
- `maneuver`(ndarray): Maneuver array to prepend.

## `arrayExtract`
Replaces `data` with the specified frame range. Useful e.g. to concatenate copies, blur across the seam, then extract only the middle part.

### Parameters
- `start`(int): Start frame of the extraction.
- `end`(int): End frame (exclusive).

### Usage
```python
array = bm.data
bm.append(array); bm.append(array)                 # triple
bm.applyTimeBlur(100)                              # blur across seams
bm.arrayExtract(array.shape[0], array.shape[0]*2)  # take the middle
```

## `arrayReflection`
Appends a time-reversed copy of `data`, creating a there-and-back (mirror) structure. No parameters.

### Usage
```python
bm.addTrans(150)
bm.arrayReflection()   # 300-frame round trip
```

## `wide_expandB`
Extrapolates the motion of the edge slits to extend the spatial dimension of `data` by `add_size` on both sides. Used to prepare wide/panorama output larger than the input video.

### Parameters
- `add_size`(int, optional, default: `3840`): Number of slits added per side.
- `sclip`(bool, optional, default: `True`): Clip spatial coordinates to `0..scan_nums-1`.
- `zclip`(bool, optional, default: `True`): Clip time coordinates to `1..count-1`.
- `spacedirection`(bool, optional, default: `True`): Direction used when converting out-of-range spatial motion into time motion at the edges.
- `z_offset`(int, optional, default: `0`): Time offset added per slit step. Produces a gradient where the right edge is in the future and the left edge in the past (matches scenery flowing right to left).

### Usage
```python
bm.addCycleTrans(300)
bm.wide_expandB(add_size=1920, z_offset=1)
```

## `interpolation_append`
Smoothly interpolates between the tail of `data` and the head of `maneuver` over `connection_num` frames, then concatenates.

### Parameters
- `maneuver`(ndarray): Maneuver array to connect.
- `connection_num`(int): Number of transition frames.
- `speed_round`(bool, optional, default: `False`): Cosine-eased transition.
- `add_maneuver`(bool, optional, default: `True`): If `False`, only the transition is added.

## `interpolation_append_byspeed`
Connects to `maneuver` computing the number of transition frames from a given speed, and automatically blurs the junction in time/space.

### Parameters
- `maneuver`(ndarray): Maneuver array to connect.
- `frame_speed`(float): Transition speed in source frames per output frame. Transition length = time gap / frame_speed.
- `speed_round`(bool, optional, default: `False`): Cosine ease.
- `add_maneuver`(bool, optional, default: `True`): Whether to append `maneuver` itself.
- `sblur`(bool, optional, default: `True`) / `tblur`(bool, optional, default: `True`): Apply spatial/temporal blur to the junction.
- `blur_range`(int, optional, default: `None`): Blur range. None = 1/6 of total length.

### Usage
```python
bm.interpolation_append_byspeed(next_array, frame_speed=4)
```

## `addFlat`
Adds a flat cross-section: space = sequential (a normal frame), time = a fixed value (i.e. a frozen normal frame), for `frame_nums` frames. Creates `data` if it is empty.

### Parameters
- `frame_nums`(int): Number of frames to add.
- `z_pos`(int, optional, default: `0`): Referenced time position (frame number).
- `z_autofit`(bool, optional, default: `True`): If existing data is present, matches the mean time of the tail (or head when `prepend=True`).
- `prepend`(bool, optional, default: `False`): Add to the head instead.
- `flip`(bool, optional, default: `False`): Reverse the spatial order.

### Usage
```python
bm.addFlat(60, z_pos=1200, z_autofit=False)
```

![addFlat (3dplot example)](images/doc_addFlat_3dplot.gif)

## `addSlicePlane`
Generates a slit-scan plane (fixed spatial position, sequential time). Only usable when `data` is empty. The flow of time itself appears within a single frame.

### Parameters
- `frame_nums`(int, optional, default: `1`): Number of frames.
- `xypoint`(float, optional, default: `0.5`): Slit position (0.0-1.0, ratio of scan_nums).
- `full_range`(bool, optional, default: `False`): Extend the time axis to the full input video (0..count).
- `z_start`(float, optional) / `z_end`(float, optional): Directly specify the time range; takes precedence over `full_range`.

### Usage
```python
bm.addSlicePlane(frame_nums=1, xypoint=0.5, full_range=True)
```

## `addFreeze`
Duplicates the last cross-section for `frame_nums` frames (a complete freeze).

### Parameters
- `frame_nums`(int): Number of frames to add.

### Usage
```python
bm.addCycleTrans(240)
bm.addFreeze(60)
```

![addFreeze (3dplot example)](images/doc_addFreeze_3dplot.gif)

## `preExtend`
Extends the head by duplicating the first frame `addframe` times (a frozen intro).

### Parameters
- `addframe`(int): Number of frames to add.

![preExtend (3dplot example)](images/doc_preExtend_3dplot.gif)

## `addExtend`
Extends the tail by duplicating the last frame `addframe` times (time rate becomes 0). Nearly identical to `addFreeze`, plus a spatial mirror option.

### Parameters
- `addframe`(int): Number of frames to add.
- `flip`(bool, optional, default: `False`): Extend with a spatially mirrored cross-section.

![addExtend (3dplot example)](images/doc_addExtend_3dplot.gif)

## `addCylinderCut`
Cuts the XYT cube with a cylinder surface and adds one closed-loop frame whose start and end coincide. Samples a circle/ellipse on the (space, time) plane with space=sin(θ), time=cos(θ).

### Parameters
- `center_time`(float): Time coordinate of the circle center (frame number).
- `center_pos`(float, optional, default: `0.5`): Spatial center (0-1).
- `time_scale`(float, optional, default: `1.0`): Time-axis scale. 1.0 = a circle relative to spatial range × `xyt_boxel_scale`; 0.5 = wide ellipse; 2.0 = tall ellipse.
- `phase`(float, optional, default: `0.0`): Start angle offset in radians.
- `output_width`(int, optional, default: `None`): Number of output slits. None = `int(π × spatial extent)`.

### Usage
```python
bm.addCylinderCut(center_time=1800, time_scale=1.0)
bm.transprocess()   # rendered as a single still frame
```

## `addBoxUnfoldCut`
Cuts the XYT cube along the unfolded four edges of a box, producing one closed-loop frame: normal frame → right-edge slit scan → flipped normal frame → left-edge slit scan.

### Parameters
- `center_time`(float): Time coordinate of the box center.
- `center_pos`(float, optional, default: `0.5`): Spatial center (0-1).
- `time_scale`(float, optional, default: `1.0`): Time-axis scale. 1.0 = square unfold.
- `output_width`(int, optional, default: `None`): Number of output slits. None = computed from the perimeter.

## `addTrans`

The `addTrans` method is a function to add a new trans maneuver trace to `wr_array` and return it. This method performs the transformation considering cyclic angle changes over a specific number of frames.

### Parameters
- `frame_nums`(int): Number of frames to add.
- `start_line`(float, optional, default: `0`): Transformation's starting line.
- `end_line`(float, optional, default: `1`): Transformation's end line.
- `speed_round`(bool, optional, default: `True`): Specifies whether the speed is smooth or not.
- `zd`(bool, optional, default: `True`): Direction setting.

### Usage Example
```python
bm.addTrans(100, start_line=0, end_line=1, speed_round=True, zd=True)
```
![Alt text](images/sample_2023_0618_Vslit+Transposition100_3dPlot.gif)
![Alt text](images/sample_2023_0618_Hslit+Transposition100_3dPlot.gif)

### Sample
[![Waves Etude [TYX-60pps 2009061303]](https://i.vimeocdn.com/video/956287334-20be93368aef7ad17c1bec20c9973f9f66c296056fbe1950d8cfa7200c9a0f11-d_640)](https://vimeo.com/457262317)

[Click to watch on Vimeo](https://vimeo.com/457262317)


## `addBlowupTrans`

`addBlowupTrans` method behaves similarly to `addTrans`, but allows the resolution of the time axis to be transitioned using the provided keyframes.

### Parameters
- `frame_nums`(int): Number of frames to add.
- `deg`(int, optional, default:360):set scan direction movement `360` for round trip. One way at `180`.
- `speed_round`(bool, optional, default: `True`): whether the transition in the scan direction is smooth or not.
- `connect_round`(int, optional, default: `1`): whether to smooth the movement between keyframes.
- `timevalues`(list, optional): list of keyframe values. The time range is specified in frames.
- `timepoints`(list, optional):list of keyframe times for `frame_nums`, specified as a ratio of 0~1.
- `timecenter`(list, optional):a list of center points for the keyframe time range transition. If not provided, it defaults to 0.5 for each keyframe.

### Usage Example
```python
bm.addBlowupTrans(frame_nums=100, deg=360, speed_round=True, connect_round=1,timevalues=[bm.count,bm.scan_nums,1,0], timepoints=[0,0.7,0.95,1], timecenter=[0.5,0.5,0.5,0.5])
```
![Alt text](images/sample_2023_0618_Vslit+addBlowupTrans_3dPlot.gif)
![Alt text](images/sample_2023_0618_Hslit+addBlowupTrans_3dPlot.gif)

### Advanced Example
```python
bm.outfps=60
bm.addBlowupTrans(addnum,1080,timevalues=[int(bm.width),int(bm.width)],timepoints=[0,1],timecenter=[0.5,0.5])
bm.applyTimeSlide(14280)
bm.applyTimebySpace(int(6*bm.recfps))#Left to Right TimeGap(sec)
```
<!-- [![Practice for Time and Space, 2023](https://i.vimeocdn.com/video/1736848219-6e1ced91a2ccc3465c8dfcefc3ed159f497845e7bdd47d535068d51cafc2c7a6-d_640)](https://vimeo.com/873371964) -->

[![Practice for Time and Space, 2023](https://i.vimeocdn.com/video/1736848219-6e1ced91a2ccc3465c8dfcefc3ed159f497845e7bdd47d535068d51cafc2c7a6-d_640)](https://vimeo.com/873371964)

[Click to watch on Vimeo](https://vimeo.com/873371964)


### Sample
[![Waves Etude[TYX 1.8-30pps 201113]](https://i.vimeocdn.com/video/1232415523-ea74b1b1dc98ff5220a9fb91892c0d6f5808a8cb3a30b91647b0878233c98cdc-d_640)](https://vimeo.com/597510638)

[Click to watch on Vimeo](https://vimeo.com/597510638)


## `addKeepSpeedTrans`
Estimates the spatial/temporal velocity vector and acceleration from the last 2-3 frames of `data`, then keeps extrapolating that motion into new frames. Acceleration 1 = constant speed, <1 = converging, >1 = accelerating. Stops when the spatial position crosses `under_xyp`/`over_xyp` or after `frame_nums` frames.

### Parameters
- `frame_nums`(int): Maximum number of frames to add.
- `under_xyp`(float, optional, default: `None`): Upper spatial limit. None = `scan_nums`.
- `over_xyp`(float, optional, default: `1`): Lower spatial limit.
- `rendertype`(int, optional, default: `0`): `0` = extrapolate with the slit-averaged vector, `1` = per-slit vectors.

### Usage
```python
bm.addTrans(100)
bm.addKeepSpeedTrans(200)   # coast on at the terminal velocity of addTrans
```

![addKeepSpeedTrans (3dplot example)](images/doc_addKeepSpeedTrans_3dplot.gif)

## `addInsertKeepSpeedTrans`
An extension of `addKeepSpeedTrans`. Computes the intersection of the tail velocity of `data` and the head velocity of `after_array`, inserts bridging frames, then concatenates `after_array` (which is auto time-slid).

### Parameters
- `frame_nums`(int): Approximate number of inserted frames.
- `under_xyp`(float, optional, default: `None`): Upper spatial limit. None = `width-1`.
- `over_xyp`(float, optional, default: `1`): Lower spatial limit.
- `after_array`(ndarray): Maneuver array to connect to.
- `rendertype`(int, optional, default: `0`): `0` = averaged vector, `1` = per slit.

## `addWideKeyframeTrans`
Expands `scan_nums` by `wide_scale`, then transitions over `frame_nums` frames to each keyframe (spatial position, time position) in `key_array`. For outputs wider than the input video, as in Mid Tide.

### Parameters
- `frame_nums`(int): Transition frames per keyframe.
- `key_array`(list): List of keyframes `[(end_line, end_time), ...]`.
- `wide_scale`(int, optional, default: `3`): Output width multiplier.
- `start_frame`(list, optional, default: `[0,0]`): Start `[space, time]` when `data` is empty.
- `speed_round`(bool, optional, default: `False`): Eased transition.

### Usage
```python
bm.addWideKeyframeTrans(300, key_array=[(1920, 0), (3840, 600)], wide_scale=3)
```

![addWideKeyframeTrans (3dplot example)](images/doc_addWideKeyframeTrans_3dplot.gif)

## `addInterpolation` 

The `interpolation` method is based on the specified maneuver data to perform interpolation and to generate new frames. This function is designed to apply complex transformations over a specific number of frames.

### Parameters
- `frame_nums`(int): Number of frames to add.
- `i_direction`(bool): Direction of interpolation.
- `z_direction`(bool): Direction of interpolation in the Z axis.
- `axis_position`(bool): Central position for rotation and interpolation.
- `s_reversal`(bool, optional, default: `0`): Reversal of spatial dimension.
- `z_reversal`(bool, optional, default: `0`): Reversal of time dimension.
- `cycle_degree`(int, optional, default: `90`): Angle per cycle.
- `extra_degree`(int, optional, default: `0`): Specifies the sectional angle at the initial stage of transformation.
- `zslide`(int, optional, default: `0`): Amount of slide in the Z direction.
- `speed_round`(bool, optional, default: `True`): Specifies if the speed is smooth.
- `rrange`(list of int, optional, default: `[0,1]`): List specifying the range of transformation.

| Parameters | i_direction                                     | z_direction                               | Axis_position                            | s_reversal                               | z_reversal                               |
|---------------|-------------------------------------------------|-------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|
| Type          | Bool                                            | Bool                                      | Bool                                     | Bool                                     | Bool                                     |
| Content       | Direction of transition                        | Direction of rotation relative to time dimension | Whether the axis is at the end or the beginning | Reversal of spatial direction             | Reversal of time direction               |
| Description   | <ul><li>False / (TY-X) -> (XY-T)</li><li>True / (XY-T) -> (TY-X)</li></ul> | <ul><li>False / Forward</li><li>True / Reverse</li></ul> | <ul><li>False / Start</li><li>True / End</li></ul> | <ul><li>False / No reversal</li><li>True / Reverse</li></ul> | <ul><li>False / No reversal</li><li>True / Reverse</li></ul> |


### Usage
```python
bm.addInterpolation(100, 0, 0, 0,s_reversal=False,z_reversal=False)
```
![Alt text](images/sample_2023_0618_Vslit+Interpolation100(ID0-ZD0-AP0-REV0)_3dPlot.gif)
```python
bm.addInterpolation(100, 0, 1, 1,s_reversal=True,z_reversal=True)
```
![Alt text](images/sample_2023_0618_Vslit+Interpolation100(ID0-ZD1-AP1-REV1)_3dPlot.gif)

```python
bm.addInterpolation(100, 0, 0, 0,s_reversal=False,z_reversal=True)
```
![Alt text](images/sample_Vslit_IP180(ID0-ZD0-AP0-SREV0-ZREV1)_3dPlot.gif)


## `rootingA_interporation`
A preset that chains `addInterpolation` in pairs `loop_num` times, producing a zig-zag trajectory that shuttles between normal frames and slit-scan planes. Each segment is `FRAME_NUMS/(loop_num*2)` frames.

### Parameters
- `FRAME_NUMS`(int): Approximate total frame count.
- `loop_num`(int, optional, default: `2`): Number of round trips.
- `axis_first_p`(int, optional, default: `0`): Initial axis position (0/1 = which edge).
- `speed_round`(bool, optional, default: `True`): Eased motion.
- `interval_nums`(int, optional, default: `0`): Frozen (addExtend) frames inserted between segments.
- `loopinterval_nums`(int, optional, default: `0`): Frozen frames inserted between loops.

### Usage
```python
bm.rootingA_interporation(1200, loop_num=2, interval_nums=60)
```

![rootingA_interporation (3dplot example)](images/doc_rootingA_interporation_3dplot.gif)

## `rootingA_interporation_single`
A single-unit version of rootingA consisting of two sub-segments. `seg_type` (0-3) selects the combination of spatial direction, axis position, and time direction.

### Parameters
- `FRAME_NUMS`(int): Total frames of the unit.
- `seg_type`(int, optional, default: `0`): `0` = fwd/rev (+1 time), `1` = rev/fwd (-1), `2`/`3` = axis variants.
- `speed_round`(bool, optional, default: `True`): Eased motion.
- `interval_nums`(int, optional, default: `0`): Frozen frames between sub-segments.
- `panorama_nums`(int, optional, default: `0`): `addExtend` (panorama freeze) frames inserted in the middle.
- `flip_axis`(bool, optional, default: `False`): Flip the axis.
- `junction_mode`(int, optional, default: `0`): `1` = blur the junction for smoothness.
- `blur_rate`(int, optional, default: `90`): Blur strength for junction_mode=1.
- `Second_FRAME_NUMS`(int, optional): Separate frame count for the second half.
- `center_time_frame`(int, optional): If set, slides the junction frame's time to this value via `zArange`.

### Usage
```python
bm.rootingA_interporation_single(600, seg_type=0, panorama_nums=120, junction_mode=1)
```

![rootingA_interporation_single (3dplot example)](images/doc_rootingA_interporation_single_3dplot.gif)

## `rootingA_interporation_trans_single`
The trans-extended variant of `rootingA_interporation_single`: replaces the panorama freeze with a spatial move via `addTrans`.

### Parameters
Shares the parameters of `rootingA_interporation_single`, plus:
- `trans_nums`(int, optional, default: `0`): Frames of the inserted addTrans.
- `trans_end_line`(float, optional, default: `0`): Target position of addTrans (0-1).
- `time_flip`(bool, optional, default: `False`): Reverse the time direction.

![rootingA_interporation_trans_single (3dplot example)](images/doc_rootingA_interporation_trans_single_3dplot.gif)

## `rootingA_interporation_RANDOM`
A randomized rootingA where each parameter is given as a range. Reproducible with `seed`.

### Parameters
- `FRAME_NUMS_range`(tuple|int): Range `(min,max)` or fixed value for segment frames.
- `interval_nums_range`(tuple|int, optional, default: `(0,0)`): Range for inter-segment freezes, randomized every loop.
- `loopinterval_nums_range`(tuple|int, optional, default: `(0,0)`): Range for inter-loop freezes.
- `loop_num`(int, optional, default: `2`) / `axis_first_p`(int) / `speed_round`(bool): As in rootingA.
- `seed`(int, optional): Random seed.
- `clamp_even`(bool, optional, default: `False`): Round to even numbers.
- `randomize_loopinterval_each_loop`(bool, optional, default: `False`): Randomize the loop interval every loop as well.
- `min_step_frames`(int, optional, default: `1`): Lower bound to keep segments non-empty.

### Usage
```python
bm.rootingA_interporation_RANDOM((300, 900), interval_nums_range=(0, 120), loop_num=4, seed=42)
```

![rootingA_interporation_RANDOM (3dplot example)](images/doc_rootingA_interporation_RANDOM_3dplot.gif)

## `rootingAA_interporation`
A rootingA variant that keeps the same spatial start/end point, producing a loop-like motion anchored on screen. Params: `FRAME_NUMS`(int), `loop_num`(int, default 2), `axis_first_p`(int), `speed_round`(bool).

![rootingAA_interporation (3dplot example)](images/doc_rootingAA_interporation_3dplot.gif)

## `rootingB_interporation`
With the axis fixed (`axis_fix_p`), the cross-section topples over continuously like a domino rolling down a slope. Params: `FRAME_NUMS`(int), `loop_num`(int, default 1), `axis_fix_p`(int: 0/1).

## `rooting8_interporation`
A preset that runs 8 connected `addInterpolation` patterns in sequence. Each segment is `FRAME_NUMS` frames, so the total is `FRAME_NUMS×8`. Single parameter: `FRAME_NUMS`(int).

### Usage
```python
bm.rooting8_interporation(150)   # 150×8 = 1200 frames
```

![rooting8_interporation (3dplot example)](images/doc_rooting8_interporation_3dplot.gif)

## `rooting8B_interporation`
The mirrored variant of `rooting8_interporation` (runs with s/z_reversal=1). Single parameter: `FRAME_NUMS`(int).

![rooting8B_interporation (3dplot example)](images/doc_rooting8B_interporation_3dplot.gif)

## `rooting4C_interporation`
A 4-pattern connected preset (two forward + two axis variants). Single parameter: `FRAME_NUMS`(int). Total = `FRAME_NUMS×4`.

![rooting4C_interporation (3dplot example)](images/doc_rooting4C_interporation_3dplot.gif)

## `rooting4D_interporation`
A 4-pattern preset variation alternating mirrored segments. Single parameter: `FRAME_NUMS`(int).

![rooting4D_interporation (3dplot example)](images/doc_rooting4D_interporation_3dplot.gif)

## `addCycleTrans`

The `addCycleTrans` method is used to add cyclical transformations (trans) to the data. It performs transformations considering cyclical angle changes over a specific number of frames.

### Parameters
- `frame_nums`(int): Number of frames to add.
- `cycle_degree`(int, optional, default: `360`): Angle per cycle.
- `zscaling`(bool, optional, default: `False`): Whether to enable scaling in the Z axis.
- `zslide`(int, optional, default: `0`): Initial value of time position.
- `extra_degree`(int, optional, default: `0`): Specifies the sectional angle at the initial stage of transformation.
- `speed_round`(bool, optional, default: `True`): Specifies if the speed is smooth.

### Usage
```python            
bm.addCycleTrans(100, cycle_degree=360, zscaling=True, zslide=10, extra_degree=5, speed_round=False)
```
![Alt text](images/sample_2023_0618_Vslit+CycleTrans360-zscale0_3dPlot.gif)

### Sample
[![Waves Etude [central v-axis rotation180 202209041744]](https://i.vimeocdn.com/video/1507673804-9e1d545fc365e48c6cb8d3bf5fdb7772843aaad792731d1537ba639806240d1e-d_640)](https://vimeo.com/749807843)

[Click to watch on Vimeo](https://vimeo.com/749807843)


## `addCustomCycleTrans`

The `addCustomCycleTrans` class function allows you to freely move the central axis in `addCycleTrans`. It linearly changes from the position specified in the `start_center` argument to the position specified in the `end_center` argument.

### Parameters
- `frame_nums`(int): Number of frames to add.
- `cycle_degree`(int): Angle per cycle.
- `start_center`(float, optional, default: `1/2`): Position of the starting central axis. Specify as a ratio to the scan direction length in the range of (0~1). For vertical slits, 0 is the left end, and 1 is the right end.
- `end_center`(float, optional, default: `1/2`): Position of the ending central axis.
- `zscaling`(bool, optional, default: `False`): Whether to enable scaling in the Z axis.
- `extra_degree`(int, optional, default: `0`): Specifies the sectional angle at the initial stage of transformation.
- `speed_round`(bool, optional, default: `True`): Specifies if the speed is smooth.
- `zslide`(int, optional, default: `0`): Initial value of time position.
- `auto_zslide`(bool, optional, default: `True`): Flag to switch the automatic adjustment of Z slide on/off.
- `zscaling_v`(float, optional, default: `0.9`): Coefficient of Z scaling.

### Usage
```python
bm.addCustomCycleTrans(100, cycle_degree=360, start_center=0.2, end_center=0.7)
```
![Alt text](images/sample_2023_0618_Vslit+CustomCycleTrans360-zscale0_3dPlot.gif)

### Sample
[![GX010182_2023_0703_Vslit+CustomCycle--7play_4_1](https://i.vimeocdn.com/video/1710602324-16f6e6c0569e43a17f42400498f679e55570f513f34add67f255da50fe445479-d_640)](https://vimeo.com/842051869)

[Click to watch on Vimeo](https://vimeo.com/842051869)


## `addWideCustomCycleTrans`
The wide-output version of `addCustomCycleTrans`. While gradually moving the rotation axis, the edge slits outside the center are displayed with time offsets, unfolding the rotating cross-section on a screen wider than the input video.

### Parameters
- `frame_nums`(int): Frames to add.
- `cycle_degree`(int): Rotation angle.
- `start_center`(float) / `end_center`(float): Start/end position of the rotation center (0-1).
- `maxz_range`(int, optional, default: `None`): Maximum time range. None = total input frames (`count`).
- `wide_scale`(int, optional, default: `3`): Output width multiplier.
- `t_auto_scaling`(bool, optional, default: `False`): Automatic time scaling.
- `extra_degree`(int, optional, default: `0`): Start-angle offset.
- `speed_round`(bool, optional, default: `True`): Eased rotation.

### Usage
```python
bm.addWideCustomCycleTrans(600, cycle_degree=360, start_center=0.3, end_center=0.7, wide_scale=3)
```

![addWideCustomCycleTrans (3dplot example)](images/doc_addWideCustomCycleTrans_3dplot.gif)

## `addFixWideCycleTrans`
A fixed-width (`width×wide_scale`) wide cycle trans. Starts rotating from a 90-degree offset so the whole section always fits within the wide screen.

### Parameters
- `frame_nums`(int): Frames to add.
- `cycle_degree`(int): Rotation angle.
- `wide_scale`(int, optional, default: `3`): Output width multiplier.
- `t_auto_scaling`(bool, optional, default: `True`): `True` = time range spans the whole input (`count`); `False` = based on spatial size.
- `extra_degree`(int, optional, default: `0`): Angle offset.
- `speed_round`(bool, optional, default: `True`): Eased rotation.

![addFixWideCycleTrans (3dplot example)](images/doc_addFixWideCycleTrans_3dplot.gif)

## `addWaveTrans`

The `addWaveTrans` method generates a playback cross-section with dynamic waveforms. When `flow` is set to `True`, it will also move the spatial axis.

### Parameters
- `frame_nums`(int): Number of frames to be added.
- `cycle_degree`(float): Wavelength of the wave. Setting it to `360` produces a 2Hz wave relative to the scan direction length, and `180` for a 1Hz wave.
- `zdepth`(float): Amplitude of the wave in the T dimension of XYT.
- `flow`(bool, default: `True`): Specifies whether to move the spatial dimension. Set to `True` to enable movement.
- `zslide`(float, default: `0`): Shift in the T dimension of XYT.
- `speed_round`(bool, optional, default: `True`): Whether to smoothly transition dynamically.

### Usage Example
```python
bm.addWaveTrans(frame_nums=8000, cycle_degree=90, zdepth=1500, flow=False)
```
![Sample Vslit](images/sample_2023_0618_Vslit+WaveTrans180xfix_3dPlot.gif)
![Saple Hslit](images/sample_2023_0618_Hslit+WaveTrans180yfix_3dPlot.gif)

```python
bm.addWaveTrans(frame_nums=8000, cycle_degree=90, zdepth=1500, flow=True)
```
![Sample Vslit](images/sample_2023_0618_Vslit+WaveTrans180xflow_3dPlot.gif)
![Saple Hslit](images/sample_2023_0618_Hslit+WaveTrans180yflow_3dPlot.gif)

### Sample
[![20220106_RFS1459-4K_Vertical_wavetrans_8000seq_90deg-Wave-Flow1500zdepthsep_index0](https://i.vimeocdn.com/video/1343615656-2079087154dd4d972a213dff4f14eb93dd838f95d9ba485c148d8ab121bfa6ed-d_640)](https://vimeo.com/663872580)

[Click to watch on Vimeo](https://vimeo.com/663872580)


## `addEventHorizonTrans`
Keeps the spatial domain unchanged (when `flow=False`) and creates a time-warp cross-section where time flows at different speeds between the screen center and the periphery. With `cycle_degree=180` the center advances while the periphery reverses and gradually returns; `360` additionally passes through a full inversion. The amplitude oscillates with the envelope `sin(π·z_osc·i/N)`.

### Parameters
- `frame_nums`(int): Frames to add.
- `zdepth`(float): Maximum time amplitude in frames.
- `z_osc`(int, optional, default: `1`): Number of envelope oscillations.
- `cycle_degree`(int, optional, default: `180`): Wavelength of the on-screen time warp.
- `flow`(bool, optional, default: `False`): Also move the spatial domain.
- `zslide`(int, optional, default: `0`): Time offset.

### Usage
```python
bm.addEventHorizonTrans(300, zdepth=600, cycle_degree=180)
```

![addEventHorizonTrans (3dplot example)](images/doc_addEventHorizonTrans_3dplot.gif)

## `applyTimeFlowKeepingExtend`
This method extends the current maneuver data (`self.data`) by prepending/appending frames. The way it extends is distinctive: **the spatial dimension (`data[:,:,0]`) holds the position of the first/last frame (stays still)**, while **the time dimension (`data[:,:,1]`, z) is extended keeping the rate of change at the boundary (i.e. the speed at which time flows)**.
Use it to add "keep flowing at the same speed" run-up / coast-down sections before and after a motion created by addTrans or addCycleTrans. Setting `fade=True` smoothly changes the speed toward `fade_speed` within the extension (useful for intro/outro).

### Parameters
- `frame_nums`(int): Number of frames to extend by. It is applied to `intro` and `outro` separately, so when both are `True` the total increases by `frame_nums × 2` frames.
- `fade`(bool, optional, default: `False`): If `True`, ease the speed in the extension toward `fade_speed`. If `False`, extend at constant speed, keeping the boundary speed.
- `intro`(bool, optional, default: `True`): If `True`, prepend (extend backward).
- `outro`(bool, optional, default: `True`): If `True`, append (extend forward).
- `fade_speed`(float, optional, default: `0`): Target speed when `fade=True`. **The unit is "how much z advances per one output frame" (= playback rate × `recfps/outfps`)**. `0` is a full stop, `recfps/outfps` is constant real speed (playback rate 1.0).
- `fade_type`(str, optional, default: `"inout"`): Easing type, `"inout"` / `"in"` / `"out"`. Note: currently the outro side uses `inOutQuad` regardless of this setting (the intro side honors it).
- `space_apply`(bool, optional, default: `False`): If `True`, extend the spatial dimension too, keeping the boundary rate of change. `False` (default) holds the space still.

### Usage Example
```python
# After addTrans, add a 60-frame "keep flowing at the same speed" section before and after
bm.addTrans(100)
bm.applyTimeFlowKeepingExtend(60)

# Extend only the tail and decelerate to a smooth stop there (for an outro)
bm.applyTimeFlowKeepingExtend(90, fade=True, intro=False, outro=True)

# Accelerate from a stop into the main part at the head (for an intro)
bm.applyTimeFlowKeepingExtend(90, fade=True, intro=True, outro=False, fade_type="in")
```

> If you need to land exactly on a specific time position (z), use the coordinate-based versions that have zero accumulated error: [`applyTimeFlowKeepingExtend_CoodinateBase_Intro`](#applytimeflowkeepingextend_coodinatebase_intro) / [`applyTimeFlowKeepingExtend_CoodinateBase_Outtro`](#applytimeflowkeepingextend_coodinatebase_outtro).


![applyTimeFlowKeepingExtend (3dplot example)](images/doc_applyTimeFlowKeepingExtend_3dplot.gif)

## `applyTimeFlowKeepingExtend_CoodinateBase_Intro`
Coordinate-based intro extension. Prepends `num_frames` linearly interpolated frames so that the head starts exactly at time `target_z`. The per-slit step is computed as `(data[0,slit,1] - target_z) / num_frames`, guaranteeing zero accumulated error.

### Parameters
- `target_z`(float): Time coordinate (frame number) the new head frame should reference.
- `num_frames`(int): Number of extension frames.

### Usage
```python
bm.applyTimeFlowKeepingExtend_CoodinateBase_Intro(target_z=0, num_frames=150)
```

![applyTimeFlowKeepingExtend_CoodinateBase_Intro (3dplot example)](images/doc_applyTimeFlowKeepingExtend_CoodinateBase_Intro_3dplot.gif)

## `applyTimeFlowKeepingExtend_CoodinateBase_Outtro`
Coordinate-based outro extension. Appends `num_frames` linearly interpolated frames so that the tail lands exactly on time `target_z`, with zero accumulated error.

### Parameters
- `target_z`(float): Time coordinate the new last frame should reference.
- `num_frames`(int): Number of extension frames.

![applyTimeFlowKeepingExtend_CoodinateBase_Outtro (3dplot example)](images/doc_applyTimeFlowKeepingExtend_CoodinateBase_Outtro_3dplot.gif)

## `applyTimeForward`
Adds a forward flow of time to the whole array: frame k gets `slide_time × k` added to its time coordinate.

### Parameters
- `slide_time`(float, optional, default: `None`): Source frames advanced per output frame. None = `outfps/recfps`. For real-time speed (rate 1.0) pass `recfps/outfps`.
- `start_frame`(int, optional, default: `0`): First frame the flow applies to.
- `end_frame`(int, optional, default: `None`): Last frame; later frames get the final shift added uniformly.

### Usage
```python
bm.addFlat(300)
bm.applyTimeForward(4)   # real-time flow for recfps=120 / outfps=30
```

![applyTimeForward (3dplot example)](images/doc_applyTimeForward_3dplot.gif)

## `applyTimeOblique`
Adds a time shift proportional to the slit position (0 to `maxgap` frames), tilting the cross-section diagonally in time.

### Parameters
- `maxgap`(int): Maximum time shift (frames) at the far edge.

![applyTimeOblique (3dplot example)](images/doc_applyTimeOblique_3dplot.gif)

## `applyTimeForwardAutoSlow`
Adds normal-speed (rate 1) intro/outro sections around a slowed-down maneuver and connects them with easing.

### Parameters
- `slide_time`(int, optional, default: `1`): Time advance of the slow part.
- `defaultAddTime`(int, optional, default: `100`): Frames added as intro/outro (at least 5 sec = 150 f).
- `addTimeEasing`(bool, optional, default: `True`): Apply eased connection.
- `easeRatio`(float, optional, default: `0.3`): Ratio of the eased region.

![applyTimeForwardAutoSlow (3dplot example)](images/doc_applyTimeForwardAutoSlow_3dplot.gif)

## `applyTimeLoop`
Reshapes the overall time flow as forward → reversed middle → forward with sine-curve connections, canceling the head/tail time difference to build a seamless loop. The constant-speed sections (`stay_time`) are auto-adjusted based on the gap.

### Parameters
- `slide_time`(float): Time advance of the forward part (frames per output frame).
- `freq`(int, optional, default: `2`): Frequency. Currently only 2 is supported.
- `stay_time`(int, optional, default: `30`): Base length of the constant-speed sections.
- `intepolation_min`(int, optional, default: `300`): Minimum length of the sine sections.
- `stay_time_min`(int, optional, default: `30`): Minimum constant-speed length.

### Usage
```python
bm.addFlat(1200)
bm.applyTimeLoop(4)
```

![applyTimeLoop (3dplot example)](images/doc_applyTimeLoop_3dplot.gif)

## `applyTimeLoopB`
Per-slit version of `applyTimeLoop`: adjusts the constant-speed lengths per slit for maneuvers whose head/tail gap differs per slit. Same parameters (default `stay_time`=90).

![applyTimeLoopB (3dplot example)](images/doc_applyTimeLoopB_3dplot.gif)

## `applyTimeChoppyLoop`
Adds a triangle-wave time offset over the whole array. Time shuttles back and forth, and the ends line up, so it can be looped.

### Parameters
- `slide_time`(float, optional, default: `None`): Amplitude scale. None = `outfps/recfps`.
- `frequency`(int, optional, default: `1`): Number of waves.
- `phase_shift`(float, optional, default: `0`): Phase offset in units of π.
- `rise`(float, optional, default: `0.5`) / `fall`(float, optional, default: `0.5`): Ratio of the rising/falling part.

![applyTimeChoppyLoop (3dplot example)](images/doc_applyTimeChoppyLoop_3dplot.gif)

## `applyTimeChoppyLoopB`
Extended choppy loop with a sine option and vertex blurring that smooths the fold-back corners of the triangle wave.

### Parameters
Shares the parameters of `applyTimeChoppyLoop`, plus:
- `wave_type`(str, optional, default: `'triangle'`): `'triangle'` | `'sine'`.
- `blur`(int, optional, default: `0`): Non-zero applies a loop blur to the waveform.

![applyTimeChoppyLoopB (3dplot example)](images/doc_applyTimeChoppyLoopB_3dplot.gif)

## `applyTimeClip`
Shifts every frame in time so that the trajectory of the specified slit stays fixed at a constant value; the data becomes a relative time structure anchored at that slit.

### Parameters
- `trackslit`(int): Index of the anchor slit.
- `cliptime`(float, optional, default: `None`): Fixed time position. None = 0.

### Usage
```python
bm.applyTimeClip(trackslit=960, cliptime=1800)
```

![applyTimeClip (3dplot example)](images/doc_applyTimeClip_3dplot.gif)

## `applyTimebySpace`
Adds a time shift proportional to each slit's spatial position, up to `v` frames.

### Parameters
- `v`(float): Maximum time shift in frames.
- `mode`(int, optional, default: `0`): `0` = per-slit spatial coordinate, `1` = based on `cycle_axis` (rotation center of CustomCycleTrans), `2` = frame-mean of spatial coordinates.

![applyTimebySpace (3dplot example)](images/doc_applyTimebySpace_3dplot.gif)

## `applyTimebyKeyframetoSpace`
Specifies the space→time-shift mapping with keyframes. The keyframe values are spread evenly across the spatial range and spline-interpolated.

### Parameters
- `keyframes`(list): List of shift amounts in frames, e.g. `[0, 300, 0]` shifts only the center by 300 frames.
- `mode`(int, optional, default: `0`): Same basis selection as `applyTimebySpace`.

![applyTimebyKeyframetoSpace (3dplot example)](images/doc_applyTimebyKeyframetoSpace_3dplot.gif)

## `applyTimeSlide`
Translates the whole data in time so that the center slit of the base frame references `settime`.

### Parameters
- `settime`(int): Target time position (frame number).
- `baseframe`(int, optional, default: `0`): Base frame. `-1` = last frame.

### Usage
```python
bm.applyTimeSlide(0)          # align the head with the start of the input
bm.applyTimeSlide(3600, -1)   # align the tail with frame 3600
```

## `applyInOutGapFix`
Distributes the head/tail time difference linearly over all frames to cancel it (seamless-loop helper). No parameters.

## `applyInFix`
Makes the head frame's time coordinates match `target_z_array`, blending linearly while keeping the tail fixed.

### Parameters
- `target_z_array`(ndarray): Per-slit target time values (shape=(scan_nums,)).

## `applyOutFix`
Makes the tail frame match `target_z_array`, keeping the head fixed; eased by default.

### Parameters
- `target_z_array`(ndarray): Per-slit target time values.
- `ease`(bool, optional, default: `True`): Ease-in-out blending; `False` = linear.

## `applyInPartFix`
Partial fix: adjusts only frames 0..`b_frame` with easing so that the time at `a_frame` equals `target_z`.

### Parameters
- `target_z`(float): Target time.
- `a_frame`(int): Frame to match.
- `b_frame`(int): Frame where the correction fades to zero.

## `applyOutPartFix`
Partial fix on the tail side: adjusts frames from `a_frame` on with easing so that the time at `b_frame` (probe slit `b_frame_s_point`) equals `target_z`.

### Parameters
- `target_z`(float): Target time.
- `a_frame`(int): Blend start frame.
- `b_frame`(int): Frame to match.
- `b_frame_s_point`(int, optional, default: `None`): Probe slit. None = center slit.

## `applyOutPartFixB`
Array version of `applyOutPartFix`: adjusts frames after `a_frame` toward per-slit targets `target_z_array`.

### Parameters
- `target_z_array`(ndarray): Per-slit target time values.
- `a_frame`(int) / `b_frame`(int): Blend range.
- `base_z_array`(ndarray, optional, default: `None`): Baseline for the gap computation. None = `data[b_frame,:,1]`.

## `applySpaceBlur`
Applies a box blur along the frame axis to the spatial coordinates (`data[:,:,0]`), smoothing spatial motion.

### Parameters
- `bl_time`(int): Blur kernel size in frames.

![applySpaceBlur (3dplot example)](images/doc_applySpaceBlur_3dplot.gif)

## `applyTimeBlur`
Applies a box blur along the frame axis to the time coordinates (`data[:,:,1]`), smoothing temporal motion. Internally pads via `timeFlowKeepingExtend`, so the edges keep their velocity.

### Parameters
- `bl_time`(int): Blur kernel size in frames.

![applyTimeBlur (3dplot example)](images/doc_applyTimeBlur_3dplot.gif)

## `applyCustomeBlur`
Applies a weighted-average blur only to the specified frame range — useful to smooth local kinks such as junctions.

### Parameters
- `s_frame`(int) / `e_frame`(int): Frame range.
- `bl_time`(int): Blur kernel size.
- `dim_num`(int, optional, default: `1`): `1` = time, `0` = space.

![applyCustomeBlur (3dplot example)](images/doc_applyCustomeBlur_3dplot.gif)

## `applyLoopBlur`
Triples `data` by concatenation and blurs across it, yielding a blur continuous across loop boundaries (use `arrayExtract` to take out the middle third afterwards).

### Parameters
- `sblur`(int) / `tblur`(int): Spatial/temporal blur kernel size. 0 disables.

### Usage
```python
n = bm.data.shape[0]
bm.applyLoopBlur(0, 100)
bm.arrayExtract(n, n*2)
```

## `applyConnectLoopBlur`
Blurs only the loop junction (the `connect_frame` frames around head and tail). Data length is unchanged.

### Parameters
- `sblur`(int) / `tblur`(int): Blur kernel sizes.
- `connect_frame`(int, optional, default: `100`): Range around the junction.

## `applyPointBlur`
Blurs only a region centered at a given frame.

### Parameters
- `point_frame`(int): Center frame.
- `sblur`(int) / `tblur`(int): Blur kernel sizes.
- `range_frame`(int, optional, default: `100`): Half range around the center.

## `applySpaceFlip`
Flips the spatial coordinates left-right (or top-bottom for horizontal slits). No parameters.

![applySpaceFlip (3dplot example)](images/doc_applySpaceFlip_3dplot.gif)

## `applySpaceFlat`
Resets the spatial coordinates to the initial sequence (0..scan_nums-1), removing spatial distortion while keeping the time warp. No parameters.

### Usage
```python
bm.addCycleTrans(300)
bm.applySpaceFlat()   # keep only the time distortion
```

![applySpaceFlat (3dplot example)](images/doc_applySpaceFlat_3dplot.gif)

## `zCenterArange`
Shifts the whole data so the midpoint of min/max time lands on the input-video center (`count/2`, or a custom value). NaN-safe.

### Parameters
- `center_time_frame`(int, optional, default: `None`): Custom center. None = `count/2`.

## `zArange`
Slides the whole data so that the mean time of the specified frame lands on the target time. Unlike `zCenterArange` (which uses the overall min/max midpoint), this anchors on a specific frame.

### Parameters
- `target_frame`(int): Frame index in `data` used as the anchor.
- `center_time_frame`(float, optional, default: `None`): Target time. None = `count/2`.

## `zStartArange`
Shifts the whole data so the minimum time coordinate becomes 0. No parameters.

## `zPointCheck`
Checks that the time coordinates fit within the valid range (0..`count`) and auto-adjusts if not: shifts when possible, otherwise scales the time axis. Use as a safety check before rendering.

### Parameters
- `subtract_count`(int, optional, default: `0`): Margin on the upper side (`count - subtract_count`).

### Usage
```python
bm.zPointCheck()
bm.transprocess()
```

## `zPointCheckandReflect`
Instead of shifting/scaling, reflects out-of-range time coordinates back at the boundaries.

### Parameters
- `subtract_count`(int, optional, default: `0`): Upper-side margin.

## `spline_interpolate`
Generic helper that interpolates a list of keyframe values (spread evenly across the spatial range) with spline/linear/bezier, returning values at positions `x`. Used internally by `applyTimebyKeyframetoSpace`.

### Parameters
- `x`(ndarray): Positions (slit coordinates) to evaluate.
- `keyframes`(list): List of values, spread evenly over `scan_nums`.
- `method`(str, optional, default: `'spline'`): `'spline'` | `'linear'` | `'bezier'`.

## `dataCheck`
Prints the shape of `data` and the min/max of the spatial/temporal coordinates. No parameters.

## `info_setting`
Thins `data` down to `thread_num` slits and computes the derived maps used by audio/CSV output (`sc_resetPositionMap` = time positions, `sc_rateMap` = playback rates, `sc_inPanMap` = spatial positions, `sc_now_depth` = time depth per frame, etc.). Called automatically by `scd_out`/`maneuver_CSV_out`, but usable on its own.

### Parameters
- `thread_num`(int, optional, default: `20`): Number of slits after thinning.
- `raw`(bool, optional, default: `False`): Use all slits without thinning.

## `maneuver_CSV_out`
Outputs the maneuver data as CSVs with headers readable by After Effects etc. Each boolean flag selects a map to export.

### Parameters
- `thread_num`(int, optional): Slit thinning count.
- `time_map`(bool, default: `True`): Export the time-position map (ResetP).
- `space_map`(bool, default: `True`): Export the spatial-position map (inPanMap).
- `time_rate_map`(bool, default: `True`): Export the playback-rate map.
- `now_depth_map`(bool, default: `False`): Export the per-frame time depth.
- `space_rate_map`(bool, default: `False`): Export the spatial-direction rate.
- `movement_rate_map`(bool, default: `False`): Export the total movement rate.

## `data_save`
Saves the maneuver `data` as an npy file, reusable later via the `datapath` argument or `np.load`.

### Parameters
- `attr`(str, optional, default: `None`): Filename suffix.
- `sep`(int, optional, default: `0`): Non-zero saves the data split into `sep` parts along the frame axis.

### Usage
```python
bm.data_save()
# later: bm = imgtrans.drawManeuver("video.mp4", 1, datapath="saved_data.npy")
```

## `split_3_npySave`
Saves `data` split spatially into Left/Center/Right npy files (for split rendering of wide outputs), plus the concatenated npy. No parameters.

## `split_3_npysavereturn`
Same as `split_3_npySave` but returns the saved file paths as an array. No parameters.

## `vsizeReturn`
Returns the output image size `(width, height)` according to the class variable `img_size_type`. No parameters.

## `maneuver_2dplot`
Outputs 2D plots (time position, rate, etc.) of the maneuver as PNG. Runs automatically after each transform when `auto_visualize_out=True`. With `video_out=True` it also renders a plot video with a seek bar.

### Main parameters
- `thread_num`(int, optional): Number of slits to plot.
- `axnum`(int, optional, default: `3`): Number of graphs to draw (0-7).
- `colormode`(str, optional, default: `'black'`): Color scheme.
- `s_frame`/`e_frame`(int, optional): Frame range to draw.
- `video_out`(bool, optional, default: `False`): Output a seek-bar video.
- `video_alpha`(bool, optional, default: `False`): Transparent video (ProRes 4444).
- `individual_output`(bool, optional, default: `False`): Save each graph as a separate file.

### Usage
```python
bm.maneuver_2dplot(50)
bm.maneuver_2dplot(video_out=True)
```

## `maneuver_3dplot`
Plots the maneuver as 3D cross-sections inside the XYT cube and outputs a rotating animation (image sequence + video).

### Main parameters
- `thread_num`(int, optional): Number of slits to plot.
- `out_framenums`(int, default: `50`) / `out_fps`(int, default: `25`): Output length and fps.
- `colormode`(str, default: `'white'`): Background scheme.
- `aspect_ratio`(tuple, default: `(1,1,1)`): XYZ aspect.
- `elev`/`azim`(float, default: `25`/`-40`): Camera elevation/azimuth.
- `zRangeFix`(bool, default: `False`): Fix the time axis to the full input range.
- `lineplot`(bool, default: `True`) / `vectorplot`(bool, default: `False`): Section lines / motion vectors (with `vector_def_frame`, `velocity`, `vector_color_amp`).
- `only_seq_img`(bool, default: `False`): Output only the image sequence.

## `maneuver_3dplot_midtide`
Mid Tide-style 3D plot preset for wide maneuvers. Parameters shared with the main options of `maneuver_3dplot` (`thread_num`, `zRangeFix`, `out_framenums`, `out_fps`, `colormode`, `aspect_ratio`, `elev`, `azim`, `dpi`).

## `maneuver_imgplot`
Exports the maneuver as 16-bit PNG map images (space / time / rate). Normalization parameters are embedded in the filenames so the maps can be converted back into `data` with [`img_to_maneuver`](#img_to_maneuver) — enabling an edit-in-image-editor round trip.

### Parameters
- `plot_mode`(str, optional, default: `"all"`): `"space"` | `"time"` | `"rate"` or combinations; `"all"` = all three.
- `colormode`(str, optional, default: `'black'`): Preview colors.
- `nticks_x`/`nticks_y`(int, optional, default: `4`): Tick counts.
- `save_png`(bool, optional, default: `True`): Save the 16-bit PNG maps.
- `time_axis`(str, optional, default: `'auto'`): `'auto'` | `'frame'` | `'sec'`.

## `img_to_maneuver`
Reconstructs `data` from the 16-bit PNG space/time maps produced by `maneuver_imgplot`. Normalization is extracted from the filenames automatically (manual override possible).

### Parameters
- `space_img_path`(str) / `time_img_path`(str): Paths to the PNG maps.
- `space_set`(float, optional): Manual spatial normalization range.
- `vrange`(tuple, optional): Manual time range `(vmin, vmax)`.

### Usage
```python
bm.img_to_maneuver("xxx_space_1920.png", "xxx_time_840-2760.png")
bm.transprocess()
```

## `img_to_maneuver_rate_based`
Reconstructs `data` by integrating a rate-map image (playback-speed variation) over time — for a workflow where you paint the rate directly.

### Parameters
- `time_rate_path`(str): Rate-map PNG.
- `space_img_path`(str, optional): Space map. None = auto-generate sequential values.
- `space_set`(float, optional): Spatial normalization range.
- `start_time`(float, optional, default: `0.0`): Integration start time.
- `rate_range`(float, optional): Rate normalization range. None = extracted from the filename.
- `rate_baseline`(float, optional) / `rate_startpoint`(float, optional): Rate baseline adjustments.

## `movement_intensity_analyze`
Measures motion intensity from inter-frame differences of the input video and saves a time-series graph as PNG — source material for designing motion-adaptive playback speed. No parameters.

## `transprocess`
This method performs video rendering.

### Parameters
- `separate_num` (int, optional, default: `1`): Specifies how many divisions to perform the rendering.
- `sep_start_num` (int, optional, default: `0`): Specifies from which division to start when rendering in divisions.
- `sep_end_num` (int or None, optional, default: `None`): Specifies up to which division to render.
- `out_type` (int, optional, default: `1`): Selects the output codec / format via the class constants (`0`=`OUT_STILL`, `1`=`OUT_H264`, `2`=`OUT_H265`, … `7`=`OUT_H265_HW`). See the [`out_type` Reference Table](#out_type-reference-table). `0` writes a still image sequence; the other values choose a video encoder.
- `XY_TransOut` (bool, optional, default: `False`): If True, saves the output video rotated by 90 degrees.
- `render_mode` (int, doptional, default: `0`): Rendering mode. `0` outputs all frames of the maneuver data integrated. `1` outputs only from `sep_start_num` to `sep_end_num`.
- `seqrender` (bool, optional, default: `False`):This mode converts the input video data once into npy sequences before rendering. The longer the video data, the faster the rendering will be exported. However, be aware that it consumes hard disk space.
- `title_atr` (str, optional, default: `None`):Append a string to the output video filename.
### Usage
```python
# Create an instance
bm = imgtrans.drawManeuver("mov/samplevideo.mp4",sd=1)

# Write maneuver data
bm.addTrans(100)
bm.applyTimeForward(1)

# Rendering
bm.transprocess()
```
## `new_transprocess`
Primary video rendering method with HDR support, FFmpeg/PyAV pipeline, and rendering downscale options. Replaces `transprocess` for most use cases.

### Parameters
- `separate_num` (int, optional, default: `None`): Number of divisions for split rendering. Auto-calculated from available memory if None.
- `sep_start_num` (int, optional, default: `0`): Start division index for split rendering.
- `sep_end_num` (int or None, optional, default: `None`): End division index.
- `out_type` (int, optional, default: `1`): Output format. Use class constants listed in the table below.
- `xy_trans_out` (bool, optional, default: `False`): If True, rotates output by 90 degrees.
- `render_mode` (int, optional, default: `0`): `0` = full render, `1` = range only, `2` = npy raw data only (no video), `3` = video from existing tmp only.
- `title_atr` (str, optional, default: `None`): Append a string to the output filename.
- `del_data` (bool, optional, default: `True`): Delete `self.data` before rendering to free memory. If True, `animationout` cannot be called afterward.
- `render_clip_start` (int, optional, default: `0`): Start frame for partial rendering.
- `render_clip_end` (int or None, optional, default: `None`): End frame for partial rendering.
- `slit_step` (int, optional, default: `1`): Downscale slit length dimension by skipping pixels. `2` = 1/2, `3` = 1/3, `4` = 1/4. Values ≤ 0 are treated as 1.
- `scan_step` (int, optional, default: `1`): Downscale scan_nums dimension by subsampling scan lines. Non-destructive — `self.data` is not modified. Values ≤ 0 are treated as 1.

### Usage
```python
# Standard rendering
bm.new_transprocess()

# HDR H.265 output
bm.new_transprocess(out_type=bm.OUT_H265)

# ProRes 422 HQ output
bm.new_transprocess(out_type=bm.OUT_PRORES_422)

# Downscaled preview (1/4 area)
bm.new_transprocess(slit_step=2, scan_step=2)

# Partial rendering (frames 100 to 500)
bm.new_transprocess(render_clip_start=100, render_clip_end=500)
```

### `out_type` Reference Table

| Value | Constant | Codec | Pixel Format | Color Space | Container | Notes |
|-------|----------|-------|-------------|-------------|-----------|-------|
| `0` | `OUT_STILL` | — | — | — | Sequential images | Still image output (jpg/bmp) |
| `1` | `OUT_H264` | libx264 | yuv420p (8bit) | SDR BT.709 | .mp4 | Default. Max 4096×2160 |
| `2` | `OUT_H265` | libx265 | yuv420p10le (10bit) | HDR10 PQ BT.2020 | .mp4 | Max 8192×4320 |
| `3` | `OUT_PRORES_422` | prores_ks | yuv422p10le (10bit) | HDR BT.2020 | .mov | ProRes 422 HQ. No resolution limit |
| `4` | `OUT_PRORES_4444` | prores_ks | yuv444p10le (10bit) | HDR BT.2020 | .mov | ProRes 4444. No resolution limit |
| `5` | `OUT_H265_SDR` | libx265 | yuv422p10le (10bit) | SDR BT.709 | .mp4 | H.265 with SDR color tags |
| `6` | `OUT_PRORES_422_SDR` | prores_ks | yuv422p10le (10bit) | SDR BT.709 | .mov | ProRes 422 HQ with SDR color tags |
| `7` | `OUT_H265_HW` | hevc_videotoolbox | p010le (10bit) | HDR10 PQ/HLG BT.2020 | .mp4 | macOS VideoToolbox hardware encode. Faster than software libx265 |

> **Resolution limits**: H.264 is limited to 4096×2160, H.265 to 8192×4320. For resolutions exceeding these limits (e.g. wide panoramic renders), use ProRes (`OUT_PRORES_422` or `OUT_PRORES_422_SDR`).

> **`rendered_npys_to_mov`**: The standalone function for combining split-rendered npy files also supports all `out_type` values above. Pass `out_type=None` (default) for legacy cv2 mp4v output.
> ```python
> import imgtrans
> imgtrans.rendered_npys_to_mov(
>     out_dir='path/to/output',          # Output path (without extension)
>     npys_path='path/to/tmp',           # Folder containing sep-*.npy files
>     out_fps=60,
>     sep_start=1, sep_end=6,
>     out_type=imgtrans.drawManeuver.OUT_PRORES_422_SDR
> )
> ```

## `pretransprocess`
Quickly renders a preview video with `data` thinned to `outnums` frames at 10 fps — for checking the maneuver before a full render. Aborts with an error message when the time coordinates are out of range.

### Parameters
- `outnums`(int, optional, default: `100`): Number of output frames (data is sampled evenly).
- `xy_trans_out`(bool, optional, default: `False`): Rotate the output by 90 degrees.

### Usage
```python
bm.addCycleTrans(1200)
bm.pretransprocess(100)
```

## `overlay_tc_rate`
Re-exports the rendered video (`out_videopath`) with timecode and playback rate overlaid. The screen is divided into `divisions` sections and the TC/rate of each section's center slit is displayed. Rate colors: yellow (+1), blue (-1), gray (0). Timecode format: `{sec}sec---{frac}f`.

### Parameters
- `output_suffix`(str, optional, default: `None`): Output filename suffix. None = `"_tc"` (`"_tc_textonly"` when text_only).
- `divisions`(int, optional, default: `5`): Number of probe sections.
- `hdr`(bool, optional, default: `None`): None = auto-detect from the input. True = H.265 10bit HDR, False = H.264 8bit SDR.
- `text_nits`(int, optional, default: `203`): Overlay brightness in HDR (nit).
- `text_only`(bool, optional, default: `False`): Discard the video and export text only as transparent ProRes 4444 (for AE compositing).
- `font_scale`(float, optional) / `thickness`(int, optional): None = auto by resolution.
- `font_scale_mult`(float, optional, default: `1.0`): Multiplier applied to the auto-computed values.

### Usage
```python
bm.transprocess()
bm.overlay_tc_rate(divisions=5)
```

## `animationout`

The `animationout` function plots the pixel color of the image on a 3D graph with reference to the output video data and outputs the result as an animation. It allows you to visualize the trajectory of the playback cross section based on spatio-temporal manipulations on the spatio-temporal cube. Unlike 'maveuver_2dplot' and 'maveuver_3dplot', this visualization is more intuitive in its correspondence with the input video data by mapping pixel colors.  
 Therefore, it can only be executed after rendering the video. If there is already video data to be exported, it is necessary to call the information in the instance variable 'out_videopath'.
 
 ```python
 bm.out_videopath = "mov/sample.mp4"
 bm.animationout()
 ```

 ### argument 
- `outFrame_nums` (`int`, default: `100`): Number of frames to output.
- `drawLineNum` (`int`, default: `250`): Number of lines to draw.(resolution)
- `dpi` (`int`, default: `200`):  DPI of the output image.
- `out_fps` (`int`, default: `10`): Output Video Frame Rate.

### Examples of Use 
![Alt text](images/20220106_RFS1459-4K_2023_0930_Vslit_interporationAset+IP2800(rootingA)_CustomeBlur300_CustomeBlur300_TimeLoop_timeSlide_zCenterArranged_img_3d-pixelMap.gif)

## `animationout_custome`
Customizable version of `animationout` with fine control over camera angle, aspect ratio, colors and vector display.

### Parameters
- `zRangeFix`(bool, optional, default: `False`): Fix the time axis to the full input range.
- `out_framenums`(int, default: `100`) / `drawLineNum`(int, default: `250`) / `dpi`(int, default: `200`) / `out_fps`(int, default: `10`): As in `animationout`.
- `aspect_ratio`(tuple, optional, default: `(16, 50, 9)`): XYZ aspect.
- `elev`(float, default: `25`) / `azim`(float, default: `-40`): Camera elevation/azimuth.
- `colormode`(str, optional, default: `'white'`): Background scheme.
- `transparent`(bool, optional, default: `False`): Transparent background.
- `gridplot`(bool, optional, default: `True`): Draw the grid.
- `vectorplot`(bool, optional, default: `False`): Draw motion vectors (with `vector_def_frame`(int, default `120`), `velocity`(float, default `10`), `vector_color_amp`(float, default `1.0`)).
- `s_frame`(int, optional, default: `0`): Start frame.

### Usage
```python
bm.out_videopath = "rendered.mp4"
bm.animationout_custome(elev=30, azim=-60, colormode='black')
```

## Standalone Functions

Module-level functions that do not belong to the class. Call them directly after `import imgtrans`.

### `export_segments`
Splits the source video into segment group A (forward) and group B (hflip + reversed), overlaying frame numbers and timecode. Supports real-time speed correction for high-speed footage (subsampling by the `recfps`/`out_fps` ratio).

- `video_path`(str): Source video path.
- `out_dir`(str): Output directory (creates A/ and B/ subfolders).
- `segment_sec`(float, default: `10`): Source seconds per segment.
- `segment_count`(int, optional): Number of segments. None = whole video.
- `out_fps`(float, default: `60`): Output fps.
- `with_frame_num`(bool, default: `True`): Draw frame numbers.
- `recfps`(float, default: `480`): Actual recording fps.
- `export_only`(str, default: `"both"`): `"A"` | `"B"` | `"both"`.

```python
imgtrans.export_segments("src.mp4", "segments/", segment_sec=10, recfps=480)
```

### `rendered_npys_to_mov`
Unified converter from npy data to a video. Accepts a folder of sep-*.npy files, a single npy, an in-memory array, or a one-npy-per-frame folder (`per_frame=True`), and outputs in any `out_type` format.

- `out_dir`(str): Output directory.
- `npys_path`(str, optional): npy folder or file.
- `out_fps`(int, default: `30`) / `out_type`(int, optional): Output settings.
- `sep_start`(int, default: `0`) / `sep_end`(int, optional): Range of sep files to join.
- `images`(ndarray, optional): Pass an in-memory array directly.
- `per_frame`(bool, default: `False`): One-npy-per-frame layout.
- `separate_out`(bool, default: `False`): Output one video per sep instead.

```python
imgtrans.rendered_npys_to_mov("out/", npys_path="tmp/", sep_start=1, sep_end=6)
```

### `rearrange_wide_video`
Splits a wide panorama video (W×H) at the center and stacks the halves vertically into (W//2)×(2H). `roll_offset` applies a horizontal circular shift. Input is a video file or a sep-*.npy folder.

- `output_path`(str): Output path.
- `input_path`(str, optional) / `npys_path`(str, optional): Input (either one).
- `roll_offset`(int, default: `0`): Horizontal circular shift in pixels.
- `sep_start`/`sep_end`(int, optional): Range for npy input.
- `out_fps`(int, default: `30`) / `out_type`(int, optional): Output settings.

### `rendered_mov_to_seq`
Extracts frames from a rendered video as still images (ffmpeg-based, HDR-color aware).

- `video_path`(str): Input video.
- `divide_num`(int, optional): Even division count.
- `img_format`(str, default: `'jpg'`): `'ultrahdr'` (Gain Map HDR JPEG) | `'png'` (16-bit) | `'jpg'` | `'avif'` (10-bit) | `'npy'`.
- `frame_array`(array, optional): Explicit frame numbers to export.
- `color_mode`(str, default: `'source'`): `'source'` (keep color metadata) | `'sdr'` (tonemap to BT.709) | `'hlg'`.

```python
imgtrans.rendered_mov_to_seq("out.mp4", divide_num=10, img_format='png')
```

### `convert_npy_to_jpg`
Converts a single npy file (a saved frame array) into a JPEG image. Params: `npy_file_path`(str), `output_folder`(str).

### `custom_blur`
Applies a weighted-average blur to the given frame range and dimension of an array and returns it. The internal implementation of `applyCustomeBlur`. Params: `data`(ndarray), `s_frame`(int), `e_frame`(int), `bl_time`(int), `dim_num`(int, default: `1`).

### `custom_onedimention_blur`
Weighted-average blur over a range of a 1-D array (e.g. a time trajectory). Params: `time_array`(ndarray), `s_frame`(int), `e_frame`(int), `bl_time`(int).

### `double_first_dimension_with_interpolation`
Doubles the first (frame) dimension of a 3-D array by inserting linearly interpolated frames between existing ones — usable for frame-rate doubling of maneuver data. Params: `arr`(ndarray), `next_first_array`(ndarray, optional: pass the next head frame for loops).

```python
bm.data = imgtrans.double_first_dimension_with_interpolation(bm.data)
```

## Recent Updates (2026)

### Rendering Downscale Options (`new_transprocess`)
Two new parameters allow rendering at reduced resolution for faster preview or smaller file output, without modifying maneuver data:
- `slit_step` (int, default: `1`): Downscale the slit length dimension by skipping pixels. `2` = 1/2, `3` = 1/3, `4` = 1/4 size. Values of 0 or less are treated as 1.
- `scan_step` (int, default: `1`): Downscale the scan_nums dimension by subsampling scan lines. `2` = 1/2, `3` = 1/3, `4` = 1/4. Non-destructive — `self.data` is not modified.

```python
bm.new_transprocess()                          # Full resolution
bm.new_transprocess(slit_step=2)               # Half slit length (e.g., 2160→1080)
bm.new_transprocess(scan_step=2)               # Half scan nums
bm.new_transprocess(slit_step=2, scan_step=2)  # Both halved (1/4 area)
```

### Seekbar Video Output (`maneuver_2dplot`)
`maneuver_2dplot` can now export an animated video with a moving seekbar overlay:
- `video_out` (bool, default: `False`): When True, outputs a video with a dotted seekbar line and timecode display.
- `video_alpha` (bool, default: `False`): When True, outputs ProRes 4444 MOV with alpha channel instead of MP4.

```python
bm.maneuver_2dplot(video_out=True)                # MP4 with white background
bm.maneuver_2dplot(video_out=True, video_alpha=True)  # ProRes 4444 with transparency
```

### Spatial Expansion with Time Offset (`wide_expandB`)
- `z_offset` (int, default: `0`): Adds a per-step time offset during spatial expansion. Right side gets `+z_offset` per step (future), left side gets `-z_offset` (past). Useful for creating natural-looking train window scenery where the expanded area has a time gradient even when the actual time difference is zero.

```python
bm.wide_expandB(add_size=3840, z_offset=1)
```

### Junction Mode and Blur Rate (`rootingA_interporation_single`, `rootingA_interporation_trans_single`)
- `junction_mode` (int, default: `0`): Selects where the time direction reversal point (junction) is placed.
  - `0`: Default — end of panorama/interval + shift
  - `1`: End of first-half interpolation
  - `2`: Midpoint of panorama/trans section
- `blur_rate` (int, default: `90`): Controls the blur range around the junction as a percentage of the segment frame count. The blur is applied per-segment rather than cumulatively across segments.

```python
bm.rootingA_interporation_single(Fnum, seg_type=0, junction_mode=2, blur_rate=90)
bm.rootingA_interporation_trans_single(Fnum, seg_type=0, junction_mode=2, blur_rate=90, time_flip=False)
```

### Head/Tail Timecode Alignment (`applyInOutGapFix`)
Auxiliary function for seamless loop creation. Calculates the difference between first and last frames' time values and linearly adjusts all frames to match.

### Bug Fix: `applyCustomeBlur` Negative Index
Fixed a bug where `applyCustomeBlur` with `s_frame=0` could produce NaN values in the first ~60 frames due to negative numpy array indexing. The mean slice lower bound is now clamped to 0.

### HDR Color Profile Support (Image Output)

When rendering to image files (`out_type=0`), the color profile of the input video is now automatically embedded in the output.
This is critical for HDR content (PQ / HLG) — without the correct profile, viewers such as macOS Preview will interpret PQ-encoded pixel values as sRGB and display incorrect colors.

**Supported formats and methods:**

| Format | HDR Profile | Method |
|--------|------------|--------|
| PNG | Full (BT.2100 PQ / HLG) | cICP chunk injection after save |
| TIFF | Primaries only (BT.2020) | `sips --embedProfile` with system ICC |
| Others | None | Standard `cv2.imwrite` |

- For PQ (SMPTE ST 2084) sources, PNG output is recommended as it embeds the complete BT.2100 PQ profile via the cICP chunk, which macOS Preview recognizes as "Rec. ITU-R BT.2100 PQ".
- For TIFF, the BT.2020 ICC profile (`ITU-2020.icc`) is embedded via macOS `sips`. This provides correct color primaries but uses a gamma-based TRC rather than PQ.
- The input video's `color_primaries`, `color_transfer`, and `colorspace` are detected automatically via `ffprobe` during initialization.

**Configuration:**
```python
bm.imgtype = ".png"   # Recommended for HDR (default)
bm.imgtype = ".tif"   # TIFF output (BT.2020 primaries only for HDR)
```

The profile is applied automatically by the internal `_save_image_with_profile()` method — no additional user action is needed. The cICP chunk uses ITU-T H.273 code points mapped from ffprobe values (e.g., `bt2020` → 9, `smpte2084` → 16, `arib-std-b67` → 18).

### Still Image Export from a Rendered Video (`rendered_mov_to_seq`)

Beyond the still-image `out_type=0` path (which writes PNG/TIFF/JPEG directly during rendering), the standalone `rendered_mov_to_seq` function extracts frames from an already-rendered video into an image sequence, with several HDR-aware output formats selected via `img_format`:

| `img_format` | Format | Notes |
|---|---|---|
| `'ultrahdr'` | **Ultra HDR JPEG** (Gain Map HDR) | Compatible with iOS 18+, Android 14+, and Instagram. Generated via `ultrahdr_app` (ISO 21496), then converted to Adobe `hdrgm`-compatible form. Requires the `ultrahdr_app` binary. |
| `'png'` | 16bit PNG | Preserves the full HDR gamut; safest option. |
| `'jpg'` | 8bit JPEG | SDR / Ultra HDR (default). |
| `'avif'` | 10bit AVIF | HDR-capable, good browser compatibility. |
| `'npy'` | NumPy array | cv2 fallback. |

`color_mode` controls the color transform: `'source'` keeps the input color-space metadata (default), `'sdr'` tone-maps to BT.709 SDR for maximum compatibility, and `'hlg'` applies a PQ→HLG conversion (requires `zscale`/libzimg, otherwise falls back to `source`).

```python
import imgtrans
imgtrans.rendered_mov_to_seq('/path/to/rendered.mov', img_format='ultrahdr')
```

### Timecode & Rate Overlay (`overlay_tc_rate`)

Overlays timecode and playback rate information onto a rendered video for debugging alignment.

- `divisions` (int, default: `5`): Number of evenly spaced probe points across the frame width. Each probe point displays its slit's timecode and instantaneous playback rate.
- Timecode is displayed in `{sec}sec---{frac}f` format based on `recfps` (e.g., `380sec---12f` for frame 182412 at 480fps).
- Rate text color indicates playback direction: yellow for forward (+1), blue for reverse (-1), gray for near-zero.
- Text is horizontally centered on each probe slit's pixel position.

```python
bm.new_transprocess(del_data=False)
bm.overlay_tc_rate(divisions=5)
```

### Segment Export (`export_segments`)

Standalone function (not a class method) that exports A/B segments from a source video with frame number overlay. Replaces the shell-based `separate.bash` workflow.

- Reads the source video, applies real-time speed correction (`recfps / out_fps` frame stepping), and exports:
  - **A segments**: Forward playback at real-time speed
  - **B segments**: Horizontal flip + reverse playback at real-time speed
- Frame numbers are drawn centered on each frame in `{sec}s{frac}f` format (based on `recfps`).
- Output format: ProRes 422 `.mov`

```python
import imgtrans

imgtrans.export_segments(
    video_path='/path/to/source.mov',
    out_dir='/path/to/output',
    segment_sec=10,       # Real-time seconds per segment
    segment_count=44,     # Number of segments
    out_fps=60,           # Output frame rate
    recfps=480,           # Original recording frame rate
    with_frame_num=True,  # Draw frame numbers
    export_only="both"    # "both", "A", or "B"
)
```

## Contribute
Contributions to this project are welcome.  
If you have any questions or feedback, please feel free to contact us at ryu.furusawa(a)gmail.com.  

## License 
This project is licensed under the MIT License, see the LICENSE.txt file for details
