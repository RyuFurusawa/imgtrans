# imgtrans

This is a programming tool for manipulating time and space of video data.
It is a python library with the DrawManeuver class as the main part.  

映像データの時間と空間を操作するためのプログラミングツールです。
DrawManeuverクラスをメインとしたpythonライブラリです。 

## Contents 目次
- [Overview プロジェクトの概要](#overview-プロジェクトの概要)
- [Background 背景](#background-背景)
- [How to install インストール方法](#how-to-install-インストール方法)
- [Usage Flow 使い方の流れ](#usage-flow-使い方の流れ)
  - [1. Library Import ライブラリのインポート](#1-library-import-ライブラリのインポート)
  - [2. Initialization of `drawManeuver` Class](#2-initialization-of-drawmaneuver-class)
    - [Slit Direction スリットの方向](#slit-direction-スリットの方向)
  - [3. Maneuver Design 軌道のデザイン](#3-maneuver-design-軌道のデザイン)
    - [1. Main Functions for Spatiotemporal Integration 主な時空間統合的な動きを加える関数](#1-MainFunctionsforSpatiotemporalIntegration)
    - [2. Main Functions to Adapt the Flow of Time 主な時間の流れを適応させる関数](#2-MainFunctionstoAdapttheFlowofTime)
    - [Examples of Combinations 実際の組み合わせの例](#ExamplesofCombinations)
    - [Save and Load of Track Data 軌道データの保存と読み込み](#SaveandLoadofTrackData)
  - [4. Visualize ヴィジュアライズ](#4-Visualize)
    - [Slit Color スリットの色](#SlitColor)
    - [2D Plot 2dプロット](#2DPlot)
    - [3D Plot 3dプロット](#3DPlot)
  - [5. Rendering レンダリング](#5-Rendering)
    - [Structure of `data` 軌道データの構造](#Structureofdata)
    - [Video Rendering 映像のレンダリング](#VideoRendering)
    - [Audio Rendering 音声のレンダリング](#AudioRendering)
        - [Four Track Data CSVs 4つの軌道データのCSV](#FourTrackDataCSVs)
        - [Four scd Files 4つのscdファイル](#FourscdFiles)
        - [Execution of scd File scdファイルの実行](#ExecutionofscdFile)
    - [Combining Audio and Video 音声と映像の結合](#CombiningAudioandVideo)
- [drawManeuver Class](#drawManeuverClass)
  - [Class Variables クラス変数](#ClassVariables)
  - [Instance Variables インスタンス変数](#InstanceVariables)
  - [List of All Class Methods 全クラスメソッドのリスト](#ListofAllClassMethods)
- [addCycleTrans](#addCycleTransMethod)
- [addcustomCycleTrans](#addcustomcycletransmethod)
- [transprocess](#transprocessmethod)
- [animationout](#animationoutmethod)
- [Contribute コントリビュート](#contribute)
- [License ライセンス](#license)

<!-- 
## 目次
- [プロジェクトの概要](#プロジェクトの概要)
- [背景](#背景)
- [インストール方法](#インストール方法)
- [使い方の流れ](#使い方の流れ)
  - [1. ライブラリのインポート](#1-ライブラリのインポート)
  - [2. drawManeuverクラスの初期化](#2-drawmaneuverクラスの初期化)
    - [スリットの方向](#スリットの方向)
  - [3. 軌道のデザイン](#3-軌道のデザイン)
    -　[1. 主な時空間統合的な動きを加える関数](#1-主な時空間統合的な動きを加える関数)
    -　[2. 主な時間の流れを適応させる関数](#2-主な時間の流れを適応させる関数)
    - [実際の組み合わせの例](#実際の組み合わせの例)
    - [実際の組み合わせの例](#実際の組み合わせの例)
    - [軌道データの保存と読み込み](#軌道データの保存と読み込み)
  - [4. ヴィジュアライズ](#4-ヴィジュアライズ)
    - [スリットの色](#スリットの色)
    - [2dプロット](#2dプロット)
    - [3dプロット](#3dプロット)
  - [5. レンダリング](#5-レンダリング)
    - [`data`の構造](#dataの構造)
    - [映像のレンダリング](#映像のレンダリング)
    - [音声のレンダリング](#音声のレンダリング)
        - [4つの軌道データのCSV](#4つの軌道データのCSV)
        - [4つのscdファイル](#4つのscdファイル)
        - [scdファイルの実行](#scdファイルの実行)
    - [音声と映像の結合](#音声と映像の結合)
- [drawManeuver クラス](#drawmaneuver-クラス)
  - [クラス変数](#クラス変数)
  - [インスタンス変数](#インスタンス変数)
  - [全クラスメソッドのリスト](#全クラスメソッドのリスト)
- [addCycleTransメソッド](#addcycletransメソッド)
- [addcustomCycleTransメソッド](#addcustomcycletransメソッド)
- [transprocessメソッド](#transprocessメソッド)
- [animationoutメソッド](#animationoutメソッド)
- [コントリビュート](#コントリビュート)
- [ライセンス](#ライセンス) -->

## Overview プロジェクトの概要
By interpreting the time dimension of the video as a three-dimensional depth axis, the video data becomes a cube composed of voxels. Here, this three-dimensional video data is called a spatio-temporal object. The spatio-temporal object has plasticity in that it stores color information determined for all of space and can be retrieved as an image with an array of colors even when cut at various angles and directions. By manipulating the cross-sectional behavior of this spatio-temporal object, a new time- and space-ordered image is created that is different from the time and space of the input image.  
In other words, this tool makes it possible to intervene in the temporal and spatial dimensions of the input image, and the anamorphic relationship between the output image and the input image provides a new perspective on movement and encourages exploration into the perception of time and space installed in the human body senses.  

映像の時間次元を三次元の奥行き軸として解釈することで、映像データはボクセルで構成されるキューブとなります。ここではこの三次元の映像データを時空間オブジェクトと呼ぶこととします。時空間オブジェクトには、空間全てに決定された色情報が蓄積されており、さまざまな角度、方向でカットしても色の配列を持つイメージとして取り出すことができる可塑性があります。この時空間オブジェクトの断面の振る舞いを操作することで、入力映像の時間と空間とは異なる、新たな時間と空間に秩序だてられた映像が作り出されます。  
つまりこのツールにより、入力された映像の時間次元、空間次元へ介入することを可能とし、出力された映像と入力映像とのアナモルフィックな関係性から、動きについての新しい視点をもたらし、人間の身体感覚にインストールされた時空間の知覚への探究を促すものです。

<div style="text-align:center;">
<img src="images/ost-illustrate20220106_RFS1459.gif" alt="illustrate movingimage manipuration by time and space" style="max-width:100%; height:auto;">
</div><br>
<!-- 
本ライブラリでは、①再生断面の軌道のデザインと、その軌道データと入力映像を紐づけて出力する②レンダリングの部分が分かれています。
#### 1. 軌道のデザイン
本ツールは変換操作がモジュール化されており、その組み合わせにより再生断面の軌道をデザインしていきます。（実際には複数のクラス関数を組み合わせます）
様々なモジュールを組み合わせていく度に軌道グラフが出力され、それらを確認しながら編集していくツールとなります。
モジュールは時空間統合的な動きのデザインと時間の流れのデザインの2タイプに別れており、それらの組み合わせにより再生断面の軌道をデザインしていきます。
#### 2. 映像のレンダリング
編集した軌道データとソースとなる映像を紐付けてレンダリングを行います。
PCメモリのRAM容量に応じて、連番イメージの一時ファイルをROMに保存し、最後に統合して映像データとして書き出されます。 
-->

## Background 背景
Development of this tool began in 2020 for filmmaker Ryu Furusawa, for whom several works have been produced.  

本ツールは2020年より映像作家の古澤龍の作品制作のために開発が始められ、いくつかの作品が制作されました。
<ul><li><a href="https://ryufurusawa.com/post/711685011289554944/mid-tide-ryu-furusawa-multi-channel-video">Mid Tide,2023 </a></li>
<li><a href="https://ryufurusawa.com/post/661228499174113280/wavesetude">Waves Etude,2020-2022</a></li>
</ul>
<!--
<div style="text-align:center;">
<img src="images/waves_etude_digest2020-2021 (Original)_00015.jpeg" alt="Waves Etude Ryu Furusawa" style="max-width:100%; height:auto;">
<p><a href="https://ryufurusawa.com/post/661228499174113280/wavesetude">Waves Etude</a> Ryu Furusawa,2020-2022</p>

<img src="images/midtide_exhibitionview1.jpg" alt="Mid Tide 2023 Ryu Furusawa" style="max-width:100%; height:auto;">
<p><a href="https://ryufurusawa.com/post/711685011289554944/mid-tide-ryu-furusawa-multi-channel-video">Mid Tide </a> Ryu Furusawa,2023</p>
</div> -->

## How to Install インストール方法
Before installing this library, please install the following external libraries

このライブラリをインストールする前に、以下の外部ライブラリをインストールしてください

```bash
pip install opencv-python numpy psutil easing-functions matplotlib librosa
```
このライブラリをインストールする
```bash
pip install git+https://github.com/ryufurusawa/imgtrans.git
```

## Usage Flow　使い方の流れ
### 1. library import ライブラリのインポート
First, import the module.   
まず、モジュールをインポートします。
```python
import imgtrans
```
### 2. Initialization of drawManeuver class
入力のビデオパスと、計算の基準となるスリットの方向を指定してdrawManeuverクラスのインスタンスを作成します。
```python
#入力映像のパス
videopath= '/Users/Movies/20230917_RFS3108_mod-HD720p.mov'

#drawManeuverクラスのインスタンスを作成する。
#２つ目の変数はスリットが縦か横かを指定する。0=横スリット、1=縦スリットを示す。
your_maneuver=imgtrans.drawManeuver(videopath,1)
```
#### Slit Direction　スリットの方向 
本ツールでは、スリットの方向を縦スリットと横スリットに限定し、最初に指定します。  
スリットの方向の違いによって、今後の軌道の操作による結果は大きく変わります。
![slit-direction illustration](images/slit-direction.png)
![Alt text](images/slit-direction-transpostion-3dplotanimation.gif)
![Alt text](images/slit-direction-transpostion-rendering.gif)

### 3. Maneuver Design 軌道のデザイン
いくつかのクラスメソッドを組み合わせて再生断面の軌道をデザインします。  
クラスメソッドは**1. 時空間統合的な動きを加える関数**と**2. 時間の流れを適応させる関数**の二つに大きく分けられます。  
これらを実行することで、インスタンス変数の`data`に軌道データが格納されます。 
各関数は、内部でインスタンス変数の`data`に新たな配列を加えたり、全体のデータへなんらかの数値を掛け合わせたり、といった処理を行っています。
ここで編集される軌道データの内実は、出力映像が入力映像のどこのスリット（空間位置、時間位置）と対応しているか、座標変換として記述されています。  
詳しくは、[`data`の構造](#dataの構造)をご確認ください。

#### 1. Main Functions for Spatiotemporal Integration 主な時空間統合的な動きを加える関数
- [`addTrans`](#addtransselfframe_numsend_line1start_line0speed_roundtrue): 空間次元と時間次元のシンプルな置き換え。
- [`addBlowupTrans`](#addblowuptransselfframe_numsdegspeed_roundtrueconnect_round1): addTransを継承しつつ時間次元のスケールの拡大縮小の操作
- [`addInterpolation`](#addinterpolationselfframe_numsinterporation_directionz_directionaxis_positionreversal0cycle_degree90extra_degree0zslide0speed_roundtruerrange01): 時空間次元の遷移
- [`addCycleTrans`](#addcycletransselfframe_numscycle_degree360zscalingfalsezslide0extra_degree0speed_roundtrue): 画面の中心線を軸に、再生断面を回転させていく。
- [`addWaveTrans`](#addwavetransselfframe_numscycle_degreezdepthflow1zslide0speed_roundtrue): 動的な波の形状による再生断面を作成。
- [`addEventHorizonTrans`](#addeventhorizontransselfframe_numszdepthz_osc1cycle_degree180flowfalsezslide0): 画面の中心と周辺で時間の進行速度が変わる。

![Alt text](images/Maneuver-examples-3dplot.gif)

#### 2. Main Functions to Adapt the Flow of Time 主な時間の流れを適応させる関数
- [`applyTimeForward`](#applytimeforwardselfslide_timenone): 配列全体に時間の順方向の流れを付与
- [`applyTimeOblique`](#applytimeobliqueselfmaxgap): 時間のずれをスリットのごとに一定数づつ時間をずらす
- [`applyTimeForwordAutoSlow`](#applytimeforwordautoslowselfslide_timeint1defaultaddtimeint100addtimeeasingbooltrueeaseratiofloat03): 再生レート１からスロー再生になり最後に再生レート１に戻る
- [`applyTimeLoop`](#applytimeloopselfslide_timefreq2stay_time0): シームレスなループ構造を付与。
- [`applyTimeClip`](#applytimeclipselftrackslitintcliptimenone): 指定したスリットの時間の流れを指定した時間に固定する。
- [`applyTimeBlur`](#applytimeblurselfbl_time): 時間的なぼかしを適用

![Alt text](images/timeManeuver-examples-3dplot.gif)

#### Examples of Combinations 実際の組み合わせの例
```python
# 軌道デザイン
bm.rootingA_interporation(270)
bm.applyTimeLoop(1)
```
![Alt text](images/mixManeuver-examples-3dplot_harf.gif)

####  Saving and loading maneuver data 軌道データの保存と読み込み
別のソフトウェアで編集したり、レンダリング自体は後に行う場合や、同じ軌道データを用いて複数の映像データをレンダリングする場合などに、軌道データのみを保存、読み込みする場合があります。

##### Maneuver data storage 軌道データの保存
書き出し用ディレクトリ内に保存されます。
```python
your_maneuver.data_save()
```
##### Loading maneuver data 軌道データの読み込み
初期化する場合。
```python
import numpy as np
your_maneuver=imgtrans.drawManeuver(videopath="path/to/video.mp4", sd=1,datapath="path/to/data.npy" )
#軌道データを確認する
print(your_maneuver.data.shape)
```
`data`だけ置き換えする場合。
```python
import numpy as np
your_maneuver.data=np.read("path/to/data.npy" )
```
いずれも、読み込ませる映像データのサイズやフレーム数で示せる座標の範囲内に治っている必要があります。例えば、入力映像の解像度がフルハイビジョン(1920x1080)
であるのに、参照する縦スリットの横位置が2000であった場合、エラーとなります。


### 4. Visualization ヴィジュアライズ
この機能は、インスタンス変数の`data`を視覚的に表現し、理解しやすくするためのものです。  
2Dプロットと3Dプロットの２つの方法でデータを可視化できます。  
3Dグラフでは、軌道の全体的な動きを直感的に把握することができます。一方、時間の流れの詳細は2Dグラフを見ることでより明確に理解することができます。  
この2つの方法を組み合わせることで、たとえば画面の左側では時間が逆行し、右側では時間が順行するといった、時間の動きの細かな設計も行うことができます。  
ヴィジュアライズのイメージデータは、入力映像と同じパスに生成された書き出し用ディレクトリ内に保存されます。

#### color of slits スリットの色
映像の空間方向が反転する振る舞いも設計可能です。そのような空間次元の方向を明示する意味で、ヴィジュアライズのスリットの描画は緑-赤のグラデーションにより区別されています。  
1. 縦スリットの場合は、緑が左端、赤が右端の出力位置に対応します。
1. 横スリットの場合は、緑が上端、赤が下端の出力位置に対応します。

#### 2d plot
2次元の軌道グラフは、軌道デザインに関する操作が行われるたびに逐次書き出されます。  
このグラフは、出力映像の時間を横軸として、以下の3つの要素を一つの図として表示します。
1. 空間方向動き
2. 時間方向の動き
3. 時間方向の動きの再生レート

```python
your_maneuver.maneuver_2dplot()
```
デフォルト設定では、スリットは20本で生成されます。maneuver_2dplotメソッドの第一引数thread_numを変更することで、表示するスリットの本数を調整できます。
```python
# 50本のスリットを描画
your_maneuver.maneuver_2dplot(50)
```

軌道デザインのコードとその軌道データの2Dプロット。  
```python
#1 時空間統合的な動きのデザインのモジュールを連結していく。
#ノーマルな状態を100フレーム分追加する
your_maneuver.addFlat(100)
#ノーマルな状態から映像フレームの左端を軸に90度回転させます。
your_maneuver.addInterpolation(100,0,1)
#100フレーム分、時間と空間を交換した軌道を加えます。
your_maneuver.addTrans(100)

#2 時間の振る舞いに関するモジュールを組み合わせる。
#軌道全体を時間方向への１フレームづつ送ります。
your_maneuver.applyTimeForward(1)
#軌道全体を時間方向の動きに対してブラーを加え滑らかに変化させる。
your_maneuver.applyTimeblur(50)

#2Dプロット出力
your_maneuver.maneuver_2dplot()
```

![visualized 2dplot image](images/GX010161_2023_0616_Vslit_Flat100+Interpolation300(ID1-ZD0-AP0-REV0)+Freeze30+Transposition300+CycleTrans_addExtend_TimeForward1_TimeBlur30_TimeBlur100_SpaceBlur100_20thread.png)

逐次書き出しが不要の場合は、クラス変数の設定を変更してください。　　
```python
your_maneuver.auto_visualize_out = False
```
#### 3D plot
三次元グラフへの軌道プロットアニメーションを出力する場合は、明示的に書く必要があります。
```python
your_maneuver.maneuver_3dplot()
```

![visualized 3dplot gifimage](images/GX010148_2023_0617_Vslit_Flat100+Interpolation300(ID1-ZD0-AP0-REV0)+Freeze30+Transposition300+CycleTrans_addExtend_TimeForward1_TimeBlur30_TimeBlur100_SpaceBlur100_3dPlot.gif)

### 5. Rendering レンダリング
入力の映像データを、インスタンス変数に`data`をもとに、時空間を組み直し、映像のレンダリングを行います。

#### Structure of `data`　データの構造
インスタンス変数`data`には軌道データが格納されています。その`data`の構造について解説します。  
本モジュールでは、スリットの方向を最初に横か、縦かを定義し、映像データへのアクセスをピクセル単位ではなくスリット単位としています。  
こうすることで、各映像データの最小単位であるスリットへのアクセスは二次元の座標（一次元位置（縦スリットであれば横）、時間）を指定することでアクセス可能となります。
`data`に保存される軌道データとは出力映像を構成する各スリットが入力映像を構成するスリットのどの座標（一次元位置、時間位置）から持ってこられたものかを示す、座標変換のマップです。
そのため、各ピクセルの色彩のデータは保存されていません。あくまで座標変換の対応が記述されているだけです。
データは、出力映像のフレーム数と、出力映像を構成するスリット数、この2次元の各データに2つのチャンネルを持たせた三次元のNUMPY配列として保存されています。

1. 参照される入力映像のスリット位置
2. 入力映像の時間位置

以下のコードは、`data`を調べるいくつかのサンプルです。

```python
print("出力する映像のフレーム数",your_maneuver.data.shape[0])
print("スキャンする数、縦スリットの場合、出力する映像の横幅のピクセル数",your_maneuver.data.shape[1])
print("出力映像の最初のフレームの右端のスリットが、入力映像のどの時間から参照されたか？",your_maneuver.data[0,-1,1])
print("入力映像から参照する時間位置の最大値",np.max(your_maneuver.data[:,:,1]))

#一つ目のスリットの出力位置の推移を描画する。
plt.plot(your_maneuver.data[:,0,0])
#一つ目のスリットの入力の時間位置の推移を描画する。
plt.plot(your_maneuver.data[:,0,2])
```
#### Video Rendering 映像のレンダリング　
レンダリング映像データは、入力映像と同じパスに生成された書き出し用のディレクトリ内に保存されます。
```python
your_maneuver.transprocess()
```
高解像度かつ、長めの映像の書き出しを行う場合には、分割して書き出すことで対応できます。端末のスペックに応じて設定して下さい。
中間ファイルはtmpディレクトリを一時的に作成して、そこにnumpyの配列データ（二次元イメージデータ）として保存します。
```python
your_maneuver.transprocess(10)#10回に分けて書き出す。
```
途中からの書き出しなど、分割したレンダリングの手法もオプションとして設置できます。  
書き出し用のディレクトリ内にtmpディレクトリがあり、そこに中間データが保存されている必要があります。もし無いと最終的にデータ統合する段階でエラーとなります。
以下の例では10段階の５段階目からレンダリングを行っています。
```python
your_maneuver.transprocess(10,sep_start_num=5,sep_end_num=10)
```
out_type変数にて指定することで静止画像の連番として書き出すことも可能です。

```python
your_maneuver.transprocess(out_type=0) #0=still, 1=video, 2=both 
```
詳細は [`transprocess`](#transprocessメソッド)こちらを参照ください。

#### Audio Rendering 音声のレンダリング
音声処理自体は、SuperColliderで行います。  
まず、SuperColliderで読み込ませるコードをクラスメソッドの`scd_out`から出力します。  
`scd_out`では、インスタンス変数の`data`で記述されたスリットの動きのデータから、音声出力するために、スリットの本数を間引いた上で出力します。   
音声ファイル名はインスタンス変数`sc_FNAME`にて指定できます。デフォルトでは、[入力映像のファイル名.aiff] としています。入力映像と同じディレクトリに保存されていることを確認して下さい。  

'scd_out'の第一引数にて、同時発話数を指定可能です。デフォルトでは、7つとなっています。  
あまり数を増やしすぎると、わずかな時間差により周波数の打ち消しが発生し、音量が極端に下がったりします。軌道の編集内容や、素材となる音の音響的な特徴を元に適切な数を指定してください。

```python
bm.sc_FNAME="GX010230-t-AIFF.aiff"
bm.scd_out(7)
```
上記を実行すると、４つのCSVdataと、４種のSuperColliderのプログラム.scdファイルが出力されます。

##### 4 types of CSV 4つの軌道データのCSV
1. *_7threads.csv : スリットの時間位置。
1. *_Rate_7threads.csv : スリットの再生レート
1. *_inPanMap_7threads.csv : スリットの空間位置
1. *_nowDepth_7threads.csv : 一枚のフレーム内における時間のずれ幅

##### 4 types of SCD 4つのscdファイル
1. *_SC_Play-7voices.scd : マルチ再生、再生レートに準じてピッチが変化する
1. *_SC_Grain-7voices.scd : グラニュラーシンセシスを用いたマルチ再生。再生レートに関係なくピッチは変化しない。
1. *_SC_Rev_Play-7voices.scd : マルチ再生。時間のずれ幅に応じてリバーブの適応。
1. *_SC_Rev_Grain-7voices.scd : グラニュラーシンセシスを用いたマルチ再生に加え、時間のずれ幅に応じてリバーブを加える。

サンプルの映像ファイルを確認して、その効果と特徴を参考にしてください。

##### Running scd files scdファイルの実行
scdファイルのうち、いずれかをSuperColliderに読み込ませます。　　
いずれも、リアルタイムに音響処理を実行し、それを仮想サーバーにてレコーディングし音声ファイルとして保存させます。
保存される音声ファイルは映像レンダリングデータと同じディレクトリに保存されます。  
音声データやCSVデータの読み込みに時間がかかるため、一括の実行を避け、2つの工程に分けています。  
`()`で括られている内容を、順に実行しください。

![Alt text](images/scd_sample.png)
 
###### 1. Audio Setting and Data Loading AudioのSettingとデータの読み込み
AudioのSetting、`SynthDef`によるシンセの定義、音声データの読み込み、CSVdataの読み込みを行います。  
オーディオのアウトプットデバイスの設定は各自の環境に合せ書き換えてください。  
デフォルトでは以下のようになっています。

```supercollier
Server.default.options.outDevice_("MacBook Proのス");
```
以下を実行することで、指定可能なデバイスリストがコンソールwindowに出力されます。
```supercollier
ServerOptions.devices; 
```
###### 2. Playback and Recording of `Synth` 再生とレコーディング
定義した `Synth`の再生をループ処理によりリアルタイムに再生を行い、Recordingを行います。  
Recordingと同時にUnix commandによりレンダリング映像ファイルをQuickTimePlayerにて再生させます。
やや時間のギャップは入りますが、映像と音声を擬似的に同期した状態で再生できます。
QuickTimePlayerのあるmacのみ実行可能ですので、それ以外の環境においてはこの部分をコメントアウトして対応してください。
```unixcmd
"open -a 'QuickTime Player' '/Users/Movies/sample-raw-mov/sample_Vslit.mp4' ".unixCmd;
```
#### Audio and video coupling 音声と映像の結合
特にプログラムを用意していません。
映像編集ソフトにて、ビデオと音声を時間同期させた上で、再書き出しを行ってください。

## drawManeuver class

このクラスはImgtransライブラリのメインとなるものです。

### Class Variables クラス変数:
- `imgtype`: レンダリングにおける静止画像のフォーマット（デフォルトは ".jpg"）
- `img_size_type`: 出力イメージのサイズの設定。入力の映像の高さをh,幅をwとすると、`0`:h,w `1`:w,w*2 `2`:総フレーム数分 `3`: square （デフォルトは `0`）
- `outfps`: 出力のフレームレート（デフォルトは 30）
- `auto_visualize_out`: 自動可視化の設定（デフォルトは True）
- `default_debugmode`: デフォルトのデバッグモード設定（デフォルトは False）
- `audio_form_out`: オーディオ形式出力の設定（デフォルトはFalse）
- `embedHistory_intoName`: 名前への履歴埋め込みの設定（デフォルトは True）
<!-- - `progressbarsize`: プログレスバーサイズ（デフォルトは 50） -->
<!-- - `sepVideoOut`: セパレート出力設定 -->

### List of all class methods 全クラスメソッドのリスト:
- [`__init__`](#__init__メソッド): ビデオパスを受け取り初期化する。
- [`append`](#appendメソッド): 別で作成していた軌道データをインスタンス変数としてもつ軌道データの後ろに追加する。
- [`prepend`](#prependメソッド): 別で作成していた軌道データを、インスタンス変数としてもつ軌道データの先頭にmaneuverを追加する。
- 軌道デザインに関わるクラスメソッド
    - 時空間統合的な動きを加える関数
        - [`addFlat`](#addFlatメソッド): フラットな配列を追加。
        - [`addFreeze`](#addFreezeメソッド): 時間軸、空間軸、ともに最終列の配列を”frame_nums”で指定されたフレーム数分生成して加える。
        - [`addSlicePlane`](#addSlicePlaneメソッド): 指定した空間位置で、時間軸方向に沿って切り出した断面フレームを指定したフレーム数追加。
        - Transposition 空間次元と時間次元の置き換え
            - [`addTrans`](#addTransメソッド): 空間次元と時間次元のシンプルな置き換え。縦スリットの場合はX軸とT軸が置換。横スリットの場合はY軸とT軸が置換。
            - [`addKeepSpeedTrans`](#addKeepSpeedTransメソッド): 既存のフレームデータの速度を維持した状態で、新しいフレームを生成し追加する。特定の空間領域に達するまで繰り返されます。
            - [`insertKeepSpeedTrans`](#insertKeepSpeedTransメソッド): `addKeepSpeedTrans`の発展版で、`self.data`に対して、`after_array`で受け取った配列の間を滑らかに補う。
            - [`addWideKeyframeTrans`](#addWideKeyframeTransメソッド): `addKeyframeTrans`の発展版。`midtide`のように、インプット画像よりもサイズを大きくして出力させる場合に使用する。
            - [`addBlowupTrans`](#addBlowupTransメソッド): blowupの動きをキーフレームにより詳細に制御するメソッド。このメソッドは、時間軸の解像度を徐々に変化させる試みがなされており、基本的にはXYT Transと同様の動きを持っています。時間軸のキーフレームに関する詳細な制御を行うためのキーパラメータは、`timevalues`と`timepoints`です。
       - 時空間次元の遷移
            - [`addInterpolation`](#addInterpolationメソッド): 与えられたパラメータをもとに補間を行い、結果をデータに追加する。
            - [`rootingA_interporation`](#rootingA_interporationメソッド): 複数のaddInterpolationを組み合わせる。ジグザグとした動き。
            - [`rootingB_interporation`](#rootingB_interporationメソッド): 複数のaddInterpolationを組み合わせる。ドミノが坂道を転がっていくような動き。
            - [`addCycleTrans`](#addCycleTransメソッド): XYTの置換を補完的に遷移させる。画面の中心線を軸に、再生断面を回転させていく。
            - [`addCustomCycleTrans`](#addCustomCycleTransメソッド): addCycleTransの回転の中心軸を動かすことができる。
        - 波打つ再生断面
            - [`addWaveTrans`](#addWaveTransメソッド): 時間と空間のピクセルのマトリクスに対して、動的な波の形状による再生断面を作成。空間軸を固定するかの切り替えも可能。
            - [`addEventHorizonTrans`](#addEventHorizonTransメソッド): 空間領域は変更なし。画面の中心と周辺で時間の進行速度が変わる。前進、後退するカメラによりキャプチャされる映像のオプティカルフローをキャンセルする。

    - 時間に特化した軌道操作
        - [`applyTimeForward`](#applyTimeForwardメソッド): 配列全体に時間の順方向の流れ（単位はslide_time）を付与
        - [`applyTimeOblique`](#applyTimeObliqueメソッド): 時間の斜め効果を適用
        - [`applyTimeForwordAutoSlow`](#applyTimeForwordAutoSlowメソッド): 基本、現在がスロー再生状態の場合に使用する。イントロ、アウトロに通常の再生速度の映像を加え、その間をイーズ処理することで滑らかに接続させる。
        - [`applyTimeFlowKeepingExtend`](#applyTimeFlowKeepingExtendメソッド): 与えた軌道配列に、延長させたフレームをプリペンド、アペンドする。XYフレームそれぞれ最終フレームと最初のフレームと同じデータで延長させる。Z(アウト時間）に関しては最終の変化量を維持して延長させる。`fade`引数をTrueでスピード０に落ち着かせる。
        - [`applyTimeLoop`](#applyTimeLoopメソッド): 与えた軌道配列全体の時間を前半、順方向、後半、逆転して、最後にまた順方向へながれ、最初と終わりの時間差がない。そのままループ再生すればシームレスなループが作られる。デフォルト周波数２hzでしか現在対応できていない。
        - [`applyTimeClip`](#applyTimeClipメソッド): 指定したスリットの時間の流れを指定した時間に固定する。
        - [`applyTimeSlide`](#applyTimeSlideメソッド): 一番初めのフレームの中心のスリットの参照時間を、指定した時間にセットする。それに合わせて全体に対してスライドさせて調節する。
        - [`applyInOutGapFix`](#applyInOutGapFixメソッド): シームレスループ作成のための補助的な関数。最初と最終フレームの差分を計算し、必要に応じてフレームの調整を行う。
        - [`applySpaceBlur`](#applySpaceBlurメソッド): 空間的なぼかしを適用
        - [`applyTimeBlur`](#applyTimeBlurメソッド): 時間的なぼかしを適用
        - [`applyCustomeBlur`](#applyCustomeBlurメソッド): カスタム範囲のブラーを適用
    - その他の軌道操作
        - [`addFreeze`](#addFreezeメソッド): 時間軸、空間軸、ともに最終列の配列を”frame_nums”で指定されたフレーム数分生成して加える。
        - [`preExtend`](#preExtendメソッド): 与えた軌道配列の１フレーム目を手前に延長させる。
        - [`addExtend`](#addExtendメソッド): 与えた軌道配列の最終フレームを延長させる。Zのレートは0になる。
        - [`zCenterArrange`](#zCenterArrangeメソッド): 軌道配列と入力映像のフレーム数を照らし合わせ、入力映像の時間的な意味での中心

- マニューバーの情報出力に関するメソッド
    - [`dataCheck`](#dataCheckメソッド):`data`の情報をコンソールに出力する。
    - [`info_setting`](#info_settingメソッド): データをスレッド数に応じて設定し、再生レートやパンを計算します。
    - [`maneuver_CSV_out`](#maneuver_CSV_outメソッド): CSVファイルに軌道配列データを出力する。軌道の可視化を外部のソフトウェア（エクセルなど）にて作成する場合に使用する。
    - [`scd_out`](#scd_outメソッド): supercolliderに読み込ませるサウンドプロセスコードを出力し、関連するデータをCSVで保存します。
    - [`data_save`](#data_saveメソッド): 軌道データをnumpyのファイルとして保存します。
    - [`split_3_npysavereturn`](#split_3_npysavereturnメソッド): 軌道データのアウトイメージの横幅が入力の３倍サイズを指定した場合に使用。Left, center, Rightとして3分割して保存。3分割されたNPYファイルのパスを配列で返す。
- マニューバーの可視化ファイルの出力に関するメソッド
    - [`maneuver_2dplot`](#maneuver_2dplotメソッド): 2Dプロットを作成して、それに関連するデータを画像として保存します。
    - [`maneuver_3dplot`](#maneuver_3dplotメソッド): 3Dプロットを生成し、その画像や動画を保存するメソッドです。
- 映像renderingに関するメソッド
    - [`transprocess`](#transprocessメソッド): 映像をトランスプロセスします。このメソッドは映像のレンダリングを行い、セパレートレンダリングや他のオプションをサポートしています。
    - [`transprocess_typeB`](#transprocess_typeBメソッド): 映像のレンダリングを行うメソッド。セパレートプロセスをアウトプットの時間次元ではなく、インプットの時間次元を分割させて処理します。時間軸方向に極端に幅を広くとるようなマニューバーが組み込まれている場合はこのメソッドを使うことで、レンダリング速度が上がります。
    - [`animationout`](#animationoutメソッド): 出力した映像データを参照して、3Dグラフ上に画像のピクセルカラーをプロットし、結果をアニメーションとして出力する。そのため、映像のレンダリングを行った後にしか実行できません。

## `__init__`
`__init__`メソッドは、ビデオのパス、スキャン方向、データ、およびフォルダ名の属性を引数として受け取る。このメソッドは、下記のインスタンス変数を初期化し、ビデオパスと同じレベルに出力用のディレクトリを作成し、そのディレクトリに移動する。最終的に出力されるあらゆるファイルはこのディレクトリ内に保存されます。

### argument 引数
- `videopath` (str): ビデオファイルへのパス。
- `sd` (bool): スリットの方向。`True`で縦スリット`False`で横スリット
- `datapath` (str, optional): 以前に保存していた軌道データを引き継ぐ場合に使用するオプション。Numpyの多次元配列として保存されたnpyデータのパス。
- `foldername_attr` (str, optional): オプションとして出力用のディレクトリの名称に、指定した名称を付け加えます。

### インスタンス変数
1. **data**: 最小単位をスリットとする再生断面の軌道データ。デフォルトは空のリスト。
1. **width**: ビデオの幅。`videopath` より読み込んだビデオ情報を反映する。
1. **height**: ビデオの高さ。
7. **count**: ビデオの総フレーム数。
8. **recfps**: ビデオのfps（フレームレート）。出力のフレームレートは[クラス変数](#クラス変数)にて設定する。
10. **scan_direction**: スリットの向きとスキャン方向の定義。初期化メソッドの引数`sd`がそのまま適応される。
11. **scan_nums**: スキャンする数。4k解像度で縦スリットの場合は3840
12. **slit_length**: 1スリットの画素数。4k解像度で縦スリットの場合は2160
15. **out_name_attr**:初期化メソッドの引数`foldername_attr` がそのまま適応される。
1. **out_videopath**: 出力したビデオのパスを保持する。初期値は空。[animationout](#animationoutメソッド)にて呼び出す。
18. **sc_FNAME**: 入力のヴィデオのファイル名に ".AIFF" を追加したものを自動で受け取るように初期設定されています。音声処理をするためのsuper collider用のコードを出力する際に使用します。
13. **sc_resetPositionMap**, **sc_rateMap**, **sc_inPanMap**, **sc_now_depth**: 軌道配列を音声処理用にスリットの分割数を落とし最適化した配列
<!-- 1. **cap**: OpenCVのVideoCaptureオブジェクト。入力ビデオの情報を持つ。
16. **log**: ログのカウンタ。デフォルトは0。マニューバーの編集工程のログを取っています。
17. **infolog**: インフォログのカウンタ。デフォルトは0。
1. **ORG_NAME**: 入力ビデオパスから取得したオリジナルの名前。
2. **ORG_PATH**: 入力ビデオパスから取得したオリジナルのパス。
3. **ORG_FNAME**: 入力ビデオパスから取得したオリジナルのファイル名。
14. **sd_attr**: スキャン方向が0の場合は"Hslit"、それ以外の場合は"Vslit"。出力ファイルの名称に適応する。 -->

### example
```python
your_maneuver=imgtrans.drawManeuver(videopath="path/to/video.mp4", sd=1)
```
以前に保存していた軌道データを引き継ぐ場合は以下の例を参照ください。
```python
import numpy as np
your_maneuver=imgtrans.drawManeuver(videopath="path/to/video.mp4", sd=1,datapath="path/to/data.npy" )
#軌道データを確認する
print(your_maneuver.data.shape)
```
<!-- 
### append(self,maneuver,auto_zslide=True,zslide=0)
### prepend(self,maneuver)
### addFlat(self,frame_nums)
### addFreeze(self,frame_nums)
### addSlicePlane(self,frame_nums,xypoint=0.5)
-->
## `addTrans`メソッド

`addTrans`メソッドは、`wr_array`に新しいトランス軌跡を追加して返すための関数です。このメソッドは、特定のフレーム数にわたって、サイクル的な角度変化を考慮して変換を行います。

### argument 引数
- `frame_nums`(int): 追加するフレーム数。
- `end_line`(float, optional, default: `1`): 変換の終了ライン。
- `start_line`(float, optional, default: `0`): 変換の開始ライン。
- `speed_round`(bool, optional, default: `True`): 速度が円滑かどうかを指定。
- `zd`(bool, optional, default: `True`): 方向設定。

### Examples of Use 使用例
```python
your_object.addTrans(100, end_line=1, start_line=0, speed_round=True, zd=True)
```
![Alt text](images/sample_2023_0618_Vslit+Transposition100_3dPlot.gif)
![Alt text](images/sample_2023_0618_Hslit+Transposition100_3dPlot.gif)
<!-- 
### addKeepSpeedTrans(self,frame_nums,under_xyp,over_xyp=1,rendertype=0)
### insertKeepSpeedTrans(self,frame_nums,under_xyp=None,over_xyp=1,after_array=[],rendertype=0)
### addWideKeyframeTrans(self,frame_nums,key_array,wide_scale=3,start_frame=[0,0],speed_round=False)
### addBlowupTrans(self,frame_nums,deg,speed_round=True,connect_round=1)
-->
## `addInterpolation`

`interpolation` メソッドは、指定された軌道データをもとにインターポレーションを行い、新たなフレームを生成するためのメソッドです。この関数は、特定のフレーム数にわたり、複雑な変換を加えるためのものです。

### argument 引数
- `frame_nums`(int): インターポレーションを行うフレーム数。
- `interporation_direction`(int): インターポレーションの方向。
- `z_direction`(int): Z方向におけるインターポレーションの方向。
- `axis_position`(int): 回転やインターポレーションの中心となる位置。
- `reversal`(int, optional, default: `0`): 動きの反転を制御する。
- `cycle_degree`(int, optional, default: `90`): 1サイクルあたりの角度。
- `extra_degree`(int, optional, default: `0`): 変換を始める最初の段階での断面の角度の指定。
- `zslide`(int, optional, default: `0`): Z方向のスライド量。
- `speed_round`(bool, optional, default: `True`): 速度が円滑かどうかを指定。
- `rrange`(list of int, optional, default: `[0,1]`): 変換の範囲を指定するリスト。

### Examples of Use 使用例
```python
your_object.addInterpolation(100, 0, 0, 0,reversal=False)
```
![Alt text](images/sample_2023_0618_Vslit+Interpolation100(ID0-ZD0-AP0-REV0)_3dPlot.gif)
```python
your_object.addInterpolation(100, 0, 1, 1,reversal=True)
```
![Alt text](images/sample_2023_0618_Vslit+Interpolation100(ID0-ZD1-AP1-REV1)_3dPlot.gif)


<!--
### rootingA_interporation(self,FRAME_NUMS,loop_num=2,axis_first_p=0)
### rootingB_interporation(self,FRAME_NUMS,loop_num=1,axis_fix_p=0)
-->

## `addCycleTrans`

`addCycleTrans` メソッドは、サイクル的な変換（トランス）をデータに追加するためのものです。特定のフレーム数にわたって、サイクル的な角度変化を考慮して変換を行います。

### argument 引数
- `frame_nums`(int): 追加するフレーム数。
- `cycle_degree`(int, optional, default: `360`): 1サイクルあたりの角度。
- `zscaling`(bool, optional, default: `False`): Z軸のスケーリングを有効にするかどうか。
- `zslide`(int, optional, default: `0`): 時間位置の初期値。
- `extra_degree`(int, optional, default: `0`): 変換を始める最初の段階での断面の角度の指定。
- `speed_round`(bool, optional, default: `True`): 速度が円滑かどうかを指定。

### Examples of Use 使用例
```python            
your_object.addCycleTrans(100, cycle_degree=360, zscaling=True, zslide=10, extra_degree=5, speed_round=False)
```
![Alt text](images/sample_2023_0618_Vslit+CycleTrans360-zscale0_3dPlot.gif)

## `addCustomCycleTrans`

`addCustomCycleTrans` このクラス関数は、`addCycleTrans`での中心軸を自由に動かすことができます。
`start_center`引数で指定した位置から`end_center`で指定した位置までリニアに変化させます。

### argument 引数
- `frame_nums`(int): 追加するフレーム数。
- `cycle_degree`(int): 1サイクルあたりの角度。
- `start_center`(float, optional, default: `1/2`): 開始の中心軸の位置。(0~1)の範囲でスキャン方向の長さに対しての比率として指定する。縦スリットの場合、0は左端、1は右端になる。
- `end_center`(float, optional, default: `1/2`): 終了の中心軸の位置。
- `zscaling`(bool, optional, default: `False`): Z軸のスケーリングを有効にするかどうか。
- `extra_degree`(int, optional, default: `0`): 変換を始める最初の段階での断面の角度の指定。
- `speed_round`(bool, optional, default: `True`): 速度が円滑かどうかを指定。
- `zslide`(int, optional, default: `0`): 時間位置の初期値。
- `auto_zslide`(bool, optional, default: `True`): Zスライドの自動調整の有効・無効を切り替えるフラグ。
- `zscaling_v`(float, optional, default: `0.9`): Zスケーリングの係数。

### Examples of Use 使用例
```python
your_object.addCustomCycleTrans(100, cycle_degree=360, start_center=0.2, end_center=0.7)
```
![Alt text](images/sample_2023_0618_Vslit+CustomCycleTrans360-zscale0_3dPlot.gif)

<!-- ### addWaveTrans(self,frame_nums,cycle_degree,zdepth,flow=1,zslide=0,speed_round=True)
### addEventHorizonTrans(self,frame_nums,zdepth,z_osc=1,cycle_degree=180,flow=False,zslide=0)
### applyTimeForward(self,slide_time=None)
### applyTimeOblique(self,maxgap)
### applyTimeForwordAutoSlow(self,slide_time:int=1,defaultAddTime:int=100,addTimeEasing:bool=True,easeRatio:float=0.3)
### applyTimeFlowKeepingExtend(self,frame_nums:int,fade:bool=False)
### applyTimeLoop(self,slide_time,freq=2,stay_time=0)
### applyTimeClip(self,trackslit:int,cliptime=None)
### applyTimeslide(self,startframe:int)
### applyInOutGapFix(self)
### applySpaceBlur(self,bl_time)
### applyTimeBlur(self,bl_time)
### applyCustomeBlur(self,s_range,e_range,bl_time)
### addFreeze(self,frame_nums)
### preExtend(self,addframe:int)
### addExtend(self,frame_nums:int)
### zCenterArrange(self)
### zPointCheck(self)
### info_setting(self,thread_num=20)
### maneuver_CSV_out(self,thread_num=None)
### scd_out(self,thread_num=None,audio_path=None)
### data_save(self)
### split_3_npysavereturn(self)
### maneuver_2dplot(self,thread_num=None,debugmode=False,w_inc=5,h_inc=9,plinewidth=1.0,individual_output=False,palpha=1.0)
### maneuver_3dplot(self,thread_num=None,zRangeFix=False,outFrame_nums=50,out_fps=10)
-->

## `transprocess`メソッド
このメソッドは映像のレンダリングを行います。

### argument 引数
- `separate_num` (int, optional, default: `1`): レンダリングを何分割で行うかの指定。
- `sep_start_num` (int, optional, default: `0`): 分割レンダリング時に、どの分割から開始するかの指定。
- `sep_end_num` (int or None, optional, default: `None`): 分割レンダリング時に、どの分割まで行うかの指定。
- `out_type` (int, optional, default: `1`): 出力形式の指定。`1`は映像、`0`は静止画、`2`は映像と静止画両方。
- `XY_TransOut` (bool, optional, default: `False`): Trueの場合、出力映像を90度回転して保存。
- `render_mode` (int, doptional, default: `0`): レンダリングモード。`0`は全軌道データのフレームを統合して出力。`1`は`sep_start_num` から `sep_end_num` までの範囲だけを出力。

### Examples of Use 使用例
```python
# インスタンスを作成
your_maneuver = imgtrans.drawManeuver("mov/samplevideo.mp4",sd=1)

#軌道データの書き込み
your_maneuver.addTrans(100)
your_maneuver.applyTimeForward(1)

# レンダリング
your_maneuver.transprocess()
```

<!-- ## transprocess_typeB(self,separate_num=1,sep_start_num=0,sep_end_num=None,out_type=1,XY_TransOut=False,render_mode=0) -->
## `animationout`

The `animationout` function plots the pixel color of the image on a 3D graph with reference to the output video data and outputs the result as an animation. It allows you to visualize the trajectory of the playback cross section based on spatio-temporal manipulations on the spatio-temporal cube. Unlike 'maveuver_2dplot' and 'maveuver_3dplot', this visualization is more intuitive in its correspondence with the input video data by mapping pixel colors.  
 Therefore, it can only be executed after rendering the video. If there is already video data to be exported, it is necessary to call the information in the instance variable 'out_videopath'.
 
 `animationout`関数は、出力した映像データを参照して、3Dグラフ上に画像のピクセルカラーをプロットし、結果をアニメーションとして出力します。時空間キューブ上での時空間操作に基づく再生断面の軌道を可視化させます。 'maveuver_2dplot','maveuver_3dplot'とは違いピクセルの色をマッピングすることでより入力の映像データとの対応を直感的に理解しやすいビジュアライズです。  
 そのため、映像のレンダリングを行った後にしか実行できません。もし、既に書き出しの映像データが存在する場合は、インスタンス変数'out_videopath'にて情報を呼び出す必要があります。
 
 ```python
 your_maneuver.out_videopath = "mov/sample.mp4"
 your_maneucer.animationout()
 ```

 ### argument 引数
- `outFrame_nums` (`int`, default: `100`): 出力するフレーム数。
- `drawLineNum` (`int`, default: `250`): 描画するラインの数。
- `dpi` (`int`, default: `200`): 出力画像のDPI。
- `out_fps` (`int`, default: `10`): 出力動画のフレームレート。

### Examples of Use 使用例
![Alt text](images/20220106_RFS1459-4K_2023_0930_Vslit_interporationAset+IP2800(rootingA)_CustomeBlur300_CustomeBlur300_TimeLoop_timeSlide_zCenterArranged_img_3d-pixelMap.gif)

## Contribute
Contributions to this project are welcome.  
If you have any questions or feedback, please feel free to contact us at ryu.furusawa(a)gmail.com.  

このプロジェクトへのコントリビュートを歓迎します。  
質問やフィードバックがあれば、ryu.furusawa(a)gmail.comまでお気軽にお問い合わせください。

## License 
This project is licensed under the MIT License, see the LICENSE.txt file for details
