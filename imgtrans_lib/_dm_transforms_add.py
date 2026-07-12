"""drawManeuver の "追加" 系変換 (TransformsAddMixin)

新しい maneuver を生成して self.data に append/拡張する系統:
- addFlat / addSlicePlane
- _cut_source_extent / _cut_finalize / addCylinderCut / addBoxUnfoldCut
- addFreeze / preExtend / addExtend
- addInterpolation / interpolation
- rooting8(B) / rootingA(_RANDOM/_single/_trans_single) / rootingAA / rootingB / rooting4C / rooting4D
- addTrans / addKeepSpeedTrans / addInsertKeepSpeedTrans / addWideKeyframeTrans
- addWaveTrans / addEventHorizonTrans
- addCycleTrans / addCustomCycleTrans / addWideCustomCycleTrans / addFixWideCycleTrans
- addBlowupTrans
- cross_point (内部ヘルパ)
"""
import sys
import math
import random
import cv2
import numpy as np
import easing
from scipy.special import comb

from ._utils import (
    bezier_interpolation,
    closest_value,
    search,
)


class TransformsAddMixin:
    def addFlat(self,frame_nums,z_pos=0,z_autofit=True,prepend=False,flip=False):
        print(sys._getframe().f_code.co_name)
        extra_array = np.zeros((frame_nums,self.scan_nums,2),dtype=np.float64)
        normalFrame=np.arange(self.scan_nums-1,-1,-1) if flip else np.arange(0,self.scan_nums)
        if len(self.data) == 0 : #replace
            zFrame=np.full(self.scan_nums,z_pos)
            for i in range(frame_nums):
                extra_array[i,:,0]=normalFrame
                if z_pos!=0 :
                    extra_array[i,:,1]=zFrame
            self.data = extra_array
        else :
            if z_autofit:
                z_pos = int(np.mean(self.data[-1,:,1])) if prepend == False else int(np.mean(self.data[0,:,1]))
            zFrame=np.full(self.scan_nums,z_pos)
            for i in range(frame_nums):
                extra_array[i,:,0]=normalFrame
                extra_array[i,:,1]=zFrame
            self.data = np.vstack((self.data,extra_array)) if prepend == False else np.vstack((extra_array,self.data))
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1]+str(frame_nums))

    # 時間方向の断面配列を"frame_nums"で指定されたフレーム数分生成。"xypoint"は0−1の範囲で指定
    #　基本、このモジュールがはじめで終わりの場合のみ
    def addSlicePlane(self,frame_nums=1,xypoint=0.5,full_range=False,z_start=None,z_end=None):
        """
        スリットスキャン平面を生成する。
        xypoint: スリット位置（0.0〜1.0、scan_nums に対する比率）
        full_range: True で時間軸を全フレーム(0〜count)に拡張
        z_start, z_end: 時間範囲を直接指定（フレーム番号）。指定時は full_range より優先。
        """
        print(sys._getframe().f_code.co_name)
        # --- 時間範囲の決定 ---
        if z_start is not None or z_end is not None:
            # z_start/z_end 明示指定
            z_s = int(z_start) if z_start is not None else 0
            z_e = int(z_end) if z_end is not None else int(self.count)
        elif full_range:
            z_s, z_e = 0, int(self.count)
        else:
            z_s, z_e = 0, self.scan_nums
        time_length = z_e - z_s
        extra_array = np.zeros((frame_nums, time_length, 2), dtype=np.float64)
        normalFrame = np.arange(z_s, z_e, dtype=np.float64)
        xyFrame = np.full(time_length, int(self.scan_nums * xypoint), dtype=np.float64)
        for i in range(frame_nums):
            extra_array[i,:,0] = xyFrame
            extra_array[i,:,1] = normalFrame
        if len(self.data) == 0 : #replace
            self.data = extra_array
        else :
            print("error")
            return
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1]+str(frame_nums))

    # =====================================================================
    # XYT空間を3D幾何学曲面で切り出し、1フレームとして追加する関数群
    # =====================================================================
    # 出力幅(scan_nums)は入力映像サイズに依存せず、断面の実寸に基づいて
    # 自由に設計可能。output_width=None の場合、空間方向の全域をカバーし
    # xyt_boxel_scale を参照して自動計算される。
    # =====================================================================

    def _cut_source_extent(self):
        """カット関数用: ソース映像の空間方向のピクセル数を返す。"""
        return int(self.width) if self.scan_direction % 2 == 1 else int(self.height)

    def _cut_finalize(self, N, space, z):
        """カット関数用: 共通の後処理（data追加, scan_nums更新）。
        space はソースピクセル座標。クリップ後に data に追加する。
        """
        source_max = self._cut_source_extent()
        space = np.clip(space, 0, source_max - 1)
        new_frame = np.zeros((1, N, 2), dtype=np.float64)
        new_frame[0, :, 0] = space
        new_frame[0, :, 1] = z
        # scan_nums を出力幅に合わせて更新
        self.scan_nums = N
        if len(self.data) == 0:
            self.data = new_frame
        else:
            self.data = np.concatenate([self.data, new_frame], axis=0)

    def addCylinderCut(self, center_time, center_pos=0.5, time_scale=1.0,
                        phase=0.0, output_width=None):
        """円筒（シリンダー）でXYT空間を切り出し、1フレームを追加する。
        (空間, 時間)平面上で円（または楕円）を描くようにサンプリングする。
        空間=sin(θ), 時間=cos(θ) で、始点と終点が完全に一致するループ構造。

        空間方向はソース映像の全域（0〜source_extent-1）をカバーする。
        時間方向は空間方向のレンジに xyt_boxel_scale を掛けたサイズを基準とし、
        time_scale で倍率を調整する。

        旧stillrender()のcylinder(form=0)に相当する。

        Args:
            center_time (float): 円の中心の時間座標（フレーム番号）
            center_pos (float): 円の中心の空間位置（0-1、デフォルト0.5=中央）
            time_scale (float): 時間方向の倍率（デフォルト1.0）。
                1.0 = 空間レンジ × xyt_boxel_scale に比例した正円相当。
                0.5 = 時間幅が半分（横長楕円）。2.0 = 時間幅が2倍（縦長楕円）。
            phase (float): 開始角度のオフセット（ラジアン、デフォルト0.0）
            output_width (int): 出力の横幅（スリット数）。
                Noneの場合 int(π * source_extent) で自動計算
                （空間全域の円周に相当する密度）。
        """
        print(sys._getframe().f_code.co_name)
        source_ext = self._cut_source_extent()
        # 空間半径: ソース全域をカバー（中心からの片側）
        radius_space = source_ext / 2.0
        # 時間半径: 空間レンジに比例 × time_scale で調整
        radius_time = radius_space * max(self.xyt_boxel_scale, 1e-6) * time_scale
        # 出力幅: 空間全域の円周に相当するサンプル密度
        if output_width is None:
            output_width = max(int(np.pi * source_ext), 64)
        N = output_width
        center_s = center_pos * (source_ext - 1)
        theta = 2.0 * np.pi * np.arange(N, dtype=np.float64) / N + phase
        space = center_s + radius_space * np.sin(theta)
        z = center_time + radius_time * np.cos(theta)
        self._cut_finalize(N, space, z)
        print(f"  radius_space={radius_space:.0f}px, radius_time={radius_time:.0f}f, "
              f"output_width={N}, time_scale={time_scale}")
        self.maneuver_log("CylinderCut")

    def addBoxUnfoldCut(self, center_time, center_pos=0.5, time_scale=1.0, output_width=None):
        """XYT直方体の辺を展開した面で切り出し、1フレームを追加する。
        直方体の4辺を順にたどるループ構造:
          辺1: 通常フレーム(t=t_start)    x: 0 → max    （映像の1フレーム）
          辺2: スリットスキャン(x=max)     t: t_start → t_end （右端スリット）
          辺3: 通常フレーム(t=t_end)      x: max → 0    （映像の1フレーム、反転）
          辺4: スリットスキャン(x=0)       t: t_end → t_start （左端スリット、反転）
        始点と終点が完全に一致する閉じたループ。

        空間方向はソース映像の全域をカバーする。
        時間方向は空間レンジ × xyt_boxel_scale × time_scale で決定される。

        Args:
            center_time (float): 直方体中心の時間座標（フレーム番号）
            center_pos (float): 空間中心位置（0-1、デフォルト0.5=中央）
            time_scale (float): 時間方向の倍率（デフォルト1.0）。
                1.0 = 空間辺と時間辺が等価な正方形展開。
                0.5 = 時間辺が短い（横長の箱）。2.0 = 時間辺が長い（縦長の箱）。
            output_width (int): 出力の横幅（スリット数）。
                Noneの場合、展開図の全周長に相当するサンプル数で自動計算。
        """
        print(sys._getframe().f_code.co_name)
        source_ext = self._cut_source_extent()

        # 空間範囲: ソース全域
        x_min = 0.0
        x_max = float(source_ext - 1)
        center_s = center_pos * x_max

        # 時間範囲: 空間レンジに比例 × time_scale
        time_half = (source_ext / 2.0) * max(self.xyt_boxel_scale, 1e-6) * time_scale
        t_start = center_time - time_half
        t_end = center_time + time_half

        # 各辺の長さ（空間ピクセル等価）
        edge_spatial = source_ext                        # 辺1,3: 空間方向の長さ
        edge_temporal = source_ext * time_scale          # 辺2,4: 時間方向の長さ（空間換算）
        perimeter = 2.0 * edge_spatial + 2.0 * edge_temporal

        if output_width is None:
            output_width = max(int(perimeter), 64)
        N = output_width

        # 各辺の割合
        frac_s = edge_spatial / perimeter   # 空間辺1つ分の割合
        frac_t = edge_temporal / perimeter  # 時間辺1つ分の割合

        # 境界点
        b1 = frac_s                  # 辺1の終わり
        b2 = frac_s + frac_t         # 辺2の終わり
        b3 = 2.0 * frac_s + frac_t   # 辺3の終わり
        # b4 = 1.0                   # 辺4の終わり

        # パラメータ t ∈ [0, 1) で直方体の辺を1周
        t_param = np.arange(N, dtype=np.float64) / N

        # 辺1: 通常フレーム(t=t_start), x: 0→max
        # 辺2: スリットスキャン(x=max), t: t_start→t_end
        # 辺3: 通常フレーム(t=t_end), x: max→0
        # 辺4: スリットスキャン(x=0), t: t_end→t_start
        space = np.where(t_param < b1,
                    (t_param / frac_s) * x_max,
                np.where(t_param < b2,
                    x_max,
                np.where(t_param < b3,
                    (1.0 - (t_param - b2) / frac_s) * x_max,
                    0.0)))

        z = np.where(t_param < b1,
                t_start,
            np.where(t_param < b2,
                t_start + ((t_param - b1) / frac_t) * (t_end - t_start),
            np.where(t_param < b3,
                t_end,
                t_end - ((t_param - b3) / max(1.0 - b3, 1e-9)) * (t_end - t_start))))

        self._cut_finalize(N, space, z)
        print(f"  source_ext={source_ext}, t_range=[{t_start:.0f}, {t_end:.0f}], "
              f"output_width={N}, time_scale={time_scale}")
        print(f"  edge_spatial={edge_spatial:.0f}px, edge_temporal={edge_temporal:.0f}px, "
              f"perimeter={perimeter:.0f}px")
        self.maneuver_log("BoxUnfoldCut")

    # =====================================================================

    # 時間軸、空間軸、ともに変化のないフラットな配列を"frame_nums"で指定されたフレーム数分生成。
    def addFreeze(self,frame_nums):
        print(sys._getframe().f_code.co_name)
        extra_array = np.zeros((frame_nums,self.scan_nums,2),dtype=np.float64)
        xyFrame=self.data[-1,:,0]
        zFrame=self.data[-1,:,1]
        for i in range(frame_nums):
            extra_array[i,:,0]=xyFrame
            extra_array[i,:,1]=zFrame
        self.data = np.vstack((self.data,extra_array))
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1]+str(frame_nums))
    """
    スロー再生を全体に適応させイントロ、アウトロに通常の再生速度の映像を加え、その間をイーズ処理することで滑らかに接続させる。
    intro,outroのフレームを追加する。
    新規のマトリクス（extra_array）のバッファを生成するため、出来上がりの映像データのフレーム数を先に計算します
    インとアウトの通常の再生時間を追加します。最低でも5秒（150f）を確保。
    """
    def preExtend(self,addframe:int):
        print(sys._getframe().f_code.co_name)
        extra_array = np.zeros((self.data.shape[0]+addframe,self.data.shape[1],2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
        introFrame = self.data[0,:,:]
        extra_array[addframe:]=self.data
        for i in range(addframe):
            extra_array[i]=introFrame
        self.data = extra_array 
        self.maneuver_log(sys._getframe().f_code.co_name+str(addframe))

    # 与えた軌道配列の最終フレームを延長させる。Zのレートは0になる。
    def addExtend(self,addframe:int,flip=False):
        print(sys._getframe().f_code.co_name)
        extra_array = np.zeros((self.data.shape[0]+int(addframe),self.data.shape[1],2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
        outroFrame = self.data[-1,::-1,:] if flip else self.data[-1,:,:]
        extra_array[:self.data.shape[0]] = self.data
        for i in range(int(addframe)):
            extra_array[self.data.shape[0]+i] = outroFrame
        self.data = extra_array
        self.maneuver_log(sys._getframe().f_code.co_name+str(addframe))

    # #時間の変化率を維持したまま、始まりと終わり部分を延長させる。fade引数をTrueでスピード０に落ち着かせる。
    # def timeFlowKeepingExtend(self,frame_nums:int,fade:bool=False):
    #     print(sys._getframe().f_code.co_name)
    #     extra_array = np.zeros((self.data.shape[0]+frame_nums*2,self.data.shape[1],2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
    #     xyfirstFrame = self.data[0,:,0]
    #     xylastFrame = self.data[-1,:,0]
    #     zfirstDiff = self.data[1,:,1]-self.data[0,:,1]
    #     zlastDiff = self.data[-2,:,1]-self.data[-1,:,1]
    #     extra_array[frame_nums:frame_nums+self.data.shape[0]] = self.data
    #     if fade:
    #         introEaseArray=np.zeros((frame_nums,self.data.shape[1]),dtype=np.float64)
    #         outroEaseArray=np.zeros((frame_nums,self.data.shape[1]),dtype=np.float64)
    #         for n in range(introEaseArray.shape[0]):
    #             introEaseArray[n]=easing.inOutQuad(n,zfirstDiff,-zfirstDiff,introEaseArray.shape[0])
    #         for n in range(outroEaseArray.shape[0]):
    #             outroEaseArray[n]=easing.inOutQuad(n,zlastDiff,-zlastDiff,outroEaseArray.shape[0])  
    #     for i in range(frame_nums):
    #         extra_array[self.data.shape[0]+frame_nums+i,:,0] = xylastFrame
    #         if fade:
    #             extra_array[self.data.shape[0]+frame_nums+i,:,1] = extra_array[self.data.shape[0]+frame_nums+i-1,:,1]-outroEaseArray[i]
    #             extra_array[frame_nums-i-1,:,1] = extra_array[frame_nums-i,:,1]-introEaseArray[i]
    #         else:
    #             extra_array[self.data.shape[0]+frame_nums+i,:,1] = extra_array[self.data.shape[0]+frame_nums+i-1,:,1]-zlastDiff
    #             extra_array[frame_nums-i-1,:,1] = extra_array[frame_nums-i,:,1] + zfirstDiff
    #         extra_array[frame_nums-i-1,:,0] = xyfirstFrame
    #     return extra_array
    
    #時間の変化率を維持したまま、始まりと終わり部分を延長させる。fade引数をTrueでスピード０に落ち着かせる。2023.12.11updated 
    def addInterpolation(self,frame_nums,i_direction,z_direction,axis_position,s_reversal=0,z_reversal=0,cycle_degree=90,extra_degree=0,zslide=0,speed_round = True,rrange=[0,1],zscale=1):
        self.interpolation(frame_nums,i_direction,z_direction,axis_position,s_reversal,z_reversal,cycle_degree,extra_degree,zslide,speed_round,rrange,zscale) 
        self.maneuver_log("+IP"+str(frame_nums)+"(ID"+str(i_direction)+"-ZD"+str(z_direction)+"-AP"+str(axis_position)+"-SREV"+str(s_reversal)+"-ZREV"+str(z_reversal)+")")

    def interpolation(self,frame_nums,i_direction,z_direction,axis_position,s_reversal=0,z_reversal=0,cycle_degree=90,extra_degree=0,zslide=0,speed_round = True,rrange=[0,1],zscale=1):
        wr_array= [] 
        # zslide= zslide * self.xyt_boxel_scale
        # zscale = zscale * self.xyt_boxel_scale
        print("sd=",self.scan_direction)
        for i in range(0,frame_nums):
            # print(((i/(frame_nums-1)*(rrange[1]-rrange[0])+rrange[0])*180))
            crad = math.radians(extra_degree)+math.radians(cycle_degree) * i / (frame_nums-1) if speed_round == False else math.radians(extra_degree)+math.radians(cycle_degree)*(1-(math.cos(math.radians((i/(frame_nums-1)*(rrange[1]-rrange[0])+rrange[0])*180))+1.0)/2)
            # crad = math.radians(extra_degree)+math.radians(cycle_degree) * i / (frame_nums-1) if speed_round == False else math.radians(cycle_degree+extra_degree)*(1-(math.cos(math.radians((i)/(frame_nums-1)*180))+1.0)/2)
            sequence=int(crad/math.radians(181))
            zcos=math.cos(crad)
            ysin=math.sin(crad)
            # print(i,sequence)
            write_array=[]
            if self.scan_direction == 0:
                for y in range(0,int(self.height)):
                    if sequence%2==1:
                        crad=math.radians(180)-crad
                        y=(self.height-1)-y
                        zcos=math.cos(crad)
                        ysin=math.sin(crad)
                    if s_reversal : 
                        pernum=y/(self.height-1) if axis_position == 1 else 1.0-y/(self.height-1)
                    else :
                        pernum=y/(self.height-1) if axis_position == 0 else 1.0-y/(self.height-1)
                    yp = zcos*(self.height-1)*pernum if i_direction == 0  else ysin*(self.height-1)*pernum #回転中心点をy＝０か,y＝2160
                    if axis_position:
                        if z_direction:
                            pernum = y/(self.height-1) if z_reversal == 1 else 1.0-y/(self.height-1)
                            # pernum = y/(self.height-1) 
                            zp = ysin*(self.height*zscale)*pernum if i_direction == 0  else zcos*(self.height*zscale)*pernum
                            zp = int(self.height*zscale) - zp
                        else : 
                            pernum = y/(self.height-1) if z_reversal == 1 else 1.0-y/(self.height-1)
                            zp = ysin*(self.height*zscale)*pernum if i_direction == 0  else zcos*(self.height*zscale)*pernum
                    else :  
                        if z_direction:
                            pernum=y/(self.height-1) if z_reversal == 0 else 1.0-y/(self.height-1)
                            # pernum=1.0-y/(self.height-1)
                            zp = ysin*(self.height*zscale)*pernum if i_direction == 0  else zcos*(self.height*zscale)*pernum
                            zp = int(self.height*zscale) - zp
                        else : 
                            pernum=y/(self.height-1) if z_reversal == 0 else 1.0-y/(self.height-1)
                            zp = ysin*(self.height*zscale)*pernum if i_direction == 0  else zcos*(self.height*zscale)*pernum
                    if sequence%2==1:
                        yp=yp*-1 #負の値になっているので、正の値に戻す
                    if axis_position == 1 :
                        yp = (self.height-1) - yp  #反転しちゃうのでキャンセル
                    write_array.append([yp,zp+zslide])
            else:
                for x in range(0,int(self.width)):
                    if sequence%2==1:
                        crad=math.radians(180)-crad
                        x=(self.width-1)-x
                        zcos=math.cos(crad)
                        ysin=math.sin(crad)
                    if s_reversal : 
                        pernum=x/(self.width-1) if axis_position == 1 else 1.0-x/(self.width-1) 
                    else :
                        pernum=x/(self.width-1) if axis_position == 0 else 1.0-x/(self.width-1) 
                    xp = zcos*(self.width-1)*pernum if i_direction == 0  else ysin*(self.width-1)*pernum #回転中心点をx＝０か,x＝3840
                    if axis_position:
                        if z_direction:
                            pernum=x/(self.width-1) if z_reversal == 1 else 1.0-x/(self.width-1)
                            zp = ysin*(self.width*zscale)*pernum if i_direction == 0  else zcos*(self.width*zscale)*pernum
                            zp = int(self.width*zscale) - zp
                        else : 
                            pernum=x/(self.width-1) if z_reversal == 1 else 1.0-x/(self.width-1)
                            zp = ysin*(self.width*zscale)*pernum if i_direction == 0  else zcos*(self.width*zscale)*pernum
                    else :  
                        if z_direction:
                            pernum=x/(self.width-1) if z_reversal == 0 else 1.0-x/(self.width-1)
                            zp = ysin*(self.width*zscale)*pernum if i_direction == 0  else zcos*(self.width*zscale)*pernum
                            zp = int(self.width*zscale) - zp
                        else : 
                            pernum=x/(self.width-1) if z_reversal == 0 else 1.0-x/(self.width-1)
                            zp = ysin*(self.width*zscale)*pernum if i_direction == 0  else zcos*(self.width*zscale)*pernum
                    if sequence%2==1:
                        xp=xp*-1 #負の値になっているので、正の値に戻す
                    if axis_position == 1 :
                        xp = (self.width-1) - xp  #反転しちゃうのでキャンセル
                    write_array.append([xp,zp+zslide])
            write_array=np.array(write_array)
            wr_array.append(write_array) 
        wr_array = np.array(wr_array)
        if len(self.data)!=0: self.data = np.vstack((self.data,wr_array))
        else : self.data = wr_array
        if self.scan_direction % 2 == 0 :
            print("xp range:",np.amin(wr_array[:,:,0]),np.amax(wr_array[:,:,0]))
        else :
            print("yp range:",np.amin(wr_array[:,:,0]),np.amax(wr_array[:,:,0]))
        print("zp range:",np.amin(wr_array[:,:,1]),np.amax(wr_array[:,:,1]))

    # addInterpolationの連結性のある８パターンの次々に実行して配列に加えていく。できた配列を返す。
    def rooting8_interporation(self,FRAME_NUMS):
        self.addInterpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=0,s_reversal=0,z_reversal=0,zslide=0,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=1,z_direction=0,axis_position=0,s_reversal=0,z_reversal=0,zslide=0,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=1,s_reversal=0,z_reversal=0,zslide=0,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=1,z_direction=0,axis_position=1,s_reversal=0,z_reversal=0,zslide=0,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=0,z_direction=1,axis_position=0,s_reversal=0,z_reversal=0,zslide=-self.scan_nums,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=1,z_direction=1,axis_position=0,s_reversal=0,z_reversal=0,zslide=-self.scan_nums,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=0,z_direction=1,axis_position=1,s_reversal=0,z_reversal=0,zslide=-self.scan_nums,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=1,z_direction=1,axis_position=1,s_reversal=0,z_reversal=0,zslide=-self.scan_nums,speed_round = True)

    def rooting8B_interporation(self,FRAME_NUMS):
        self.addInterpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=0,s_reversal=1,z_reversal=1,zslide=0,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=1,z_direction=0,axis_position=0,s_reversal=1,z_reversal=1,zslide=0,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=1,s_reversal=1,z_reversal=1,zslide=0,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=1,z_direction=0,axis_position=1,s_reversal=1,z_reversal=1,zslide=0,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=0,z_direction=1,axis_position=0,s_reversal=1,z_reversal=1,zslide=self.scan_nums,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=1,z_direction=1,axis_position=0,s_reversal=1,z_reversal=1,zslide=self.scan_nums,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=0,z_direction=1,axis_position=1,s_reversal=1,z_reversal=1,zslide=self.scan_nums,speed_round = True)
        self.addInterpolation(FRAME_NUMS,i_direction=1,z_direction=1,axis_position=1,s_reversal=1,z_reversal=1,zslide=self.scan_nums,speed_round = True)

        # addInterpolationの連結性のある８パターンの次々に実行して配列に加えていく。できた配列を返す。
    def rootingA_interporation(self,FRAME_NUMS,loop_num=2,axis_first_p=0,speed_round=True,interval_nums=0,loopinterval_nums=0):
        ap=axis_first_p
        r=0
        z=0
        id=0
        for i in range(loop_num):
            if i != 0 and loopinterval_nums > 0 :
                 self.addExtend(loopinterval_nums)
            if interval_nums > 0 :
                if self.log==0:
                    self.addFlat(int(interval_nums//2))
                else :
                    self.addExtend(int(interval_nums//2))
            self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=id%2,z_direction=z%2,axis_position=ap%2,s_reversal=r%2,z_reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round)
            r+=1
            z+=1
            id+=1
            if interval_nums > 0 :
                self.addExtend(interval_nums)
            self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=id%2,z_direction=z%2,axis_position=ap%2,s_reversal=r%2,z_reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round)
            ap+=1
            z+=1
            id+=1
            if interval_nums > 0 :
                self.addExtend(interval_nums//2)
        self.maneuver_log("IP"+str(FRAME_NUMS)+"(rootingA)")
    
 
    #     # addInterpolationの連結性のある８パターンの次々に実行して配列に加えていく。できた配列を返す。
    # def rootingA_interporation_Random(self,FRAME_NUMS,loop_num=2,axis_first_p=0,speed_round=True,interval_nums=0,loopinterval_nums=0):
    #     ap=axis_first_p
    #     r=0
    #     z=0
    #     id=0
    #     for i in range(loop_num):
    #         if i != 0 and loopinterval_nums > 0 :
    #              self.addExtend(loopinterval_nums)
    #         if interval_nums > 0 :
    #             if self.log==0:
    #                 self.addFlat(int(interval_nums//2))
    #             else :
    #                 self.addExtend(int(interval_nums//2))
    #         self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=id%2,z_direction=z%2,axis_position=ap%2,s_reversal=r%2,z_reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round)
    #         r+=1
    #         z+=1
    #         id+=1
    #         if interval_nums > 0 :
    #             self.addExtend(interval_nums)
    #         self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=id%2,z_direction=z%2,axis_position=ap%2,s_reversal=r%2,z_reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round)
    #         ap+=1
    #         z+=1
    #         id+=1
    #         if interval_nums > 0 :
    #             self.addExtend(interval_nums//2)
    #     self.maneuver_log("IP"+str(FRAME_NUMS)+"(rootingA)")
       
    
    def rootingA_interporation_RANDOM(
        self,
        FRAME_NUMS_range,                 # (min,max) or int
        interval_nums_range=(0, 0),       # (min,max) or int  ※毎周ランダム
        loopinterval_nums_range=(0, 0),   # (min,max) or int
        loop_num=2,
        axis_first_p=0,
        speed_round=True,
        seed=None,
        clamp_even=False,
        randomize_loopinterval_each_loop=False,  # Trueでloopintervalも毎周ランダム
        min_step_frames=1                 # interpolationのフレームが0にならない保険
        ):  
        if seed is not None:
            random.seed(seed)

        def _pick_int(rng, name):
            if isinstance(rng, (list, tuple)):
                if len(rng) != 2:
                    raise ValueError(f"{name} は (min,max) で渡してね")
                a, b = int(rng[0]), int(rng[1])
                if a > b:
                    a, b = b, a
                v = random.randint(a, b)
            else:
                v = int(rng)

            if clamp_even:
                v = (v // 2) * 2

            return max(0, v)

        # FRAME_NUMS は最初に1回だけ決める（実行中にブレない）
        FRAME_NUMS = _pick_int(FRAME_NUMS_range, "FRAME_NUMS_range")

        # loopinterval は基本固定（オプションで毎周ランダム）
        loopinterval_fixed = _pick_int(loopinterval_nums_range, "loopinterval_nums_range")

        ap = axis_first_p
        r = 0
        z = 0
        id = 0

        step_frames = int(FRAME_NUMS / (loop_num * 2))
        if step_frames < min_step_frames:
            step_frames = min_step_frames

        for i in range(loop_num):
            # ★ここが「毎周ランダム」：interval_nums をループのたびに引き直す
            interval_nums = _pick_int(interval_nums_range, "interval_nums_range")

            # loopinterval も毎周ランダムにしたいならここで引き直す
            loopinterval_nums = (
                _pick_int(loopinterval_nums_range, "loopinterval_nums_range")
                if randomize_loopinterval_each_loop
                else loopinterval_fixed
            )

            if i != 0 and loopinterval_nums > 0:
                self.addExtend(loopinterval_nums)

            if interval_nums > 0:
                if self.log == 0:
                    self.addFlat(int(interval_nums // 2))
                else:
                    self.addExtend(int(interval_nums // 2))

            self.interpolation(
                step_frames,
                i_direction=id % 2,
                z_direction=z % 2,
                axis_position=ap % 2,
                s_reversal=r % 2,
                z_reversal=r % 2,
                zslide=(self.scan_nums - 1) * i,
                speed_round=speed_round
            )

            r += 1
            z += 1
            id += 1

            if interval_nums > 0:
                self.addExtend(interval_nums)

            self.interpolation(
                step_frames,
                i_direction=id % 2,
                z_direction=z % 2,
                axis_position=ap % 2,
                s_reversal=r % 2,
                z_reversal=r % 2,
                zslide=(self.scan_nums - 1) * i,
                speed_round=speed_round
            )

            ap += 1
            z += 1
            id += 1

            if interval_nums > 0:
                self.addExtend(interval_nums // 2)

        self.maneuver_log(
            f"IP{FRAME_NUMS}(rootingA_RANDOM_EACHLOOP)"
            f"[interval={interval_nums_range}, loopinterval={loopinterval_nums_range}, "
            f"loopinterval_each={randomize_loopinterval_each_loop}]"
        )
        # addInterpolationの連結性のある８パターンの次々に実行して配列に加えていく。できた配列を返す。
    def rootingA_interporation_single(self,FRAME_NUMS,seg_type=0,speed_round=True,interval_nums=0,panorama_nums=0,flip_axis=False,junction_mode=0,blur_rate=90,Second_FRAME_NUMS=None,center_time_frame=None):
        """
        前半と後半の2サブセグメントを内包した1ユニット。
        ─ seg_typeとパラメータの対応 (flip_axis=False) ──────────
          seg_type | s_front | axis_position | 時間方向
             0     |    0    |      0        |   +1(前半順/後半逆)
             1     |    1    |      1        |   -1(前半逆/後半順)
             2     |    0    |      1        |   +1(前半順/後半逆)
             3     |    1    |      0        |   -1(前半逆/後半順)
        ─ flip_axis=True のとき axis_position を反転 ────────────
          seg_type | s_front | axis_position | 時間方向
             0     |    0    |      1        |   +1(前半順/後半逆)
             1     |    1    |      0        |   -1(前半逆/後半順)
             2     |    0    |      0        |   +1(前半順/後半逆)
             3     |    1    |      1        |   -1(前半逆/後半順)
        ─ axis_position = [0,1,1,0][seg_type%4] XOR flip_axis  ──
        ─ s_reversal   = seg_type % 2 (0,2→0 / 1,3→1)         ──
        ─ 空間接続チェーン(axis_positionが同じ連続呼び出し時) ──────
          偶数末端: reversed → 奇数始点(reversed)✓
          奇数末端: identity → 偶数始点(identity)✓
        ─ 前半は常にi_direction=0, 後半は常にi_direction=1 ─────
        panorama_nums: 前半・後半の間(空間zeros状態)に追加する停留フレーム数。
          全フレームを前半側(junction_base直前)に挿入。後半側には挿入しない。
          → 前半 interpolation → interval//2 → panorama_nums → [junction_base]
            → interval//2 → 後半 interpolation → interval//2
        junction_shift = interval_nums//4 で junction を後半flat領域の序盤に配置。
        slide_time = int(recfps/outfps) 自動計算。blur_time = FRAME_NUMS//3 自動計算。
        """
        s_front  = seg_type % 2                               # 前半 s_reversal: 0,2→0 / 1,3→1
        s_back   = 1 - s_front                                # 後半 s_reversal: 前半と逆
        axis_pos = [0,1,1,0][seg_type % 4] ^ int(flip_axis)  # flip_axis=TrueでXOR反転(0↔1)
        flip     = (seg_type % 2 == 1)  # addFlat初回のみ使用
        seg_start= len(self.data)
        # z anchor: axis_pos=0→scan_line 0がzero基点 / axis_pos=1→最終scan_lineがzero基点
        z_anchor = -1 if axis_pos == 1 else 0

        # ── 前半: i_direction=0, z_direction=0, s_reversal=s_front, z_reversal=0 ──
        #    start: s_front=0→identity / s_front=1→reversed   end: 常にzeros
        if interval_nums > 0:
            if self.log==0:
                self.addFlat(int(interval_nums//2),flip=flip)  # 初回: s_frontに合わせた始点
            else:
                self.addExtend(int(interval_nums//2))          # 続き: 前セグから自然延長
        junc_p1=len(self.data)                                 # ① 前半flat終端 / 前半interp始端
        zslide=self.data[-1,z_anchor,1] if len(self.data)>0 else 0
        self.interpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=axis_pos,s_reversal=s_front,z_reversal=0,speed_round=speed_round,zslide=zslide)
        junc_p2=len(self.data)                                 # ② 前半interp終端 / flat+panorama始端
        if interval_nums > 0:
            self.addExtend(interval_nums//2)                   # 前半終端(zeros)をそのまま延長
        panorama_start = len(self.data)                        # パノラマ開始位置（junction_mode=2 用）
        if panorama_nums > 0:
            self.addExtend(panorama_nums)                      # パノラマ: zeros状態を前半側に全量停留

        junction_base=len(self.data)                           # パノラマ含む前半全フレームの終端

        # ── 後半: i_direction=1, z_direction=1, s_reversal=s_back, z_reversal=1 ──
        #    start: 常にzeros(前半終端と一致)   end: s_back=1→reversed / s_back=0→identity
        if interval_nums > 0:
            self.addExtend(int(interval_nums//2))              # 後半前フラット: zeros状態をそのまま延長
        junc_p3=len(self.data)                                 # ③ flat+panorama終端 / 後半interp始端
        zslide=self.data[-1,z_anchor,1]
        fnum_back = Second_FRAME_NUMS if Second_FRAME_NUMS is not None else FRAME_NUMS
        self.interpolation(fnum_back,i_direction=1,z_direction=1,axis_position=axis_pos,s_reversal=s_back,z_reversal=1,speed_round=speed_round,zslide=zslide)
        junc_p4=len(self.data)                                 # ④ 後半interp終端 / 後半flat始端
        if interval_nums > 0:
            self.addExtend(interval_nums//2)

        # ── junction選択 (junction_mode) ──
        # 0: パノラマ終端 + shift（既存動作）
        # 1: 前半interp終端
        # 2: パノラマ中間
        if junction_mode == 1:
            junction = junc_p2
        elif junction_mode == 2:
            junction = panorama_start + panorama_nums // 2
        else:  # junction_mode == 0
            junction_shift = interval_nums // 2
            junction = junction_base + junction_shift

        # ── 時間方向: 前半と後半に逆方向のslide_timeを適用 ──
        slide_time=int(self.recfps/self.outfps)
        direction=1 if seg_type%2==0 else -1
        self.applyTimeForward(direction*slide_time,start_frame=seg_start,end_frame=junction)
        self.applyTimeForward(-direction*slide_time,start_frame=junction)
        # ── セグメント境界のstagnation回避 ─────────────────────
        # applyTimeForwardはseg_start先頭に+0を加算するため、
        # 前セグ末フレームとseg_start先頭が同一z値(変化率ゼロ)になってしまう。
        # 2回目以降(self.log!=0)はdirection*slide_time分だけ現セグ全体をオフセット。
        if self.log != 0:
            self.data[seg_start:, :, 1] += direction * slide_time

        # ── blur①: junction中心 / 範囲は当該セグメントのblur_rate% ────────
        seg_frames   = len(self.data) - seg_start  # 当該セグメントのフレーム数
        blur_margin  = max(1, int(seg_frames * blur_rate / 200))  # ÷2 → 片側分
        blur_time    = max(1, blur_margin // 2)
        self.applyCustomeBlur(s_frame=max(0,junction-blur_margin),
                              e_frame=min(len(self.data),junction+blur_margin),
                              bl_time=blur_time,dim_num=1)
        self.applyCustomeBlur(s_frame=max(0,junction-blur_margin),
                              e_frame=min(len(self.data),junction+blur_margin),
                              bl_time=blur_time,dim_num=0)

        # ── blur②: 接続点境界にタイム+空間ブラー ───────────────
        total = len(self.data)
        if interval_nums > 0:
            # interval_numsあり: 4接続点(interp↔interval境界)を ±interval_nums//2 でカバー
            # junc_p1: 前半flat終端/前半interp始端
            # junc_p2: 前半interp終端/flat+panorama始端
            # junc_p3: flat+panorama終端/後半interp始端
            # junc_p4: 後半interp終端/後半flat始端
            ibl  = max(1, interval_nums//4)
            half = interval_nums//2
            for pt in [junc_p1, junc_p2, junc_p3, junc_p4]:
                s = max(0, pt - half)
                e = min(total, pt + half)
                self.applyCustomeBlur(s_frame=s,e_frame=e,bl_time=ibl,dim_num=1)
                self.applyCustomeBlur(s_frame=s,e_frame=e,bl_time=ibl,dim_num=0)
        elif panorama_nums > 0:
            # interval_nums=0 かつ panorama_numsあり:
            # 前半interp終端(junc_p2)と後半interp始端(junc_p3)の2点を
            # ±panorama_nums//2 の範囲でカバー
            pbl  = max(1, panorama_nums//4)
            half = panorama_nums//2
            for pt in [junc_p2, junc_p3]:
                s = max(0, pt - half)
                e = min(total, pt + half)
                self.applyCustomeBlur(s_frame=s,e_frame=e,bl_time=pbl,dim_num=1)
                self.applyCustomeBlur(s_frame=s,e_frame=e,bl_time=pbl,dim_num=0)

        # ── zArange: junction フレームの z を center_time_frame へスライド ──
        if center_time_frame is not None:
            self.zArange(junction, center_time_frame)

        self.maneuver_log("IP"+str(FRAME_NUMS)+"-"+str(panorama_nums)+"-"+str(Second_FRAME_NUMS)+"-i"+str(interval_nums)+(sys._getframe().f_code.co_name))

    def rootingA_interporation_trans_single(self,FRAME_NUMS,seg_type=0,speed_round=True,interval_nums=0,trans_nums=0,trans_end_line=0,flip_axis=False,junction_mode=0,blur_rate=90,time_flip=False,Second_FRAME_NUMS=None,center_time_frame=None):
        """
        rootingA_interporation_single の trans拡張版。
        panorama_nums(addExtend)の代わりに addTrans() で空間位置を移動させる。
        前半と後半で axis_position が異なる場合がある点が通常版との主な違い。

        ─ 前半 axis_position = [0,1,1,0][seg_type%4] ^ flip_axis  (通常版と同じ)
        ─ 後半 axis_position = int(trans_end_line) で独立決定
            trans_end_line=0: addTransがzeros(0)へ移動 → 後半はax=0(zeros始点)
            trans_end_line=1: addTransがwidth-1へ移動  → 後半はax=1(width-1始点)
        ─ フレーム構造:
            [interval//2] → interp(前半) → [interval//2] → addTrans(trans_nums)
            → [junction_base] → [interval//2] → interp(後半) → [interval//2]
        ─ junction_shift = interval_nums//4
        ─ slide_time / blur_time は通常版と同じ自動計算
        ─ junction_mode (int, default=0): 順→逆の時間切り替え点
            0: addTrans終端(junction_base)  [デフォルト]
               → 前半interp+addTransが順再生 / 後半interpが逆再生
            1: 前半interp終端(junc_p2)
               → 前半interpのみ順再生 / addTrans+後半interpが丸ごと逆再生
            2: addTrans中間(trans_start + trans_nums//2)
               → 前半interp+addTrans前半が順再生 / addTrans後半+後半interpが逆再生
        ─ blur_rate (int, default=90): junction中心ブラー①の範囲を総フレーム数の%で指定
               blur_margin = total_frames * blur_rate / 100 / 2  (junction ± blur_margin)
               blur_time   = blur_margin // 2
               例) blur_rate=90, 総60秒 → ±27秒 / blur_rate=10 → ±3秒
        ─ time_flip (bool, default=False): interpolation の時間方向(z分布)を反転
            False: 前半 z_reversal=0 → axis_pos=1時は左端(x=0)が時間先行
            True : 前半 z_reversal=1 → axis_pos=1時は右端(x=width-1)が時間先行
                   (右端固定・左端が時間後方へ移動するように見える)
            z_anchor と z_reversal_back も連動して自動反転。
        """
        s_front      = seg_type % 2
        s_back       = 1 - s_front
        axis_pos     = [0,1,1,0][seg_type % 4] ^ int(flip_axis)  # 前半 axis_position
        # seg_type=1 系では、前半interp終端(axis_pos)と trans_end_line が一致し
        # addTransが空間移動しない問題があるため、自動で反対側へ横断させる。
        # seg_type=0 系は従来通り trans_end_line をそのまま使う。
        if seg_type % 2 == 1:
            trans_target  = 1 - axis_pos                          # addTransの行き先 = axis_posの反対
            axis_pos_back = 1 - axis_pos                          # 後半interpもその位置から
        else:
            trans_target  = trans_end_line
            axis_pos_back = int(round(trans_end_line))            # 後半 axis_position: trans終端から決定
        seg_start= len(self.data)

        # ── time_flip によるz方向制御 ────────────────────────────────────────
        # z_reversal_front: 前半のz_reversal。time_flip=TrueでXOR反転。
        # z_reversal_back : 後半のz_reversal。axis_pos変化時は連続性のため反転し、さらにtime_flipでXOR反転。
        # z_anchor        : zslide読み出し基点。axis_pos=1 XOR time_flip で決定。
        #   (axis_pos=1, time_flip=False → -1 / axis_pos=1, time_flip=True → 0 など)
        z_reversal_front = int(time_flip)
        z_reversal_back  = (1 if axis_pos_back == axis_pos else 0) ^ int(time_flip)
        z_anchor         = -1 if (axis_pos == 1) != bool(time_flip) else 0

        flip           = (seg_type % 2 == 1)
        seg_start      = len(self.data)

        # ── 前半: i_direction=0, axis_position=axis_pos ──────────────────
        if interval_nums > 0:
            if self.log==0:
                self.addFlat(int(interval_nums//2),flip=flip)
            else:
                self.addExtend(int(interval_nums//2))
        junc_p1=len(self.data)                                   # ① 前半flat終端/前半interp始端
        zslide=self.data[-1,z_anchor,1] if len(self.data)>0 else 0
        self.interpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=axis_pos,s_reversal=s_front,z_reversal=z_reversal_front,speed_round=speed_round,zslide=zslide,zscale=self.xyt_boxel_scale)
        junc_p2=len(self.data)                                   # ② 前半interp終端/addTrans始端
        if interval_nums > 0:
            self.addExtend(interval_nums//2)

        # ── addTrans: 前半終端からtrans_end_line位置へ空間移動 ─────────────
        trans_start=len(self.data)                               # addTrans始端(junction_mode=2用)
        if trans_nums > 0:
            self.addTrans(trans_nums,end_line=trans_target,speed_round=speed_round)

        junction_base=len(self.data)                             # addTrans終端

        # ── 後半: i_direction=1, axis_position=axis_pos_back ─────────────
        if interval_nums > 0:
            self.addExtend(int(interval_nums//2))
        junc_p3=len(self.data)                                   # ③ addTrans終端/後半interp始端
        zslide=self.data[-1,z_anchor,1]                         # 前半のz_anchorから読む(axis_pos変化時も連続)
        fnum_back = Second_FRAME_NUMS if Second_FRAME_NUMS is not None else FRAME_NUMS
        self.interpolation(fnum_back,i_direction=1,z_direction=1,axis_position=axis_pos_back,s_reversal=s_back,z_reversal=z_reversal_back,speed_round=speed_round,zslide=zslide,zscale=self.xyt_boxel_scale)
        junc_p4=len(self.data)                                   # ④ 後半interp終端/後半flat始端
        if interval_nums > 0:
            self.addExtend(interval_nums//2)

        # ── junction: 順→逆の時間切り替え点を junction_mode で選択 ──────────
        if junction_mode == 1:
            junction = junc_p2                                   # 前半interp終端
        elif junction_mode == 2:
            junction = trans_start + trans_nums // 2             # addTrans中間
        else:                                                     # 0(デフォルト)
            junction_shift = interval_nums // 4
            junction       = junction_base + junction_shift      # addTrans終端

        slide_time = int(self.recfps/self.outfps)
        direction  = 1 if seg_type%2==0 else -1
        self.applyTimeForward( direction*slide_time, start_frame=seg_start, end_frame=junction)
        self.applyTimeForward(-direction*slide_time, start_frame=junction)
        if self.log != 0:
            self.data[seg_start:,:,1] += direction * slide_time

        # ── blur①: junction中心 / 範囲は総フレーム数のblur_rate% ─────────
        # blur_margin: junction ± blur_rate/2 % の範囲をカバー
        # blur_time  : blur_marginの半分 (ブラーカーネル幅)
        seg_frames = len(self.data)- seg_start  # 当該セグメントのフレーム数
        blur_margin  = max(1, int(seg_frames * blur_rate / 200))  # ÷2 → 片側分
        blur_time    = max(1, blur_margin // 2)
        # self.applyCustomeBlur(s_frame=max(0,junction-blur_margin),
        #                       e_frame=min(seg_frames,junction+blur_margin),
        #                       bl_time=blur_time,dim_num=1)
        # self.applyCustomeBlur(s_frame=max(0,junction-blur_margin),
        #                       e_frame=min(seg_frames,junction+blur_margin),
        #                       bl_time=blur_time,dim_num=0)
        
        self.applyCustomeBlur(s_frame=max(0,junction-blur_margin),
                              e_frame=min(len(self.data),junction+blur_margin),
                              bl_time=blur_time,dim_num=1)
        self.applyCustomeBlur(s_frame=max(0,junction-blur_margin),
                              e_frame=min(len(self.data),junction+blur_margin),
                              bl_time=blur_time,dim_num=0)

        # ── blur②: 4接続点境界(interp↔interval/trans境界) ─────────────────
        total = len(self.data)
        if interval_nums > 0:
            ibl  = max(1, interval_nums//4)
            half = interval_nums//2
            for pt in [junc_p1, junc_p2, junc_p3, junc_p4]:
                s = max(0, pt - half)
                e = min(total, pt + half)
                self.applyCustomeBlur(s_frame=s,e_frame=e,bl_time=ibl,dim_num=1)
                self.applyCustomeBlur(s_frame=s,e_frame=e,bl_time=ibl,dim_num=0)
        elif trans_nums > 0:
            # interval_nums=0 かつ trans_numsあり: addTrans両端2点を±trans_nums//2でカバー
            tbl  = max(1, trans_nums//4)
            half = trans_nums//2
            for pt in [junc_p2, junc_p3]:
                s = max(0, pt - half)
                e = min(total, pt + half)
                self.applyCustomeBlur(s_frame=s,e_frame=e,bl_time=tbl,dim_num=1)
                self.applyCustomeBlur(s_frame=s,e_frame=e,bl_time=tbl,dim_num=0)

        # ── zArange: junction フレームの z を center_time_frame へスライド ──
        if center_time_frame is not None:
            self.zArange(junction, center_time_frame)

        self.maneuver_log("IP"+str(FRAME_NUMS)+"-"+str(trans_nums)+"-"+str(Second_FRAME_NUMS)+"-i"+str(interval_nums)+(sys._getframe().f_code.co_name))

        # addInterpolationの連結性のある８パターンの次々に実行して配列に加えていく。できた配列を返す。
    def rootingAA_interporation(self,FRAME_NUMS,loop_num=2,axis_first_p=0,speed_round=True):
        ap=axis_first_p
        r=0
        for i in range(loop_num):
            if r%2 == 0 :
                self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=0,z_direction=0,axis_position=ap%2,s_reversal=r%2,z_reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round)
            else :
                self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=0,z_direction=0,axis_position=ap%2,s_reversal=r%2,z_reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round,cycle_degree=180,extra_degree=-90,rrange=[0.5,1])
            r+=1
            if r%2 == 1 :
                self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=1,z_direction=1,axis_position=ap%2,s_reversal=r%2,z_reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round)
            else : 
                self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=1,z_direction=1,axis_position=ap%2,s_reversal=r%2,z_reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round,cycle_degree=180,extra_degree=0,rrange=[0,0.5])
            
            ap+=1
        self.maneuver_log("IP"+str(FRAME_NUMS)+"(rootingAA)")
          
        # loop_num=2でi=1の連結の際に、フリーズフレームが発生してしまう
    def rootingB_interporation(self,FRAME_NUMS,loop_num=1,axis_fix_p=0):
        r=0
        for i in range(loop_num):
            self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=1,z_direction=1,axis_position=axis_fix_p,reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = True,cycle_degree=180,rrange=[0.5,1])
            r+=1
            self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=1,z_direction=1,axis_position=axis_fix_p,reversal=r%2,zslide=(self.scan_nums-1)*i+(self.scan_nums-1), speed_round = True,cycle_degree=180,rrange=[0,0.5])
        self.maneuver_log("IP"+str(FRAME_NUMS)+"(rootingB_axis"+str(axis_fix_p)+")")


    def rooting4C_interporation(self,FRAME_NUMS):
        self.interpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=0,s_reversal=0,z_reversal=0,zslide=0,speed_round = True)
        self.interpolation(FRAME_NUMS,i_direction=1,z_direction=1,axis_position=0,s_reversal=0,z_reversal=0,zslide=0,speed_round = True)
        self.interpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=1,s_reversal=0,z_reversal=0,zslide=self.scan_nums,speed_round = True)
        self.interpolation(FRAME_NUMS,i_direction=1,z_direction=1,axis_position=1,s_reversal=0,z_reversal=0,zslide=self.scan_nums,speed_round = True)

    def rooting4D_interporation(self,FRAME_NUMS):
        self.interpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=0,s_reversal=0,z_reversal=0,zslide=0,speed_round = True)
        self.interpolation(FRAME_NUMS,i_direction=1,z_direction=0,axis_position=0,s_reversal=1,z_reversal=1,zslide=0,speed_round = True)
        self.interpolation(FRAME_NUMS,i_direction=0,z_direction=0,axis_position=0,s_reversal=1,z_reversal=1,zslide=0,speed_round = True)
        self.interpolation(FRAME_NUMS,i_direction=1,z_direction=0,axis_position=0,s_reversal=0,z_reversal=0,zslide=0,speed_round = True)

    # wr_arrayに新たなTrans（）の軌跡を加えて返す関数    
    def addTrans(self,frame_nums,start_line=0,end_line=1,speed_round = True,zd=True,zscale=1):
        zscale = zscale * self.xyt_boxel_scale
        if len(self.data) != 0 : 
            extra_array = np.zeros((self.data.shape[0]+frame_nums,self.scan_nums,2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
            extra_array[0:self.data.shape[0]] = self.data
            outroFrameXP = self.data[-1,:,0]
            outroFrameZ = self.data[-1,:,1]
        else : 
            extra_array = np.zeros((frame_nums,self.scan_nums,2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
            outroFrameXP = np.full(self.scan_nums,start_line*(self.scan_nums-1)) 
            # outroFrameZ = np.arange(0,self.scan_nums* zscale,1 * zscale) if zd else np.arange(self.scan_nums * zscale -1, -1, -1 * zscale)
            if zd:
                outroFrameZ = np.linspace(
                    0,
                    (self.scan_nums - 1) * zscale,
                    self.scan_nums,
                    dtype=np.float64
                )
            else:
                outroFrameZ = np.linspace(
                    (self.scan_nums - 1) * zscale,
                    0,
                    self.scan_nums,
                    dtype=np.float64
                )
            outroFrameZ = np.round(outroFrameZ).astype(int)
            
        endFrame = np.full(self.scan_nums,end_line*(self.scan_nums-1)) 
        normalFrame=np.arange(0,self.scan_nums)
        diff_array = endFrame - outroFrameXP
        if len(self.data) != 0 : 
            for i in range(frame_nums):
                extra_array[self.data.shape[0]+i,:,0]=extra_array[self.data.shape[0]+i-1,:,0]+(diff_array/frame_nums) if speed_round == False else outroFrameXP + diff_array*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)
                extra_array[self.data.shape[0]+i,:,1]=outroFrameZ
        else:
            for i in range(frame_nums):
                extra_array[i,:,0]=outroFrameXP+(diff_array/frame_nums)*i if speed_round == False else outroFrameXP + diff_array*(math.sin(math.radians(i/(frame_nums)*180-90))+1.0)/2
                extra_array[i,:,1]=outroFrameZ
        
        self.data = extra_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1]+str(frame_nums)+"ease"+str(speed_round))

    """
    self.dataで受け取った配列の最後の２フレームを比較して、差分を出し、最終フレームでベクトル（移動方向と速度）を空間領域、時間領域ごとに取得する。
    最後から二つ手前のフレームと一つ手前のフレームも計測して、ベクトルを出し、最終フレームとのベクトルを割ることで変化度合い(acceleration_xyp,acceleration_zp)を抽出。
    １であれば、一定の変化。１以下であれば、収束に向かう。１以上であれば加速していることが判る。
    "under_xyp"　"over_xyp"で指定した空間領域の位置を越えるまで、その移動歩行と速度に対して、変化度合いをマイフレーム掛け合わせながら新規フレームを追加していく。
    """
    def addKeepSpeedTrans(self,frame_nums,under_xyp=None,over_xyp=1,rendertype=0):
        if under_xyp == None : under_xyp = self.scan_nums
        # rendertype１ の場合は、配列ごとに移動の差分ベクトルを取得する。0は平均のベクトルを出す。
        print(sys._getframe().f_code.co_name)
        outroFrameZ = self.data[-1,:,1]
        #xp,ypの空間領域(SpaceDomain)の差分計算
        vector_xyp = self.data[-1,:,0]-self.data[-2,:,0] if rendertype != 0 else np.mean( self.data[-1,:,0]-self.data[-2,:,0] )
        prevector_xyp = self.data[-2,:,0]-self.data[-3,:,0] if rendertype != 0 else np.mean( self.data[-2,:,0]-self.data[-3,:,0] )
        acceleration_xyp = vector_xyp/prevector_xyp 
        
        #zpの差分計算
        vector_zp = self.data[-1,:,1]-self.data[-2,:,1] if rendertype != 0 else np.mean( self.data[-1,:,1]-self.data[-2,:,1] )
        prevector_zp = self.data[-2,:,1]-self.data[-3,:,1] if rendertype != 0 else np.mean( self.data[-2,:,1]-self.data[-3,:,1] )
        acceleration_zp = vector_zp/prevector_zp 
        # print(acceleration)
        normalFrame=np.arange(0,self.scan_nums)
        n=0
        while np.amax(self.data[-1,:,0]) < under_xyp :
            newframe=np.zeros((1,self.scan_nums,2),dtype=np.float64)
            vector_xyp=vector_xyp*acceleration_xyp
            vector_zp=vector_zp*acceleration_zp
            # print(n,vector_xyp[0])
            # print(n,vector_xyp)
            # if vector_xyp < 1.0 : vector_xyp=1
            newframe[:,:,0] = self.data[-1,:,0] + vector_xyp
            newframe[:,:,1] = self.data[-1,:,1] + vector_zp
            # plt.plot(newframe[0,:,0])
            if np.amax(newframe[:,:,0]) > under_xyp:
                print("under limit _addkeepspeed=",n)
                break
            if np.amin(newframe[:,:,0]) < over_xyp:
                print("over_limit addkeepspeed=",n)
                break
            self.data=np.vstack((self.data,newframe))
            n+=1
            if n == frame_nums:
                print("addkeepspeed=",n)
                break
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1])

    # 線分ABと線分CDの交点を求める関数
    def cross_point(pointA, pointB, pointC, pointD):
        cross_point = (0,0)
        bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])
        # 直線が平行な場合は(0,0)を返す
        if (bunbo == 0):
            return False, cross_point
        vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
        r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1]) / bunbo
        s = ((pointB[1] - pointA[1]) * vectorAC[0] - (pointB[0] - pointA[0]) * vectorAC[1]) / bunbo
        distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
        cross_point = (int(pointA[0] + distance[0]), int(pointA[1] + distance[1]))
        return True, cross_point

    # addKeepSpeedTrans を発展させた変数
    # self.data、に対して、after_arrayで受け取った配列の間を滑らかに補う。after_arrayは自動的にzslideさせる
    def addInsertKeepSpeedTrans(self,frame_nums,under_xyp=None,over_xyp=1,after_array=[],rendertype=0):
        # rendertype １ の場合は、配列ごと移動のベクトルを取得する。0は平均のベクトルを出す。
        print(sys._getframe().f_code.co_name)
        if under_xyp==None:
            under_xyp=self.width-1
        # ラスト1frameのxpの差分計算
        delta_xyp = self.data[-1,:,0]-self.data[-2,:,0] if rendertype != 0 else np.mean(self.data[-1,:,0]-self.data[-2,:,0])
        #ラスト1frameのzpの差分計算
        delta_zp = self.data[-1,:,1]-self.data[-2,:,1] if rendertype != 0 else np.mean(self.data[-1,:,1]-self.data[-2,:,1])
        #最初の１フレームのafter_xypの差分計算
        delta_after_xyp = after_array[1,:,0]-after_array[0,:,0] if rendertype != 0 else np.mean(after_array[1,:,0]-after_array[0,:,0])
        #最初の１フレームのafter_zpの差分計算
        delta_after_zp = after_array[1,:,1]-after_array[0,:,1] if rendertype != 0 else np.mean(after_array[1,:,1]-after_array[0,:,1])

        # right_side = np.amax(self.data[:,:,0])
        # left_side = np.amin(self.data[:,:,0])
        # #xq2の計算
        # xret2,xq2 =cross_point((0,self.data[-1,0,1]),(1,self.data[-1,0,1]+ delta_xyp) ,(frame_nums,after_array[0,0,1]),(frame_nums-1,after_array[0,0,1] - delta_after_xyp) )
        # print(xret2,xq2)
        # xret2,xq2 =cross_point((0,self.data[-1,0,1]),(1,self.data[-1,0,1]+ delta_xyp) , (0,right_side), (1,right_side))
        # print(xret2,xq2)
        # #xq3の計算
        # xret3,xq3 =cross_point((frame_nums,after_array[0,0,1]),(frame_nums-1,after_array[0,0,1] - delta_after_xyp) , (0,right_side), (1,right_side))
        # print(xret3,xq3)

        
        deltaZ_front_back=delta_after_zp-delta_zp
        deltaXYP_front_back=delta_after_xyp-delta_xyp

        intermediateZarray=np.zeros((frame_nums,self.data.shape[1]),dtype=np.float64)
        intermediateXYarray=np.zeros((frame_nums,self.data.shape[1]),dtype=np.float64)
        for n in range(intermediateZarray.shape[0]):
            intermediateZarray[n]=easing.inOutQuad(n,delta_zp,deltaZ_front_back,intermediateZarray.shape[0])
        for n in range(intermediateXYarray.shape[0]):
            intermediateXYarray[n]=easing.inOutQuad(n,delta_xyp,deltaXYP_front_back,intermediateXYarray.shape[0]) 
        # 2023/7/3 XYPのズレ補正
        xy_gap=self.data[-1,:,0]+np.sum(intermediateXYarray,axis=0)-after_array[0,:,0]
        intermediateXY_Gaps=np.zeros((frame_nums,self.data.shape[1]),dtype=np.float64)
        for n in range(intermediateXY_Gaps.shape[0]):
            intermediateXY_Gaps[n]=easing.inOutQuad(n,0,-xy_gap,intermediateXY_Gaps.shape[0])  

        for n in range(frame_nums):
            newframe=np.zeros((1,self.scan_nums,2),dtype=np.float64)
            # delta_xyp=delta_xyp*acceleration_xyp
            # delta_zp=delta_zp*acceleration_zp
            # print(n,delta_xyp)
            # 2023/7/3 以下書き換え、
            # newframe[:,:,0]=self.data[-1,:,0] + (deltaXYP_front_back/frame_nums * n + delta_xyp)
            # newframe[:,:,1]=self.data[-1,:,1] + (deltaZ_front_back/frame_nums * n + delta_zp)
            newframe[:,:,0] = self.data[-1,:,0] + intermediateXYarray[n]
            newframe[:,:,1] = self.data[-1,:,1] + intermediateZarray[n]
            if np.amin(newframe[:,:,0]) < over_xyp:
                newframe[:,:,0]=self.data[-1,:,0] 
            if np.amax(newframe[:,:,0]) > under_xyp:
                newframe[:,:,0]=self.data[-1,:,0]
            self.data=np.vstack((self.data,newframe))
        
         # 2023/7/3 XYPのズレ補正
        for n in range(intermediateXY_Gaps.shape[0]):
            self.data[-1-n,:,0]+=intermediateXY_Gaps[-1-n]

        if len(after_array) != 0: 
            deftime=self.data[-1,:,1]-after_array[0,:,1]
            print("Insetrt時間軸のずれの補正",deftime)
            if abs(deftime[0]) > 1 :
                after_array[:,:,1]+=deftime
            self.data = np.vstack((self.data,after_array))
        self.maneuver_log(sys._getframe().f_code.co_name)

    #addKeyframeTrans の発展版。midtideのように、インプット画像よりもサイズを大きくして出力させる場合に使用する。将来的にはaddKeyframeTrans と統合した方が良さそう。
    def addWideKeyframeTrans(self,frame_nums,key_array,wide_scale=3,start_frame=[0,0],speed_round = False):
        maxwidth=self.width*wide_scale
        maxheight=self.height*wide_scale
        self.scan_nums = int(maxwidth) if self.scan_direction % 2 == 1 else int(maxheight)
        for keyframe in key_array:
            if len(self.data) != 0 : 
                #self.dataに連続させる場合
                extra_array = np.zeros((self.data.shape[0]+frame_nums,self.scan_nums,2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
                extra_array[0:self.data.shape[0]]=self.data
                outroFrameXP = self.data[-1,:,0]
                outroFrameZ = self.data[-1,:,1]
            else : 
                #新規で作成する場合
                extra_array = np.zeros((frame_nums,self.scan_nums,2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
                outroFrameXP = np.full(self.scan_nums,start_frame[0]) 
                outroFrameZ = np.arange(start_frame[1],start_frame[1]+self.scan_nums)
            end_line = keyframe[0]
            end_time = keyframe[1]
            endFrame = np.full(self.scan_nums,end_line) 
            normalFrame=np.arange(0,self.scan_nums)
            diff_array = endFrame - outroFrameXP
            diff_time = np.arange(end_time,end_time+self.scan_nums) - outroFrameZ
            if len(self.data) != 0 : 
                for i in range(frame_nums):
                    extra_array[self.data.shape[0]+i,:,0]=extra_array[self.data.shape[0]+i-1,:,0]+(diff_array/frame_nums) if speed_round == False else outroFrameXP + diff_array*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)
                    extra_array[self.data.shape[0]+i,:,1]=outroFrameZ+(diff_time/frame_nums)*i if speed_round == False else outroFrameZ+diff_time*(math.sin(math.radians(i/(frame_nums)*180-90))+1.0)/2
            else:
                for i in range(frame_nums):
                    extra_array[i,:,0]=outroFrameXP+(diff_array/frame_nums)*i if speed_round == False else outroFrameXP + diff_array*(math.sin(math.radians(i/(frame_nums)*180-90))+1.0)/2
                    extra_array[i,:,1]=outroFrameZ+(diff_time/frame_nums)*i if speed_round == False else outroFrameZ+diff_time*(math.sin(math.radians(i/(frame_nums)*180-90))+1.0)/2
            self.data=extra_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1])
                
    #midtideなどのインプット画像よりもスキャン方向サイズを大きくして出力させる場合にのみ使用。NPYファイルをLeft,center,Rightと3分割して出力させる。
    # 連結したNPYファイルも出力する。
    def addWaveTrans(self,frame_nums,cycle_degree,zdepth,flow=True,zslide=0,speed_round = True):
        #wr_arrayへの書き込み[Out画像のXorY,引用するin画像のXorY,in画像のz]
        wr_array=[]
        for i in range(0,frame_nums):
            crad = math.radians(cycle_degree)* i / (frame_nums-1) if speed_round == False else math.radians(cycle_degree)*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)
            ysin=math.sin(crad)
            zcos=math.cos(crad)
            adoptive_zdepth = zdepth * (math.cos(math.radians(360)* i / (frame_nums-1) + math.radians(180))+1.0)/ 2 #https://editor.p5js.org/ryufurusawa/sketches/uc-BTFcVl
            write_array = []
            if self.scan_direction%2 == 0:
                for y in range(0,int(self.height)):
                    pernum=math.cos(math.radians(y/self.height*cycle_degree+i/frame_nums*360))
                    yp = y/self.height*(self.height*0.945)*math.cos(math.radians(i/frame_nums*180))+i/frame_nums*self.height if flow==True else y
                    zp = zslide-(adoptive_zdepth*pernum*math.sin(math.radians(i/frame_nums*180)))
                    write_array.append([yp,zp])
            else:
                for x in range(0,int(self.width)):
                    pernum = math.cos(math.radians(x/self.width*cycle_degree+i/frame_nums*360)) #+i/frame_nums*360 でwaveを流す
                    xp = x/self.width*(self.width*0.945)*math.cos(math.radians(i/frame_nums*180))+i/frame_nums*self.width if flow == True else x
                    zp =zslide-(adoptive_zdepth*pernum*math.sin(math.radians(i/frame_nums*180)))
                    write_array.append([xp,zp])
            write_array = np.array(write_array)
            wr_array.append(write_array)
        wr_array = np.array(wr_array)
        if len(self.data)!=0: self.data = np.vstack((self.data,wr_array)) 
        else : self.data = wr_array
        if self.scan_direction==0:
            flowstr="yflow" if flow else "yfix"
        else:
            flowstr="xflow" if flow else "xfix"
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1].split("Trans")[0]+str(cycle_degree)+flowstr)
    '''
    縦、横の空間領域は変更せず、時間の変化だけを作る。画面の中心と周辺で時間の進行速度が変わる。
    zdepth=180中心が進行、周縁が逆行、徐々に戻る。360＝中心が進行、周縁が逆行、徐々に戻るり、逆転のち、徐々にノーマルに戻る
    '''
    def addEventHorizonTrans(self,frame_nums,zdepth,z_osc=1,cycle_degree=180,flow=False,zslide=0):
        print(sys._getframe().f_code.co_name)
        # #wr_arrayへの書き込み[Out画像のXorY,引用するin画像のXorY,in画像のZ]
        wr_array=[]
        for i in range(0,frame_nums):
            adoptive_zdepth = zdepth * (math.sin(math.pi * z_osc * i / (frame_nums-1))) 
            write_array = []
            if self.scan_direction%2 == 0:
                for y in range(0,int(self.height)):
                    pernum=math.sin(math.radians(y/self.height*cycle_degree))
                    yp = y/self.height*(self.height*0.945)*math.cos(math.radians(i/frame_nums*180))+i/frame_nums*self.height if flow==True else y
                    zp = zslide+(adoptive_zdepth*pernum)
                    write_array.append([yp,zp])
            else:
                for x in range(0,int(self.width)):
                    pernum = math.sin(math.radians(x/self.width*cycle_degree))
                    xp = x/self.width*(self.width*0.945)*math.cos(math.radians(i/frame_nums*180))+i/frame_nums*self.width if flow == True else x
                    zp =zslide+(adoptive_zdepth*pernum) - (adoptive_zdepth*math.sin(math.radians(cycle_degree*0.5)))/2
                    write_array.append([xp,zp])
            write_array = np.array(write_array)
            wr_array.append(write_array)
        wr_array = np.array(wr_array)
        if len(self.data)!=0: self.data = np.vstack((self.data,wr_array))
        else : self.data = wr_array
        
        self.maneuver_log(sys._getframe().f_code.co_name.split("add")[1].split("Trans")[0]+str(zdepth))

    # XYTの置換を補完的に遷移させる。画面の中心線を軸に、再生断面を回転させていく。spaceflow=Falseの設定で、空間は固定で、時間軸だけがゆがむ
    def addCycleTrans(self,frame_nums,cycle_degree=360,t_auto_scaling=False,zslide=0,extra_degree=0,speed_round = True,spaceflow=True,zscale=1):
        print(sys._getframe().f_code.co_name)
        zscale = zscale * self.xyt_boxel_scale
        wr_array=[]
        for i in range(0,frame_nums):
            crad = math.radians(extra_degree)+math.radians(cycle_degree)* i / (frame_nums-1) if speed_round == False else math.radians(extra_degree)+ math.radians(cycle_degree)*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)
            csin=math.sin(crad)
            ccos=math.cos(crad)
            write_array=[]
            if self.scan_direction%2 == 0:
                for y in range(0,int(self.height)):
                    pernum = (y-(self.height/2))/self.height
                    yp = self.height/2 + ccos*(self.height-1)*pernum if spaceflow else y
                    maxz = self.count  if t_auto_scaling == True else self.height * zscale
                    zp = zslide - csin*maxz*pernum
                    write_array.append([yp,zp])
            else:
                for x in range(0,int(self.width)):
                    pernum=(x-(self.width/2))/self.width
                    maxz = self.count *0.9 if t_auto_scaling == True else self.width  * zscale
                    xp = self.width/2 + ccos*(self.width-1)*pernum if spaceflow else x
                    zp = zslide -csin*maxz*pernum
                    write_array.append([xp,zp])
            write_array = np.array(write_array)
            wr_array.append(write_array)
        wr_array = np.array(wr_array)
        if len(self.data) != 0: self.data = np.vstack((self.data,wr_array))
        else : self.data = wr_array
        spf_attr="" if spaceflow else "-spacefix"
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1].split("Trans")[0]+str(cycle_degree)+"deg-zscale:"+str(maxz/self.scan_nums)+"-"+str(frame_nums)+"f"+spf_attr)

    # 回転の中心軸を漸次的に変化させる
    #extradegree=90,180,270などの設定の時に、初期値（i=0）が数値が飛ぶ。i=0のときに、うまくshift_num=0で始まってi=1のときにshift_num=1にするようにしたいが、いろいろ調整が難しそう。現状90.1あるいは89などの数値をextradegreeにはめてあげると解決策になる
    def addCustomCycleTrans(self,frame_nums,cycle_degree,start_center=1/2,end_center=1/2,t_auto_scaling=False,extra_degree=0,speed_round = True,zslide=0,auto_zslide=True,t_auto_scaling_num=0.9,zscale=1,spaceflow=True):
        print(sys._getframe().f_code.co_name)
        zscale = zscale * self.xyt_boxel_scale
        permit_auto_zslide=False
        if zslide == 0:
            if auto_zslide and len(self.data)>0:
                permit_auto_zslide=True
        else :
            zslide= zslide * self.xyt_boxel_scale
        wr_array=[]
        defcenter=end_center-start_center
        # csinarray=[]
        # ccosarray=[]

        if t_auto_scaling:
            if len(self.data)>0 :
                maxz =(self.count-self.data.shape[0]-(np.max(self.data[:,:,1])-np.min(self.data[:,:,1]))-frame_nums) * t_auto_scaling_num
            else:
                maxz =(self.count-frame_nums) * t_auto_scaling_num
        else:
            maxz = self.height * zscale if self.scan_direction == 0 else self.width * zscale
        
        shift_num=int(extra_degree/90)#反転する回数を記録
        if len(self.cycle_axis) : self.cycle_axis=[]
        
        for i in range(0,frame_nums):
            crad = math.radians(extra_degree)+math.radians(cycle_degree)* i / (frame_nums-1) if speed_round == False else math.radians(extra_degree)+math.radians(cycle_degree)*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)    
            if i > 1 and ((math.cos(crad) < 0 and ccos > 0) or (math.cos(crad) > 0 and ccos < 0)):#反転するタイミング
                if cycle_degree > 0:                    
                    shift_num+=1
                else :
                    shift_num-=1
            csin=math.sin(crad)
            ccos=math.cos(crad)
            # print(i, csin,ccos,shift_num)
            write_array=[]
            # csinarray.append(csin)
            # ccosarray.append(ccos)
         
            if self.scan_direction == 0:
                ycenter =(start_center+i*(defcenter/frame_nums))*(self.height-1)
                for y in range(0,int(self.height)):
                    pernum=(y-(ycenter))/self.height if ccos > 0 else (y-(self.height-ycenter))/self.height #yのズレのための計算,ccosが90度を超えるとはみ出てしまう問題の解消
                    yp = ycenter + ccos * (self.height-1) * pernum if spaceflow else y
                    if permit_auto_zslide : zslide=self.data[-1,y,1]
                    zp=zslide -csin*maxz*pernum - ((y-ycenter)+((self.height-y)-ycenter))*shift_num * maxz/self.height #反転するごとにZのズレを補正していく
                    # if shift_num>0:print(((y-ycenter)+((self.height-y)-ycenter))*shift_num )
                    yp = sorted([0, yp, self.height-1])[1]
                    write_array.append([yp,zp])
                    # if y == 0 : print(i,y,zp,maxz/self.height,shift_num,self.height/2-ycenter,((y-ycenter)+((self.height-y)-ycenter)))
            else:
                xcenter =(start_center+i*(defcenter/frame_nums))*(self.width-1)
                for x in range(0,int(self.width)):
                    pernum=(x-(xcenter))/self.width if ccos > 0 else  (x-(self.width-xcenter))/self.width #xのズレのための計算,ccosが90度を超えるとはみ出てしまう問題の解消
                    xp = xcenter + ccos * (self.width-1) * pernum  if spaceflow else x
                    if permit_auto_zslide : zslide=self.data[-1,x,1]
                    zp = zslide-csin*maxz*pernum-((x-xcenter)+((self.width-x)-xcenter))* shift_num * maxz/self.width #反転するごとにZのズレを補正していく
                    # if shift_num>0:print(((x-xcenter)+((self.width-x)-xcenter))*shift_num )
                    xp = sorted([0, xp, self.width-1])[1]
                    write_array.append([xp,zp])
                    # if x == 0 : 
                        # print(i,(x-(xcenter))/self.width,pernum,ccos,xcenter,(x-(self.width-xcenter))/self.width)
                        # print(i,x,zp,maxz/self.width,shift_num,zslide,pernum,self.width/2-xcenter,((x-xcenter)+((self.width-x)-xcenter)))
                        # print(i,zp,shift_num,zslide,pernum,(ccos > 0), ccos,zslide-csin*maxz*pernum,((x-xcenter)+((self.width-x)-xcenter))* shift_num * maxz/self.width)
            # print(write_array)
            # print(np.array(write_array).shape)
            write_array = np.array(write_array)
            wr_array.append(write_array)
            if self.scan_direction == 1:
                self.cycle_axis.append(xcenter)
            else : 
                self.cycle_axis.append(ycenter)
        wr_array = np.array(wr_array)
        # plt.plot(csinarray)
        # plt.plot(ccosarray)
        # plt.plot(wr_array[:,0,1]/np.amax(wr_array[:,0,1]))
        if t_auto_scaling :
            if len(self.data) != 0: 
                if permit_auto_zslide and extra_degree > 0:
                    zslidearray=self.data[-1,:,1]-wr_array[0,:,1]
                    print("auto_zslide:",self.data[-1,0,1],wr_array[0,0,1],zslidearray[0])
                    wr_array[:,:,1]+=zslidearray
                    print("auto_zslide調整後のwr_array:",wr_array[0,0,1])
                if np.max(np.vstack((self.data,wr_array)))-np.min(np.vstack((self.data,wr_array))) > self.count:
                    diff_v=(np.max(np.vstack((self.data,wr_array)))-np.min(np.vstack((self.data,wr_array))))-self.count
                    print("映像データの長さを"+str(diff_v)+"frame超えています。再計算します。",t_auto_scaling_num*0.9,maxz)
                    self.addCustomCycleTrans(frame_nums,cycle_degree,start_center,end_center,t_auto_scaling,extra_degree,speed_round,0,auto_zslide,t_auto_scaling_num=t_auto_scaling_num*0.9)
                    return 
                print("frame数に問題がないので、self.dataに接続する",permit_auto_zslide)
                self.data = np.vstack((self.data,wr_array))
            else :  
                if np.max(wr_array)-np.min(wr_array) > self.count:
                    diff_v=(np.max(wr_array)-np.min(wr_array))-self.count
                    print("映像データの長さを"+str(diff_v)+"frame超えています。再計算します。",t_auto_scaling_num*0.9,maxz)
                    self.addCustomCycleTrans(frame_nums,cycle_degree,start_center,end_center,t_auto_scaling,extra_degree,speed_round,0,auto_zslide,t_auto_scaling_num=t_auto_scaling_num*0.9)
                    return
                else : 
                    self.data = wr_array
        else :
            if len(self.data) != 0: 
                if permit_auto_zslide and extra_degree > 0:
                    zslidearray=self.data[-1,:,1]-wr_array[0,:,1]
                    print("auto_zslide:",self.data[-1,0,1],wr_array[0,0,1],zslidearray[0])
                    wr_array[:,:,1]+=zslidearray
                self.data = np.vstack((self.data,wr_array))
            else : self.data = wr_array
        
        spf_attr="" if spaceflow else "-spacefix"
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1].split("Trans")[0]+str(cycle_degree)+"-zscale"+str(t_auto_scaling)+"_"+str(start_center)+"->"+str(end_center)+spf_attr)
    #中心外を、エッジのスリットに時間差をつけて表示する。回転の中心軸を漸次的に変化させる
    def addWideCustomCycleTrans(self,frame_nums,cycle_degree,start_center,end_center,maxz_range=None,wide_scale=3,t_auto_scaling=False,extra_degree=0,speed_round = True):
        print(sys._getframe().f_code.co_name)
        wr_array=[]
        defcenter=end_center-start_center
        maxwidth=self.width*wide_scale
        ccos_array=[]
        csin_array=[]
        zscale_array=[]
        zp_array=[]
        if maxz_range==None : 
            maxz_range = self.count
        
        shift_num=int(extra_degree/90)#反転する回数を記録
        for i in range(0,frame_nums):
            crad = math.radians(extra_degree)+math.radians(cycle_degree)* i / (frame_nums-1) if speed_round == False else math.radians(extra_degree)+math.radians(cycle_degree)*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)
            if i > 1 and ((math.cos(crad) < 0 and ccos > 0) or (math.cos(crad) > 0 and ccos < 0)):#反転するタイミング
                shift_num+=1
            csin=math.sin(crad)
            ccos=math.cos(crad)
            ccos_array.append(ccos)
            csin_array.append(csin)
            write_array=[]
            if self.scan_direction == 0:
                for y in range(0,int(self.height)):
                    pernum = (y-(self.height/2))/self.height
                    yp = self.height/2 + csin*(self.height-1)*pernum
                    maxz =maxz_range if t_auto_scaling == True else self.height
                    zp = maxz/2 - ccos*maxz*pernum
                    write_array.append([yp,zp])
            else:
                widew=abs(self.width/ccos)
                widez=abs(self.width*math.tan(crad))
                if widew>maxwidth :
                    widew=maxwidth
                    widez = widew*csin
                xcenterw =(start_center+i*(defcenter/frame_nums)) * widew
                xcenter =(start_center+i*(defcenter/frame_nums)) * self.width
                # wratio=widew/self.width
                # print(xcenter,widew,widez)
                for x in range(0,int(maxwidth)):
                    if x > (maxwidth-widew)/2 and x < maxwidth-(maxwidth-widew)/2:
                        # pernum=((x-(maxwidth-widew)/2)-(widew/2))/widew
                        pernum=((x-(maxwidth-widew)/2)-xcenterw)/widew if ccos > 0 else ((x-(maxwidth-widew)/2)-(widew-xcenterw))/widew
                        # xp = self.width/2 + ccos*(self.width-1)*pernum
                        # xp = xcenter + ccos * (self.width-1) * pernum if ccos > 0 else ((self.width-1)-xcenter) + ccos*(self.width-1)*pernum 
                        xp = xcenter + ccos * (self.width-1) * pernum 
                        # xp = sorted([0, xp, self.width-1])[1]
                        pernum=(x-((maxwidth-widew)/2+xcenterw))/maxwidth
                        maxz = maxz_range if t_auto_scaling == True else abs(widez)
                        zp = -csin*maxz*pernum
                        # zp = -csin*maxz*pernum - (((x-(maxwidth-widew)/2)-xcenterw)+((widew-(x-(maxwidth-widew)/2))-xcenterw))*shift_num #反転するごとにZのズレを補正していく
                        # zp = -csin*maxz*pernum - (((x-(maxwidth-widew)/2)/wratio-xcenter)+((self.width-(x-(maxwidth-widew)/2)/wratio)-xcenter))*shift_num #反転するごとにZのズレを補正していく
                        # if x== (maxwidth-widew)/2 + 1 :
                            # print(i,zp,csin,pernum,shift_num,zp)
                        zscale_array.append(maxz/widew)
                    else :
                        if  x < (maxwidth-widew)/2+1 :
                            # pernum=(0-(widew/2))/widew
                            pernum=(0-(xcenterw))/widew  if ccos > 0 else (0-(widew-xcenterw))/widew 
                        else:
                            # pernum=(widew-(widew/2))/widew
                            pernum=(widew-(xcenterw))/widew if ccos > 0 else (widew-(widew-xcenterw))/widew 
                        # xp =  xcenter + ccos * (self.width-1) * pernum if ccos > 0 else ((self.width-1)-xcenter) + ccos*(self.width-1)*pernum 
                        xp =  xcenter + ccos * (self.width-1) * pernum 
                        # pernum=(x-(maxwidth/2))/maxwidth
                        pernum=((x-(maxwidth-widew)/2)-xcenterw)/maxwidth 
                        maxz = maxz_range if t_auto_scaling == True else abs(widez)
                        zp = -csin*maxz*pernum
                        # zp = -csin*maxz*pernum - (((x-(maxwidth-widew)/2)/wratio-xcenter)+((self.width-(x-(maxwidth-widew)/2)/wratio)-xcenter))*shift_num #反転するごとにZのズレを補正していく
                        # zp = -csin*maxz*pernum - (((x-(maxwidth-widew)/2)-xcenterw)+((widew-(x-(maxwidth-widew)/2))-xcenterw))*shift_num #反転するごとにZのズレを補正していく
                    # if x == int(maxwidth/4): print(i,zp,widez)
                    xp = sorted([0, xp, self.width-1])[1]
                    write_array.append([xp,zp])
            write_array = np.array(write_array)
            wr_array.append(write_array)
        wr_array = np.array(wr_array)
        if len(self.data) != 0: 
            deftime=self.data[-1,int(start_center*(self.width-1)),1]-wr_array[0,int(start_center*(self.width-1)),1]
            print("時間軸のずれ",deftime,2)
            if abs(deftime) > 1 :
                wr_array[:,:,1]+=deftime
            self.data = np.vstack((self.data,wr_array))
        else : self.data = wr_array
        
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1].split("Trans")[0]+str(cycle_degree)+"-zscale"+str(t_auto_scaling)+"_"+str(start_center)+"->"+str(end_center))
    #横に拡大。
    def addFixWideCycleTrans(self,frame_nums,cycle_degree,wide_scale=3,t_auto_scaling=True,extra_degree=0,speed_round = True):
        wr_array=[]
        widewidth=int(self.width*wide_scale)
        for i in range(0,frame_nums):
            crad = math.radians(90+extra_degree)+math.radians(cycle_degree)* i / (frame_nums-1) if speed_round == False else math.radians(90+extra_degree)+math.radians(cycle_degree)*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)
            ysin=math.sin(crad)
            zcos=math.cos(crad)
            write_array=[]
            if self.scan_direction == 0:
                for y in range(0,int(self.height)):
                    pernum = (y-(self.height/2))/self.height
                    yp = self.height/2 + ysin*(self.height-1)*pernum
                    maxz = self.count  if t_auto_scaling == True else self.height
                    zp = maxz/2 - zcos*maxz*pernum
                    write_array.append([yp,zp])
            else:
                for x in range(0,widewidth):
                    pernum=(x-(widewidth/2))/(widewidth)
                    maxz = self.count  if t_auto_scaling == True else self.width
                    xp = self.width/2 + ysin*(self.width-1)*pernum
                    zp = maxz/2 - zcos*maxz*pernum
                    write_array.append([xp,zp])
            write_array = np.array(write_array)
            wr_array.append(write_array)
        wr_array = np.array(wr_array)
        if len(self.data) != 0: self.data = np.vstack((self.data,wr_array))
        else: self.data=wr_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1].split("Trans")[0]+str(frame_nums))

    '''
    時間軸の解像度を徐々に変化させる試みがなされている。基本はTransと同じ。
    時間軸の解像度を徐々に変化させる試みがなされている。基本はXYT Transと同じ
    blowupの動きをキーフレームにより詳細に制御するには、関数内の以下のパラメーターを調整すること。
        timevalues:読み込む時間軸方向のピクセル数 (時間幅)
        timepoints=書き出す総フレームに対する時間軸のキーフレーム （0~1）
    wave_type=0:サイン波
    wave_type=1:三角波      
    connect_round=キーフレーをリニアかイーズインアウトするか？　オプションで、配列で送った場合に、かくキーポイントの保管方法を設定できる。
    '''
    def addBlowupTrans(self,frame_nums,deg=360,speed_round = True,connect_round=[],timevalues=[],timepoints=[],timecenter=[],extra_degree=0,wave_type=0,zslide=0,z_autofit=False):
        #2021.09.02　New。
        #2022.09.19 プロセスをcycletransと同等に
        """
        blowupの動きをキーフレームにより制御する
        timevalues:読み込む時間軸方向のピクセル数(px)
        timepoints=書き出す総フレームに対する時間軸のキーフレーム(0~1)
        """
        if len(timevalues)==0:
            timevalues=[self.count,self.scan_nums,1,0]#左端から右端までの時間差（Frame）
            timepoints=[0,0.7,0.95,1]
            if len(timecenter) != len(timevalues): timecenter=[0.5] * len(timevalues)
        else : 
            if timevalues[0] > self.count : timevalues[0] = self.count
            #もし、timepoints が設定されていなければ、自動的に入力する
            if len(timevalues)!=len(timevalues):
                for i in range(0,len(timevalues)):timepoints.append(1/(len(timevalues)-1)*i)
            if len(timecenter) == 0:
                timecenter=np.full(len(timevalues), 0.5)
            if len(connect_round) != len(timevalues):
                if connect_round==[]:
                    connect_round= np.full(len(timevalues),1)
                else:
                    connect_round= np.full(len(timevalues),connect_round)

        wr_array=[]
        print("timevalues:",timevalues,"timepoints:",timepoints)
        firstzrange=np.max(timevalues)
        # pre_front_point = int((self.count-firstzrange)/2)
        pre_front_point = (self.count-firstzrange)/2
        prezrange=firstzrange
        for i in range(0,frame_nums):
            # 角度をラジアンに変換
            crad = math.radians(extra_degree) + math.radians(deg) * i / (frame_nums-1) if speed_round == False else  math.radians(extra_degree) + math.radians(deg)*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)
            if wave_type == 0 :
                zcos = math.cos(crad)
            else:
                # 三角波の計算
                # ここで 'phase' は三角波の周期を制御するために使われる
                phase = crad / (2 * math.pi)
                phase = phase - math.floor(phase)
                # 周期を0から1の範囲に制限
                triangular_wave_value = 1 - abs(1 - 2 * phase)
                # 三角波の値を-1から1の範囲に調整
                triangular_wave_value = 2 * (triangular_wave_value - 0.5)

            write_array = []
            fn = search(i,frame_nums,timepoints)#timepointsのうちの何番目の領域の計算か？
            gaptime = timevalues[fn]-timevalues[fn+1]
            # ステップiの進行度（0から1までの範囲）
            ts = (i-frame_nums*timepoints[fn])/(frame_nums*(timepoints[fn+1]-timepoints[fn])) 
            # 細かい動きをつける場合。リニアかノンリニアか変数で選択
            # ts = ts if connect_round == 0 else ((math.sin(math.radians(ts*180-90)))/2+0.5)
            ts = ts if connect_round[fn] == 0 else ((math.sin(math.radians(ts*180-90)))/2+0.5)
            
            # print(fn,i,ts)
            if self.scan_direction == 1 : 
                ajstlen=timevalues[fn+1]/self.width+(gaptime/self.width)-(gaptime/self.width)*ts
                nowzrange=ajstlen*self.scan_nums
                centerdirection= timecenter[fn] * (1 - ts) +  timecenter[fn+1] * ts  # 線形補完
            else : 
                ajstlen=timevalues[fn+1]/self.height+(gaptime/self.height)-(gaptime/self.height)*ts
                nowzrange=ajstlen*self.scan_nums
                centerdirection= timecenter[fn] * (1 - ts) +  timecenter[fn+1] * ts  # 線形補完
            diffzrange = prezrange-nowzrange
            # pre_front_point += int(diffzrange*centerdirection)
            pre_front_point += diffzrange*centerdirection
            prezrange = nowzrange
            if self.scan_direction == 1:
                xp = (self.width-1)-(zcos*(self.width-1)/2+(self.width-1)/2) if wave_type == 0 else  (triangular_wave_value * (self.width - 1) / 2 + (self.width - 1) / 2)
                for x in range(0,int(self.width)):#Xは固定でzポイントが飛び飛びから詰まっていく。
                    zp=pre_front_point+x*ajstlen
                    if z_autofit :
                        if x==0 and i==0 and len(self.data) != 0:
                            zslide=self.data[-1,0,1]-zp
                    write_array.append([xp,zp+zslide])
            else:
                yp = (self.height-1)-(zcos*(self.height-1)/2+(self.height-1)/2) if wave_type == 0 else  (triangular_wave_value * (self.height - 1) / 2 + (self.height - 1) / 2)
                for y in range(0,int(self.height)):#Xは固定でzポイントが飛び飛びから詰まっていく。
                    zp=pre_front_point+y*ajstlen
                    if z_autofit :
                        if y==0 and i==0 and len(self.data) != 0:
                            zslide=self.data[-1,0,1]-zp
                    write_array.append([yp,zp+zslide])
            wr_array.append(write_array)
        if len(self.data) != 0: self.data = np.vstack((self.data,np.array(wr_array)))
        else: self.data=np.array(wr_array)
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1].split("Trans")[0]+str(frame_nums)+"-deg"+str(deg))
        return np.array(frame_nums * np.array(timepoints),dtype='int')
    

