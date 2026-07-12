"""drawManeuver のデータ操作 (DataOpsMixin)

self.data や派生配列 (sc_resetPositionMap 等) への基本的な操作:
- append / prepend / wide_expandB / arrayReflection / arrayExtract
- interpolation_append / interpolation_append_byspeed / vsizeReturn
- spline_interpolate
- zCenterArange / zArange / zStartArange
- dataCheck / zPointCheck / zPointCheckandReflect
- applyTimeSlide / applyInOutGapFix
- applyInFix / applyOutFix / applyInPartFix / applyOutPartFix / applyOutPartFixB
- img_to_maneuver / img_to_maneuver_rate_based
- data_save
- split_3_npySave / split_3_npysavereturn
"""
import os
import sys
import gc
import math
import cv2
import numpy as np
from scipy.interpolate import splrep, splev, interp1d
import easing

from ._utils import (
    custom_blur,
    custom_onedimention_blur,
    onedimention_LoopBlur,
    bezier_interpolation,
)


class DataOpsMixin:
    def append(self,maneuver,auto_zslide=True,zslide=0):
        print(sys._getframe().f_code.co_name)
        permit_auto_zslide=False
        if zslide == 0:
            if auto_zslide and len(self.data)>0:
                permit_auto_zslide=True
        if permit_auto_zslide : 
            zslide = maneuver[0,:,1]-self.data[-1,:,1]
            print("append gap=",zslide)
        maneuver[:,:,1]=maneuver[:,:,1]-zslide
        if maneuver.shape[0] > 0:
            self.data = np.vstack((self.data,maneuver))
            self.maneuver_log(sys._getframe().f_code.co_name)
        else:
            print("no data")

    #added 2025 6/20
    def prepend(self, maneuver):
        print(sys._getframe().f_code.co_name)
        if maneuver.shape[0] > 0:
            self.data = np.vstack((maneuver, self.data))
            self.maneuver_log(sys._getframe().f_code.co_name)
        else:
            print("no data")


    #added 2025 5/19
    def wide_expandB(self,add_size=3840,sclip=True,zclip=True,spacedirection=True,z_offset=0):
        """
        z_offset: 1ステップあたりの時間差分オフセット（デフォルト0）
            右方向へ1スリット進むごとに +z_offset、左方向へ1スリット進むごとに -z_offset を加算。
            実際の時間差分(lz_diff/rz_diff)が0のときでもグラデーションを付与できる。
            例: z_offset=1 → 右端が未来、左端が過去（右→左へ流れる車窓風景に対応）
        """
        print(sys._getframe().f_code.co_name)
        sp_dir_v = 1 if spacedirection else -1
        # 2次元目にパディングを適用（左右に均等にパディングを追加）
        new_array = np.pad(self.data, ((0, 0), (add_size, add_size), (0, 0)), mode='constant')
        for i in range(self.data.shape[0]):
            ls_diff = self.data[i,0,0]-self.data[i,1,0]#左側の空間変化
            lz_diff = self.data[i,0,1]-self.data[i,1,1]#左側の時間変化
            rs_diff = self.data[i,-1,0]-self.data[i,-2,0]
            rz_diff = self.data[i,-1,1]-self.data[i,-2,1]
            # print(ls_diff,lz_diff,rs_diff,rz_diff)
            for k in range(add_size):
                #左側の処理（中心から離れるごとに -z_offset）
                if not (0 <= new_array[i, add_size - k, 0] + ls_diff <= self.scan_nums):
                    new_array[i,add_size-k-1,0]=new_array[i,add_size-k,0]
                    new_array[i,add_size-k-1,1]=new_array[i,add_size-k,1]+lz_diff+ls_diff*self.xyt_boxel_scale * sp_dir_v - z_offset
                else :
                    new_array[i,add_size-k-1,0]=new_array[i,add_size-k,0]+ls_diff
                    new_array[i,add_size-k-1,1]=new_array[i,add_size-k,1]+lz_diff - z_offset
                #右側の処理（中心から離れるごとに +z_offset）
                if not (0 <= new_array[i,add_size+self.data.shape[1]+k-1,0]+rs_diff <= self.scan_nums):
                    new_array[i,add_size+self.data.shape[1]+k,0]=new_array[i,add_size+self.data.shape[1]+k-1,0]
                    new_array[i,add_size+self.data.shape[1]+k,1]=new_array[i,add_size+self.data.shape[1]+k-1,1]+rz_diff+rs_diff*self.xyt_boxel_scale*sp_dir_v + z_offset
                else :
                    new_array[i,add_size+self.data.shape[1]+k,0]=new_array[i,add_size+self.data.shape[1]+k-1,0]+rs_diff
                    new_array[i,add_size+self.data.shape[1]+k,1]=new_array[i,add_size+self.data.shape[1]+k-1,1]+rz_diff + z_offset
                # print(new_array[i,add_size-k,1],new_array[i,add_size+self.data.shape[1]+k,1])
        if sclip : new_array[:,:,0] = np.clip(new_array[:,:,0],0,self.scan_nums-1)
        if zclip : new_array[:,:,1] = np.clip(new_array[:,:,1],1,self.count-1)
        self.data = new_array
        self.maneuver_log(sys._getframe().f_code.co_name)

    #added 2023 11/24 DATaの鏡面反転
    def arrayReflection(self):
        print(sys._getframe().f_code.co_name)
        self.data= np.vstack((self.data,self.data[::-1,:,:]))
        self.maneuver_log(sys._getframe().f_code.co_name)    
    
    #added 2023 11/13 
    '''example
    array=bm.data
    bm.append(array)
    bm.append(array)
    bm.applyTimeBlur(100)
    bm.applySpaceBlur(100)
    bm.arrayExtract(array.shape[0],array.shape[0]+array.shape[0])
    '''
    def arrayExtract(self,start,end):
        print(sys._getframe().f_code.co_name)
        self.data = self.data[start:end,:,:]
        self.maneuver_log(sys._getframe().f_code.co_name)

    #added 2023 11/24 
    def interpolation_append(self,maneuver,connection_num,speed_round=False,add_maneuver=True):
        zslide = maneuver[0,:,1]-self.data[-1,:,1]
        sslide = maneuver[0,:,0]-self.data[-1,:,0]
        # conection_array = np.zeros([connection_num,self.scan_nums,2],dtype=np.float64)
        conection_array = np.zeros([connection_num,self.data.shape[1],2],dtype=np.float64)
        for i in range(connection_num):
            conection_array[i,:,1] = self.data[-1,:,1]+(zslide/connection_num)*i if (speed_round == False or connection_num==1) else self.data[-1,:,1]+zslide*(1-(math.cos(math.radians((i/(connection_num-1))*180))+1.0)/2)
            conection_array[i,:,0] = self.data[-1,:,0]+(sslide/connection_num)*i if (speed_round == False or connection_num==1) else self.data[-1,:,0]+sslide*(1-(math.cos(math.radians((i/(connection_num-1))*180))+1.0)/2)  
        self.data = np.vstack((self.data,conection_array))
        if add_maneuver : self.data = np.vstack((self.data,maneuver))
        self.maneuver_log(sys._getframe().f_code.co_name)

    #指定したスピードで保管する。追加されるフレーム数で調整する。
    def interpolation_append_byspeed(self,maneuver,frame_speed,speed_round=False,add_maneuver=True,sblur=True,tblur=True,blur_range=None):
        print(sys._getframe().f_code.co_name)
        zslide = maneuver[0,:,1]-self.data[-1,:,1]
        sslide = maneuver[0,:,0]-self.data[-1,:,0]
        # conection_array = np.zeros([connection_num,self.scan_nums,2],dtype=np.float64)
        connection_num = abs(int(np.mean(zslide/frame_speed)))
        conection_array = np.zeros([connection_num,self.data.shape[1],2],dtype=np.float64)
        for i in range(connection_num):
            conection_array[i,:,1] = self.data[-1,:,1]+(zslide/connection_num)*i if (speed_round == False or connection_num==1) else self.data[-1,:,1]+zslide*(1-(math.cos(math.radians((i/(connection_num-1))*180))+1.0)/2)
            conection_array[i,:,0] = self.data[-1,:,0]+(sslide/connection_num)*i if (speed_round == False or connection_num==1) else self.data[-1,:,0]+sslide*(1-(math.cos(math.radians((i/(connection_num-1))*180))+1.0)/2)  
        frame_sum = self.data.shape[0]
        connect_frame_sum = conection_array.shape[0]
        maneuver_frame_sum= maneuver.shape[0]
        if blur_range == None:
            blur_range = (maneuver_frame_sum + frame_sum) // 6
            #blur_range = connect_frame_sum // 2 if frame_sum > connect_frame_sum*2 else frame_sum // 4
        self.data = np.vstack((self.data,conection_array))
        if add_maneuver : self.data = np.vstack((self.data,maneuver))
        if tblur :
            self.data = custom_blur(self.data,s_frame=frame_sum-blur_range,e_frame=frame_sum+connect_frame_sum+blur_range,bl_time=blur_range,dim_num=1)
        if sblur :
            self.data = custom_blur(self.data,s_frame=frame_sum-blur_range,e_frame=frame_sum+connect_frame_sum+blur_range,bl_time=blur_range,dim_num=0)
        
        self.maneuver_log(sys._getframe().f_code.co_name)
    


    # グローバル関数"img_size_type"に応じて、出力するイメージのサイズ（縦、横）を返す。
    def vsizeReturn(self):
        if self.img_size_type == 0 :
            vs= (int(self.width),int(self.height))
        elif self.img_size_type == 1 :
            vs= (self.slit_length,self.slit_length*2) if self.scan_direction == 0 else (self.slit_length *2,self.slit_length)#fullHD縦映像＞４kサイズにする
        elif self.img_size_type == 2 :
            vs = (int(self.width),int(self.count))if self.scan_direction == 0 else (int(self.count),int(self.height))
        elif self.img_size_type == 3 :
            vs = (self.slit_length,self.slit_length)
        return vs
    
    def spline_interpolate(self,x, keyframes, method='spline'):
        # キーフレームの要素数
        num_keyframes = len(keyframes)

        # キーフレームのx座標を計算
        x_positions = [i * self.scan_nums / (num_keyframes - 1) for i in range(num_keyframes)]

        # 中間の制御点を追加
        control_x = [(x_positions[i] + x_positions[i+1]) / 2 for i in range(num_keyframes-1)]
        linear_interp = interp1d(x_positions, keyframes)
        control_y = linear_interp(control_x)

        x_positions_controlled = []
        keyframes_controlled = []
        for i in range(num_keyframes-1):
            x_positions_controlled.extend([x_positions[i], control_x[i]])
            keyframes_controlled.extend([keyframes[i], control_y[i]])
        x_positions_controlled.append(x_positions[-1])
        keyframes_controlled.append(keyframes[-1])

        if method == 'spline':
            # スプライン次数の調整
            k = min(3, len(keyframes_controlled) - 1) 

            # スプライン補完のための曲線を計算
            tck = splrep(x_positions_controlled, keyframes_controlled, k=k)
            return splev(x, tck)
            
        elif method == 'linear':
            return linear_interp(x)
        elif method == 'bezier':
            results = []
            for xi in x:
                for i in range(num_keyframes - 1):
                    if x_positions[i] <= xi <= x_positions[i+1]:
                        t = (xi - x_positions[i]) / (x_positions[i+1] - x_positions[i])
                        control_x = (x_positions[i] + x_positions[i+1]) / 2
                        control_y = (keyframes[i] + keyframes[i+1]) / 2
                        result = bezier_interpolation(keyframes[i], control_y, keyframes[i+1], t)
                        results.append(result)
            return results


    #軌道配列と入力映像のフレーム数を照あわせて、入力映像の時間的な意味での中心フレームに寄せる。
    def zCenterArange(self, center_time_frame=None):
        print(sys._getframe().f_code.co_name)
        nan_count = np.sum(np.isnan(self.data[:,:,1]))
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values detected in z-data (time dimension)")
        z_min = np.nanmin(self.data[:,:,1])
        z_max = np.nanmax(self.data[:,:,1])
        if np.isnan(z_min) or np.isnan(z_max):
            print("Warning: all z-values are NaN, zCenterArange skipped")
            return
        target_frame = center_time_frame if center_time_frame is not None else self.count / 2
        shift = int(target_frame - (z_max + z_min) / 2)
        if shift == 0:
            return
        self.data[:,:,1] += shift
        self.maneuver_log("zCenterAranged")
        
    def zArange(self, target_frame, center_time_frame=None):
        """target_frame のフレームの z 中央値が center_time_frame に来るよう全体をスライド。

        zCenterArange() がデータ全体の z min/max 中央を基準にするのに対し、
        こちらは指定フレームの z 値を基準にスライドする。
        target_frame: self.data 内のフレームインデックス
        center_time_frame: 目標の時間座標。None なら self.count/2 (入力映像の中央)。
        """
        print(sys._getframe().f_code.co_name, f"target_frame={target_frame}")
        if target_frame < 0 or target_frame >= len(self.data):
            print(f"Warning: target_frame={target_frame} is out of range [0, {len(self.data)-1}]")
            return
        target_z = np.nanmean(self.data[target_frame, :, 1])
        if np.isnan(target_z):
            print("Warning: target_frame z-values are all NaN, zArange skipped")
            return
        dest = center_time_frame if center_time_frame is not None else self.count / 2
        shift = int(dest - target_z)
        if shift == 0:
            return
        self.data[:, :, 1] += shift
        self.maneuver_log(f"zAranged_f{target_frame}")

        #軌道配列と入力映像のフレーム数を照あわせて、入力映像の時間的な意味での０に最小の時間位置を持つフレームに合わせる
    def zStartArange(self):
        print(sys._getframe().f_code.co_name)
        idiff = np.amin(self.data[:,:,1])
        if idiff > 0 :
            self.data[:,:,1] -= idiff #0基準に戻す
            # self.data[:,:,1] += int((idiff+ediff)/2)
            self.maneuver_log("zStartAranged")
        else : 
            return
    
    def dataCheck(self):
        print(sys._getframe().f_code.co_name)
        print("data.shape = ",self.data.shape)
        print("Reference_Space(data[:,:,0]) min-max = ",np.amin(self.data[:,:,0]),np.amax(self.data[:,:,0]))
        print("Reference_Time(data[:,:,1]) min-max = ",np.amin(self.data[:,:,1]),np.amax(self.data[:,:,1]))

    #軌道配列の中で、時間座標が負の値になっていないかチェックして、負の値になっている場合、最小値を０になるように全体に対してスライドさせて調節する。
    def zPointCheck(self,subtract_count=0):    
        print(sys._getframe().f_code.co_name)
        self.dataCheck()
        if np.amin(self.data[:,:,1]) < 0  or np.amax(self.data[:,:,1]) > (self.count-subtract_count) :
            add_attr=""
            if np.amin(self.data[:,:,1]) < 0 :
                add_frame=np.amin(self.data[:,:,1])*-1
                self.data[:,:,1]+=add_frame
                print("zp 0以下、range-調整後:",np.amin(self.data[:,:,1]),np.amax(self.data[:,:,1]))
                add_attr+="_timeSlide"+str(int(add_frame))+"f"
            if np.amax(self.data[:,:,1]) > (self.count-subtract_count):
                diffframe=np.amax(self.data[:,:,1])-(self.count-subtract_count)
                if diffframe < np.amin(self.data[:,:,1]):
                    add_frame = diffframe*-1
                    self.data[:,:,1] += add_frame
                    print("zp footageMaxcount",self.count-subtract_count,"overed, range-調整後:",np.amin(self.data[:,:,1]),np.amax(self.data[:,:,1]))
                    add_attr+="_timeSlide"+str(int(add_frame))+"f"
                else:
                    self.data[:,:,1] += np.amin(self.data[:,:,1])*-1
                    scale_rate=(self.count-subtract_count) /np.amax(self.data[:,:,1])
                    self.data[:,:,1]=self.data[:,:,1]*scale_rate
                    add_attr+="_timeScaling"+str(scale_rate)
                    print("zp range-Scaling調整後:",np.amin(self.data[:,:,1]),np.amax(self.data[:,:,1]))
            self.maneuver_log(add_attr)
        else : print("check ok!!!")


    # #軌道配列の中で、時間座標が負の値になっていないかチェックして、負の値になっている場合、リフレクトする。。
    def zPointCheckandReflect(self,subtract_count=0):    
        print(sys._getframe().f_code.co_name)
        self.dataCheck()
        if np.amin(self.data[:,:,1]) < 0  or np.amax(self.data[:,:,1]) > (self.count-subtract_count) :
            add_attr=""
            if np.amin(self.data[:,:,1]) < 0 :
                self.data[:,:,1][self.data[:,:,1] < 0 ] = -self.data[:,:,1][self.data[:,:,1] < 0 ]
                add_attr+="_0-reflect"
            if np.amax(self.data[:,:,1]) > (self.count-subtract_count):
                self.data[:,:,1][self.data[:,:,1] > (self.count-subtract_count) ] = (self.count-subtract_count) - (self.data[:,:,1][self.data[:,:,1] > (self.count-subtract_count) ] - (self.count-subtract_count) )
                add_attr+="_Max-reflect"
            self.maneuver_log(add_attr)
        else : print("check ok!!!")


     #一番初めのフレームの中心のスリットの参照時間を、指定した時間にセットする。それに合わせて全体に対してスライドさせて調節する。baseframe＝ー1で最終フレームを軸とする
    def applyTimeSlide(self,settime:int,baseframe:int=0):    
        print(sys._getframe().f_code.co_name)
        deff = settime - self.data[baseframe,int(self.scan_nums/2),1]
        self.data[:,:,1]+=deff
        print("zp range-調整後:",np.amin(self.data[:,:,1]),np.amax(self.data[:,:,1]))
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(baseframe)+"->"+str(settime))
        
    #シームレスループ作成のための補助的な関数。"addLoopSlidetime"の方が改善すれば、この関数は不要。
    #最初と最終フレームの差分を計算して、差分があれば、差分を埋め合わせるように、全てのフレームにたいして、調整する。
    
    def applyInOutGapFix(self):
        print(sys._getframe().f_code.co_name)
        gapinout=(self.data[0,:,1]-self.data[-1,:,1])
        if abs(np.amax(gapinout)) > 1:
            print("Gapあり",np.amax(gapinout))
            for k in range(self.data.shape[0]):
                self.data[k,:,1]+=gapinout*(k/self.data.shape[0])
        self.maneuver_log(sys._getframe().f_code.co_name)

     #指定したフレームZアレイと最初フレームの差分を計算して、差分があれば、差分を埋め合わせるように、全てのフレームにたいして、調整する。
     #最後のフレームは固定する
    def applyInFix(self,target_z_array):
        print(sys._getframe().f_code.co_name)
        gap=(self.data[0,:,1]-target_z_array)
        if abs(np.amax(gap)) > 1:
            print("Gapあり",np.amax(gap))
            for k in range(self.data.shape[0]):
                self.data[k,:,1]-=gap*((self.data.shape[0]-k)/self.data.shape[0])
        self.maneuver_log(sys._getframe().f_code.co_name)
    #指定したZアレイと最後のフレームの差分を計算して、差分があれば、差分を埋め合わせるように、全てのフレームにたいして、調整する。
    #最初のフレームは固定する
    def applyOutFix(self,target_z_array,ease=True):
        print(sys._getframe().f_code.co_name)
        gap=(self.data[-1,:,1]-target_z_array)
        if abs(np.amax(gap)) > 1:
            print("Gapあり",np.amax(gap))
            for k in range(self.data.shape[0]):
                if ease :
                    # イーズインアウト計算（シンプルなsin関数を使用）
                    t =np.clip(k/self.data.shape[0],0,1)  # 進行度（0から1）
                    easing = 0.5 - math.cos(t * math.pi) * 0.5  # イーズインアウト補間
                    # データの調整
                    self.data[k, :, 1] -= gap * easing
                else:
                    self.data[k,:,1]-=gap*(k/self.data.shape[0])
        self.maneuver_log(sys._getframe().f_code.co_name)
    
    #指定した時間と指定したフレーム（a-frame）の差分を計算して、差分があれば、差分を埋め合わせるように、A-Bの範囲のフレームにたいして、調整する。
    def applyInPartFix(self,target_z,a_frame,b_frame):
        print(sys._getframe().f_code.co_name)
        gap=(self.data[a_frame,:,1]-target_z)
        applyrange=b_frame-a_frame
        if np.amax(abs(gap))> 1:
            print("Gapあり",np.amax(gap))
            for k in range(0,b_frame):
                # self.data[k,:,1]-=gap*((applyrange-k)/applyrange)
                # イーズインアウト計算（シンプルなsin関数を使用）
                t =np.clip((b_frame - k) / applyrange,0,1)  # 進行度（0から1）
                easing = 0.5 - math.cos(t * math.pi) * 0.5  # イーズインアウト補間
                # データの調整
                self.data[k, :, 1] -= gap * easing
        self.maneuver_log(sys._getframe().f_code.co_name)

    def applyOutPartFix(self,target_z,a_frame,b_frame,b_frame_s_point=None):
        print(sys._getframe().f_code.co_name)
        if b_frame_s_point == None :  b_frame_s_point = int(self.data.shape[1]//2)
        gap=(self.data[b_frame,b_frame_s_point,1]-target_z)
        applyrange=b_frame-a_frame
        if abs(gap) > 1:
            print("Gapあり",np.amax(gap))
            for k in range(a_frame,self.data.shape[0]):
                # self.data[k,:,1]-=gap*((applyrange-k)/applyrange)
                # イーズインアウト計算（シンプルなsin関数を使用）
                t =np.clip((k-a_frame) / applyrange,0,1)  # 進行度（0から1）
                easing = 0.5 - math.cos(t * math.pi) * 0.5  # イーズインアウト補間
                # データの調整
                self.data[k, :, 1] -= gap * easing
        self.maneuver_log(sys._getframe().f_code.co_name)

    
    def applyOutPartFixB(self,target_z_array,a_frame,b_frame,base_z_array=None):
        print(sys._getframe().f_code.co_name)
        if base_z_array is None :
            gap=(self.data[b_frame,:,1]-target_z_array)
        else :
            gap=(base_z_array-target_z_array)
        applyrange=b_frame-a_frame
        if np.amax(abs(gap))> 1:
            print("Gapあり",np.amax(gap))
            for k in range(a_frame,self.data.shape[0]):
                # self.data[k,:,1]-=gap*((applyrange-k)/applyrange)
                # イーズインアウト計算（シンプルなsin関数を使用）
                t =np.clip((k-a_frame) / applyrange,0,1)  # 進行度（0から1）
                easing = 0.5 - math.cos(t * math.pi) * 0.5  # イーズインアウト補間
                # データの調整
                self.data[k, :, 1] -= gap * easing
        self.maneuver_log(sys._getframe().f_code.co_name)



    def img_to_maneuver(self, space_img_path, time_img_path,space_set=None,vrange=None):
        # -------- ファイル情報抽出 --------
        if space_set != None :
            space_range = space_set
        else:
            try:
                info_space = self.extract_params_from_filename(space_img_path)
                space_range = info_space["range"]
            except ValueError:
                print(f"[警告] ファイル名から情報を抽出できなかったため、scanwidth={self.scan_nums} を使用します")
                space_range = self.scan_nums 

        
        # -------- 画像読み込み（16bit）--------
        loaded_images = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in [space_img_path, time_img_path]]

        # # --- scan_direction == 1 のときは上下反転で読み込む ---
        if getattr(self, "scan_direction", 1) == 0:
            loaded_images = [img.T for img in loaded_images]
        else :
            loaded_images = [img for img in loaded_images]

        # -------- Space 正規化解除 --------
        normalized_space = (loaded_images[0] / 65535) * (space_range - 1)

        # -------- Time 正規化解除 --------
        if vrange == None: 
            info_time  = self.extract_params_from_filename(time_img_path)
            vmin, vmax = info_time["vmin"], info_time["vmax"]
        else :
            vmin, vmax = vrange[0], vrange[1]        
        normalization_value = vmax - vmin
        normalized_time = (loaded_images[1] / 65535) * normalization_value + vmin

        # -------- データ統合 --------
        reconstructed_data = np.stack((normalized_space, normalized_time), axis=-1)
        self.data = reconstructed_data
    
    def img_to_maneuver_rate_based(self, time_rate_path, space_img_path=None, space_set=None,start_time=0.0, rate_range=None,rate_baseline=None,rate_startpoint=None):
        # -------- max_dev の取得 --------
        max_dev = self.extract_params_from_filename(time_rate_path)["max_dev"] if rate_range is None else rate_range

        # -------- レート画像読み込み --------
        rate_image = cv2.imread(time_rate_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # --- scan_direction == 1 のとき上下反転 ---
        if getattr(self, "scan_direction", 1) == 1:
            rate_image = rate_image
        else:
            rate_image = rate_image.T  # 座標系を合わせる

        # -------- スペース画像または自動生成 --------
        if space_img_path is not None:
            try:
                if space_set ==None:
                    info_space = self.extract_params_from_filename(space_img_path)
                    space_range = info_space["range"]
                else:
                    space_range = space_set
            except ValueError:
                print(f"[警告] ファイル名から情報を抽出できなかったため、scanwidth={self.scan_nums} を使用します")
                space_range = self.scan_nums 

            loaded_space_image = cv2.imread(space_img_path, cv2.IMREAD_UNCHANGED)

            # scan_direction==1 の場合は上下反転して合わせる
            if getattr(self, "scan_direction", 1) == 1:
                loaded_space_image = loaded_space_image
            else:
                loaded_space_image = loaded_space_image.T
            normalized_space = (loaded_space_image / 65535) * (space_range - 1)

        else:
            # スペース情報がない場合は横方向のインデックスを線形に作成
            normal_space = np.arange(rate_image.shape[1], dtype=np.float32)
            normalized_space = np.tile(normal_space, (rate_image.shape[0], 1))

        # -------- 再生レート復元 --------"
        baseline=1.0 if rate_baseline == None else rate_baseline
        vmin, vmax = baseline - max_dev, baseline + max_dev
        playback_rates = (rate_image / 65535.0) * (vmax - vmin) + vmin

        # -------- 再生時間積算 --------
        frame_step = self.recfps / self.outfps
        cumulative_time = np.zeros_like(playback_rates)
        rate_startpoint=0 if rate_startpoint == None else rate_startpoint
        cumulative_time[0, :] = start_time
        for t in range(1, playback_rates.shape[0]):
            cumulative_time[t, :] = cumulative_time[t - 1, :] + playback_rates[t - 1, :] * frame_step

        # -------- データ統合 --------
        reconstructed_data = np.stack((normalized_space, cumulative_time), axis=-1)
        self.data = reconstructed_data


    # def img_to_maneuver_rate_based(
    #     self,
    #     space_flow_img_path,
    #     time_rate_img_path,
    #     start_time=0.0,
    #     range_play_rate=5.0,
    #     standard_play_rate=1.0
    # ):
    #     file_names = [
    #         space_flow_img_path,
    #         time_rate_img_path
    #     ]

    #     # 画像データを読み込んでリストに格納（16bit画像）
    #     loaded_images = [cv2.imread(file_name, cv2.IMREAD_UNCHANGED) for file_name in file_names]
    #     # 強制的にグレースケールで読み込む！
    #     # loaded_images = [cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) for file_name in file_names]

    #     print("Image 1 shape:", loaded_images[0].shape)
    #     print("Image 2 shape:", loaded_images[1].shape)

    #     # 転置して元の軸（[frame, scan]）に戻す
    #     loaded_images = [img.T for img in loaded_images]

    #     # 正規化
    #     # 1枚目：スリット方向の値（例: 位置） → scan_numsに合わせてスケーリング
    #     normalized_space = (loaded_images[0] / 65535) * (self.scan_nums - 1)

    #     # 再生レートの線形マッピング（0→standard-range, 32768→standard, 65535→standard+range）
    #     rate_image = loaded_images[1].astype(np.float32)
    #     neutral_gray = 32768.0

    #     slope = (2 * range_play_rate) / 65535.0
    #     playback_rates = slope * (rate_image - neutral_gray) + standard_play_rate

    #    # 🔧 フレーム単位で進む時間
    #     frame_step = self.recfps / self.outfps  # 例: 120 / 30 = 4.0

    #     # 時間位置の累積
    #     cumulative_time = np.zeros_like(playback_rates)
    #     cumulative_time[0, :] = start_time
    #     for t in range(1, playback_rates.shape[0]):
    #         # 前の時間位置 + (前のフレームの再生レート × frame_step)
    #         cumulative_time[t, :] = cumulative_time[t - 1, :] + playback_rates[t - 1, :] * frame_step
        
    #     # 軌道データ再構成
    #     reconstructed_data = np.stack((normalized_space, cumulative_time), axis=-1)

    #     print("Reconstructed Data Shape:", reconstructed_data.shape)
    #     print("Data Type:", reconstructed_data.dtype)

    #     self.data = reconstructed_data

    # def extract_normalization_value(self, file_name):
    #     # ファイル名から 'range' と '.png' の間の数値を抽出
    #     match = re.search(r'range(\d+\.?\d*)\.png', file_name)
    #     if match:
    #         return float(match.group(1))
    #     else:
    #         raise ValueError("Normalization value not found in the file name.")

    # def img_to_maneuver(self, space_flow_img_path, time_img_path):
    #     file_names = [
    #         space_flow_img_path,
    #         time_img_path
    #     ]

    #     # 画像データを読み込んでリストに格納（16bit画像）
    #     loaded_images = [cv2.imread(file_name, cv2.IMREAD_UNCHANGED) for file_name in file_names]
    #     # 強制的にグレースケールで読み込む！
    #     # loaded_images = [cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) for file_name in file_names]

    #     print("Image 1 shape:", loaded_images[0].shape)
    #     print("Image 2 shape:", loaded_images[1].shape)

    #     # 転置して元の軸（[frame, scan]）に戻す
    #     loaded_images = [img.T for img in loaded_images]

    #     # 正規化
    #     # 1枚目：スリット方向の値（例: 位置） → scan_numsに合わせてスケーリング
    #     normalized_image_1 = (loaded_images[0] / 65535) * (self.scan_nums - 1)

    #     # 2枚目：時間方向の値 → 自前のnormalization値でスケーリング
    #     normalization_value = self.extract_normalization_value(file_names[1])
    #     normalized_image_2 = (loaded_images[1] / 65535) * normalization_value

    #     # 軌道データとして再構成：[frame数, scan数, 2]
    #     reconstructed_data = np.stack((normalized_image_1, normalized_image_2), axis=-1)

    #     print("Reconstructed Data Shape:", reconstructed_data.shape)
    #     print("Data Type:", reconstructed_data.dtype)

    #     self.data = reconstructed_data

    def data_save(self, attr=None, sep=0):
        if attr is None:
            attr = ""
        saved_paths = []
        if sep == 0:
            filename = f"{self.ORG_NAME}_{self.out_name_attr}{attr}_raw.npy"
            np.save(filename, self.data)
            saved_paths.append(filename)
        else:
            split_data = np.array_split(self.data, sep)
            for i, part in enumerate(split_data):
                filename = f"{self.ORG_NAME}_{self.out_name_attr}{attr}_raw{i+1}.npy"
                np.save(filename, part)
                saved_paths.append(filename)
        return saved_paths


    # 映像のレンダリング   
    # ========================================
    '''出力形式の指定フラグ（out_type）
    new_transprocess(out_type=...) に渡す整数で、
    書き出す映像フォーマット・ビット深度・色域を制御します。
    
      0: OUT_STILL
          - 各フレームを静止画（PNG, JPGなど）で保存。
          - SDR (8bit) 想定。
    
      1: OUT_H264
          - H.264 / AVC コーデック。
          - SDR 8bit, yuv420p 出力。
          - 軽量で汎用的（再生・配布用）。
    
      2: OUT_H265
          - HEVC / H.265 コーデック。
          - HDR10 (PQ, BT.2020, 10bit) 出力。
          - GPUエンコード（VideoToolbox）対応。
          - 高画質・高圧縮率・視聴向け。
    
      3: OUT_PRORES_422
          - Apple ProRes 422 HQ。
          - 10bit 4:2:2, 高品質中間素材向け。
          - CPUエンコードのみ。
    
      4: OUT_PRORES_4444
          - Apple ProRes 4444。
          - 10bit 4:4:4, 最高品質（ポスプロ用）。
          - アルファチャンネルを保持可能。
    
      5: OUT_H265_SDR
          - HEVC / H.265 (SDR 10bit, BT.709)。
          - 色空間はBT.709、PQ/HLGなし。
          - 高品質なSDR配信用。
    
      6: OUT_PRORES_422_SDR
          - ProRes 422 HQ (SDR BT.709)。
          - Rec.709、10bit、4:2:2。
          - 編集用中間素材（HDRでないカメラ素材用）。
    
    例：
      self.new_transprocess(out_type=self.OUT_H265)
      self.new_transprocess(out_type=self.OUT_PRORES_422_SDR)
    ========================================
    '''

    # ====================================================================
    # transprocess() — new_transprocess のリファクタリング版
    # ====================================================================
    def split_3_npySave(self):
        #ディレクトリ作成、そのディデクトリに移動
        NPATH = self.ORG_PATH+"/"+self.ORG_NAME+"_"+self.out_name_attr
        if os.path.isdir(NPATH)==False:
            os.makedirs(NPATH)#ディレクトリ作成
        os.chdir(NPATH)
        np.save(self.ORG_NAME+"_"+self.out_name_attr +'_raw.npy',self.data)
        np.save(self.ORG_NAME+"_"+self.out_name_attr +'_raw_L.npy',self.data[:,:self.width,:])
        new_array=self.data[:,self.width:self.width*2,:]
        new_array[:,:,0]-=self.width
        np.save(self.ORG_NAME+"_"+self.out_name_attr +'_raw_C.npy',new_array)
        new_array=self.data[:,self.width*2:,:]
        new_array[:,:,0]-=self.width*2
        np.save(self.ORG_NAME+"_"+self.out_name_attr +'_raw_R.npy',new_array)
        print(self.data[:,:self.width,:].shape)
        print(self.data[:,self.width:self.width*2,:].shape)
        print(self.data[:,self.width*2:,:].shape)
        self.data=None
        new_array=None
        gc.collect()

    # midtideなどのインプット画像よりも３倍サイズを大きくして出力させる場合にのみ使用。
    # NPYファイルをLeft,center,Rightと3分割して出力させる。連結したNPYファイルも出力する。
    # 3分割したNPYファイルのパスを配列で返す。
    def split_3_npysavereturn(self):
        #ディレクトリ作成、そのディデクトリに移動
        # NPATH = self.ORG_PATH+"/"+self.ORG_NAME+"_"+self.out_name_attr
        # if os.path.isdir(NPATH)==False:
        #     os.makedirs(NPATH)#ディレクトリ作成
        # os.chdir(NPATH)
        # np.save(self.ORG_NAME+'_raw.npy',self.data)
        np.save(self.ORG_NAME+'_raw_L.npy',self.data[:,:int(self.width),:])
        new_array=self.data[:,int(self.width):int(self.width*2),:]
        # new_array[:,:,0]-=self.width
        np.save(self.ORG_NAME+'_raw_C.npy',new_array)
        new_array=self.data[:,int(self.width*2):,:]
        # new_array[:,:,0]-=self.width*2
        np.save(self.ORG_NAME+'_raw_R.npy',new_array)
        print(self.data[:,:int(self.width),:].shape)
        print(self.data[:,int(self.width):int(self.width*2),:].shape)
        print(self.data[:,int(self.width*2):,:].shape)
        self.data=None
        new_array=None
        gc.collect()
        current_directory = os.getcwd()
        return [current_directory +"/"+self.ORG_NAME+'_raw_L.npy',current_directory +"/"+self.ORG_NAME+'_raw_C.npy',current_directory +"/"+self.ORG_NAME+'_raw_R.npy']

    # 時間と空間のピクセルのマトリクスに対して、動的な波の形状により再生断面を得る。空間軸を固定するか否かの切り替えも可能
