"""drawManeuver の "適用" 系変換 (TransformsApplyMixin)

self.data (3D 配列) に対してその場で時間/空間を変形/ぼかす:
- applyLoopBlur / applyConnectLoopBlur / applyPointBlur
- applySpaceBlur / applyTimeBlur / applyCustomeBlur
- applyTimeOblique / applyTimeForwardAutoSlow
- applyTimeForward / applyTimeChoppyLoop(B) / applyTimeLoop(B)
- applyTimeClip / applyTimebySpace / applyTimebyKeyframetoSpace
- applySpaceFlip / applySpaceFlat
- timeFlowKeepingExtend / applyTimeFlowKeepingExtend(_*)
"""
import os
import sys
import math
import cv2
import numpy as np
import easing

from ._utils import (
    custom_blur,
    custom_onedimention_blur,
    onedimention_LoopBlur,
    calculate_parallel_perpendicular,
)


class TransformsApplyMixin:
    def applyLoopBlur(self,sblur,tblur):
        print(sys._getframe().f_code.co_name)
        array=self.data
        self.data= np.vstack((np.vstack((array,self.data)),array))
        if sblur > 0 : self.data[:,:,0]=cv2.blur(self.data[:,:,0],(1,int(sblur)))
        if tblur > 0 : self.data[:,:,1]=cv2.blur(self.data[:,:,1],(1,int(tblur)))
        self.data = self.data[array.shape[0]:array.shape[0]+array.shape[0],:,:]
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+"s"+str(sblur)+"b"+str(tblur))
    
    #applyCustomeBlurの埋め込み
    def applyConnectLoopBlur(self,sblur,tblur,connect_frame=100):
        print(sys._getframe().f_code.co_name)
        array=self.data
        self.data= np.vstack((np.vstack((array,self.data)),array))
        # print(self.data.shape)
        # self.data[:,:,0]=cv2.blur(self.data[:,:,0],(1,int(sblur)))
        # self.data[:,:,1]=cv2.blur(self.data[:,:,1],(1,int(tblur)))
        if tblur > 0 :
            self.data = custom_blur(self.data,s_frame=array.shape[0]-connect_frame, e_frame=array.shape[0]+connect_frame, bl_time = tblur, dim_num=1)
            # self.maneuver_2dplot()
            self.data = custom_blur(self.data,s_frame=array.shape[0]*2-connect_frame, e_frame=array.shape[0]*2+connect_frame, bl_time = tblur, dim_num=1)
            # self.maneuver_2dplot()
            
        if sblur > 0 :
            self.data = custom_blur(self.data,s_frame=array.shape[0]-connect_frame, e_frame=array.shape[0]+connect_frame, bl_time = sblur, dim_num=0)
            self.data = custom_blur(self.data,s_frame=array.shape[0]*2-connect_frame, e_frame=array.shape[0]*2+connect_frame, bl_time = sblur, dim_num=0)
        self.data = self.data[array.shape[0]:array.shape[0]+array.shape[0],:,:]
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+"s"+str(sblur)+"b"+str(tblur))
    
   #applyCustomeBlurの埋め込み added 2023 12/12
    def applyPointBlur(self,point_frame,sblur,tblur,range_frame=100):
        print(sys._getframe().f_code.co_name)
        if tblur > 0 :
            self.data = custom_blur(self.data,s_frame=point_frame-range_frame, e_frame=point_frame+range_frame, bl_time = tblur, dim_num=1)
        if sblur > 0 :
            self.data = custom_blur(self.data,s_frame=point_frame-range_frame, e_frame=point_frame+range_frame, bl_time = sblur, dim_num=0)
        
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+"s"+str(sblur)+"b"+str(tblur))
    
    #added 2023 10/22
    #指定したフレーム数で保管する。保管するz距離に応じて速度が変わる。
    def applySpaceBlur(self,bl_time):
        print(sys._getframe().f_code.co_name)
        self.data[:,:,0]=cv2.blur(self.timeFlowKeepingExtend(int(np.ceil(bl_time/2)))[:,:,0],(1,int(int(np.ceil(bl_time/2)))))[int(np.ceil(bl_time/2)):-int(np.ceil(bl_time/2)),:]
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(bl_time))

    def applyTimeBlur(self,bl_time):
        print(sys._getframe().f_code.co_name)
        self.data[:,:,1]=cv2.blur(self.timeFlowKeepingExtend(int(np.ceil(bl_time/2)))[:,:,1],(1,int(int(np.ceil(bl_time/2)))))[int(np.ceil(bl_time/2)):-int(np.ceil(bl_time/2)),:]
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(bl_time))

    

#範囲指定をして配列に対してブラーをかける
    def applyCustomeBlur(self,s_frame,e_frame,bl_time,dim_num=1):
        time_array=self.data[:,:,dim_num]
        print(sys._getframe().f_code.co_name)
        blur_array=np.zeros([e_frame-s_frame,self.scan_nums],dtype=np.float64)
        apply_bl_time=0
        weightmean=1
        if e_frame-s_frame < bl_time/2 :
            bl_time = (e_frame-s_frame)*2
        for y in range(e_frame-s_frame):
            if y > (bl_time/2)-1 and y < (e_frame-s_frame)- bl_time/2 :
                apply_bl_time=max(1,int(bl_time/2))  # bl_time=1のときint(0.5)=0→np.mean([])=NaN防止
                # print("BlurProcess:if:",y,apply_bl_time)
            else:
                apply_bl_time= max(1, y+1 if y < bl_time/2 else (e_frame-s_frame)-y)
                # print("BlurProcess:else:",y,apply_bl_time)
            # print("blur range:",apply_bl_time)
            #重み付平均
            if weightmean:
                if y < bl_time :
                        wA=bl_time-y
                        wB=y
                elif ((e_frame-s_frame-1)-y)  < bl_time :
                    wA=bl_time-((e_frame-s_frame-1)-y) 
                    wB=((e_frame-s_frame-1)-y) 
                else : 
                    wA=0
                    wB=1
                # print(wA,wB)
            for x in range(time_array.shape[1]):
                sl_s = max(0, int(s_frame+y-apply_bl_time))
                sl_e = int(s_frame+y+apply_bl_time)
                if weightmean:
                    blur_array[y,x]= (time_array[s_frame+y,x]*wA + np.mean(time_array[sl_s:sl_e,x])*wB) / (wA+wB)
                else : blur_array[y,x]=np.mean(time_array[sl_s:sl_e,x])
                # blur_array[y,x]=np.mean(time_array[int(s_frame+y-apply_bl_time):int(s_frame+y+apply_bl_time)+1,x])
        time_array[s_frame:e_frame,:]=blur_array
        print(time_array.shape)
        self.data[:,:,dim_num]=time_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(bl_time))
        
    
    # 時間軸、空間軸、ともに変化のないフラットな配列を"frame_nums"で指定されたフレーム数分生成。
    def applyTimeOblique(self,maxgap):
        print(sys._getframe().f_code.co_name)
        for i in range(self.scan_nums):
            self.data[:,i,1] += maxgap * i/self.scan_nums 
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1])

    # 時間軸、空間軸、ともに最終列の配列を"frame_nums"で指定されたフレーム数分生成して加える。
    def applyTimeForwardAutoSlow(self,slide_time:int=1,defaultAddTime:int=100,addTimeEasing:bool=True,easeRatio:float=0.3):
        print(sys._getframe().f_code.co_name)
        for k in range(self.data[:,:,1].shape[0]):
            self.data[k,:,1] += slide_time*k
        if addTimeEasing == True :
            # introNormalTime=int((self.recfps/self.outfps)*defaultAddTime*(1.0-easeRatio))
            introFrameNum=int(defaultAddTime*(1.0-easeRatio))
            introEaseArray=np.zeros(defaultAddTime-introFrameNum)
            outroEaseArray=np.zeros(defaultAddTime-introFrameNum)
            for n in range(introEaseArray.shape[0]):
                introEaseArray[n]=easing.inOutQuad(n,self.recfps/self.outfps,(slide_time-self.recfps/self.outfps),introEaseArray.shape[0])
                # introEaseArray[n]=easing.inOutQuad(n,self.recfps/self.outfps,slide_time,introEaseArray.shape[0])
            for n in range(outroEaseArray.shape[0]):
                outroEaseArray[n]=easing.inOutQuad(n,slide_time,(self.recfps/self.outfps-slide_time),outroEaseArray.shape[0])
                # outroEaseArray[n]=easing.inOutQuad(n,slide_time,self.recfps/self.outfps,outroEaseArray.shape[0])
            outroNormalTime=int((self.recfps/self.outfps)*defaultAddTime*(1.0-easeRatio))
            outroFrameNum=int(defaultAddTime*(1.0-easeRatio))
            extratime = defaultAddTime
            introtime = extratime
            outrotime = extratime
        else:
            extratime = defaultAddTime
            introtime=extratime
            outrotime=extratime
        extra_array = np.zeros((introtime+self.data.shape[0]+outrotime,self.data.shape[1],2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
        normalFrame=np.arange(0,self.data.shape[1])
        introFrame = self.data[0,:, 0]
        outroFrame = self.data[-1,:,0]
        #introとoutroの間のこれまでの計算部分をまるっと代入する。
        extra_array[introtime:-outrotime]=self.data

        if addTimeEasing == True :
            #introのEase部分のフレーム拡張
            for i in range(introEaseArray.shape[0]):
                extra_array[introtime-i-1,:,0] = introFrame 
                #audioへの適合もあるため、この時点ではビットレートを高くして計測
                #introEaseArrayには、各フレーム間の経過フレームがフロートで格納されているので、そのデータをself.dataの最初のフレームに刻まれたタイムコードから減算して記録しなおうす。
                extra_array[introtime-i-1,:,1] = extra_array[introtime-i,:,1] - introEaseArray[-1-i]
            print(np.amin(extra_array[:,:,1]),np.amax(extra_array[:,:,1]))
            
            #introのEase前の普通レート再生部分のフレーム拡張
            for i in range(introFrameNum):
                extra_array[introFrameNum-i-1,:,0] = introFrame
                extra_array[introFrameNum-i-1,:,1] =extra_array[introFrameNum,:,1] - ((i+1)*self.recfps/self.outfps)#audioへの適合もあるため、この時点ではビットレートを高くして計測
            print(np.amin(extra_array[:,:,1]),np.amax(extra_array[:,:,1]))
            print(np.amin(extra_array[:,:,1]),np.amax(extra_array[:,:,1]))
            print(np.amin(extra_array[:,:,1]),np.amax(extra_array[:,:,1]))
            
            #outroのEase部分のフレーム拡張
            for i in range(outroEaseArray.shape[0]):
                extra_array[i+introtime+self.data.shape[0],:,0]=outroFrame
                extra_array[i+introtime+self.data.shape[0],:,1]=self.data[-1,:,1]+np.sum(outroEaseArray[0:i+1])#audioへの適合もあるため、この時点ではビットレートを高くして計測
            print(np.amin(extra_array[:,:,1]),np.amax(extra_array[:,:,1]))
            #introのEase後の普通レート再生部分のフレーム拡張
            for i in range(outroFrameNum):
                extra_array[i+introtime+self.data.shape[0]+outroEaseArray.shape[0],:,0]= outroFrame
                extra_array[i+introtime+self.data.shape[0]+outroEaseArray.shape[0],:,1]= extra_array[introtime+self.data.shape[0]+outroEaseArray.shape[0]-1,:,1]+(i+1)*self.recfps/self.outfps
            print(np.amin(extra_array[:,:,1]),np.amax(extra_array[:,:,1]))
        else :
            #Ease処理を必要としない場合、シンプルに再生レート（slide_time）に従い拡張する。
            for i in range(introtime):
                extra_array[introtime-i-1,:,0] = introFrame
                extra_array[introtime-i-1,:,1] = self.data[0,:,1] -  ((i+1)*self.recfps/self.outfps)
            extra_array[introtime: -outrotime] = self.data
            for i in range(outrotime):
                extra_array[i+introtime+self.data.shape[0],:,0]=outroFrame
                extra_array[i+introtime+self.data.shape[0],:,1]=self.data[-1,:,1]+ ((i+1)*self.recfps/self.outfps)
        self.data=extra_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("applyTimeForward")[1])

    # 与えた軌道配列の１フレーム目を手前に延長させる。Zのレートは0になる。
    def timeFlowKeepingExtend(self,frame_nums: int, fade: bool = False, intro: bool = True, outro: bool = True,fade_speed=0,fade_type="inout",space_apply=False):
        print(sys._getframe().f_code.co_name)
        if intro == False and outro == False: return

        if intro == False or outro == False:
            extra_array = np.zeros((self.data.shape[0] + frame_nums, self.data.shape[1], 2), dtype=np.float64)    
        else:
            extra_array = np.zeros((self.data.shape[0] + frame_nums * 2, self.data.shape[1], 2), dtype=np.float64)
        
        xyfirstFrame = self.data[0, :, 0]
        xylastFrame = self.data[-1, :, 0]
        if space_apply:
            sfirstDiff = self.data[1, :, 0] - self.data[0, :, 0]
            slastDiff = self.data[-2, :, 0] - self.data[-1, :, 0]
        if intro:
            zfirstDiff = self.data[1, :, 1] - self.data[0, :, 1]
            print("zfirstDiff=",np.nanmean(zfirstDiff))
        if outro:
            zlastDiff = self.data[-2, :, 1] - self.data[-1, :, 1]
            print("zlastDiff=",np.nanmean(zlastDiff))
        fade_speed_array=np.full(self.data.shape[1], -1*fade_speed)

        # コピーするデータの範囲を決定
        if intro:
            copy_start = frame_nums
            copy_end = frame_nums + self.data.shape[0] 
        else :
            copy_start = 0
            copy_end =  self.data.shape[0] 
        

        extra_array[copy_start:copy_end] = self.data

        if fade:
            introEaseArray = np.zeros((frame_nums, self.data.shape[1]), dtype=np.float64) if intro else None
            outroEaseArray = np.zeros((frame_nums, self.data.shape[1]), dtype=np.float64) if outro else None
            if space_apply :
                s_introEaseArray = np.zeros((frame_nums, self.data.shape[1]), dtype=np.float64) if intro else None
                s_outroEaseArray = np.zeros((frame_nums, self.data.shape[1]), dtype=np.float64) if outro else None
            if intro:
                for n in range(introEaseArray.shape[0]):
                    if fade_type=="inout":
                        introEaseArray[n] = easing.inOutQuad(n, zfirstDiff, -fade_speed_array-zfirstDiff, introEaseArray.shape[0])
                    elif fade_type=="in":
                        introEaseArray[n] = easing.inQuad(n, zfirstDiff, -fade_speed_array-zfirstDiff, introEaseArray.shape[0])
                    elif fade_type=="out":
                        introEaseArray[n] = easing.outQuad(n, zfirstDiff, -fade_speed_array-zfirstDiff, introEaseArray.shape[0])
                    if space_apply :
                        if fade_type=="inout":
                            s_introEaseArray[n] = easing.inOutQuad(n, sfirstDiff, -fade_speed_array-sfirstDiff, introEaseArray.shape[0])
                        elif fade_type=="in":
                            s_introEaseArray[n] = easing.inQuad(n, sfirstDiff, -fade_speed_array-sfirstDiff, introEaseArray.shape[0])
                        elif fade_type=="out":
                            s_introEaseArray[n] = easing.outQuad(n, sfirstDiff, -fade_speed_array-sfirstDiff, introEaseArray.shape[0])

            if outro:
                for n in range(outroEaseArray.shape[0]):
                    if fade_type=="inout":
                        outroEaseArray[n] = easing.inOutQuad(n, zlastDiff, fade_speed_array-zlastDiff, outroEaseArray.shape[0])
                    elif fade_type=="in":
                        outroEaseArray[n] = easing.inOutQuad(n, zlastDiff, fade_speed_array-zlastDiff, outroEaseArray.shape[0])
                    elif fade_type=="out":
                        outroEaseArray[n] = easing.inOutQuad(n, zlastDiff, fade_speed_array-zlastDiff, outroEaseArray.shape[0])
                if space_apply :
                        if fade_type=="inout":
                            s_outroEaseArray[n] = easing.inOutQuad(n, slastDiff, fade_speed_array-slastDiff, outroEaseArray.shape[0])
                        elif fade_type=="in":
                            s_outroEaseArray[n] = easing.inOutQuad(n, slastDiff, fade_speed_array-slastDiff, outroEaseArray.shape[0])
                        elif fade_type=="out":
                            s_outroEaseArray[n] = easing.inOutQuad(n, slastDiff, fade_speed_array-slastDiff, outroEaseArray.shape[0])

        for i in range(frame_nums):
            if outro:
                if space_apply :
                    if fade:
                        extra_array[copy_end + i, :, 0] = extra_array[copy_end + i - 1, :, 0] - s_outroEaseArray[i]
                    else:
                        extra_array[copy_end + i, :, 0] = extra_array[copy_end + i - 1, :, 0] - slastDiff
                else : 
                    extra_array[copy_end + i, :, 0] = xylastFrame

                if fade:
                    extra_array[copy_end + i, :, 1] = extra_array[copy_end + i - 1, :, 1] - outroEaseArray[i]
                else:
                    extra_array[copy_end + i, :, 1] = extra_array[copy_end + i - 1, :, 1] - zlastDiff
            if intro:
                if space_apply :
                    if fade:
                        extra_array[frame_nums - i - 1, :, 0] = extra_array[frame_nums - i, :, 0] - s_introEaseArray[i]
                    else:
                        extra_array[frame_nums - i - 1, :, 0] = extra_array[frame_nums - i, :, 0] - sfirstDiff
                else :
                    extra_array[frame_nums - i - 1, :, 0] = xyfirstFrame

                if fade:
                    extra_array[frame_nums - i - 1, :, 1] = extra_array[frame_nums - i, :, 1] - introEaseArray[i]
                else:
                    extra_array[frame_nums - i - 1, :, 1] = extra_array[frame_nums - i, :, 1] - zfirstDiff
        return extra_array


    # 与えた軌道配列に、延長させたフレームをプリペンド、アペンドする。XYフレームそれぞれ最終フレームと最初のフレームと同じデータで延長させる。Z(アウト時間）に関しては最終の変化量を維持して延長させる。fade引数をTrueでスピード０に落ち着かせる。
    def applyTimeFlowKeepingExtend(self,frame_nums:int,fade:bool=False,intro: bool = True, outro: bool = True,fade_speed=0,fade_type="inout",space_apply=False):
        print(sys._getframe().f_code.co_name)
        self.data = self.timeFlowKeepingExtend(frame_nums,fade,intro,outro,fade_speed,fade_type,space_apply)
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1])

    def applyTimeFlowKeepingExtend_CoodinateBase_Intro(self, target_z, num_frames):
        """イントロ延長: target_zを先頭フレームのz値としてnum_frames分プリペンド。
        各スリットのステップ = (data[0, slit, 1] - target_z) / num_frames で計算し、
        累積誤差ゼロを保証する。"""
        print(sys._getframe().f_code.co_name, f"target_z={target_z}, num_frames={num_frames}")
        extra_array = np.zeros((self.data.shape[0] + num_frames, self.data.shape[1], 2), dtype=np.float64)
        # 既存データをnum_framesオフセットでコピー
        extra_array[num_frames:] = self.data
        # 各スリットのx座標は先頭フレームと同じ
        xyfirstFrame = self.data[0, :, 0]
        # 各スリットごとのステップ（target_z → data[0]へ向かう方向）
        z_current = self.data[0, :, 1]  # shape: (num_slits,)
        step_per_slit = (z_current - target_z) / num_frames  # shape: (num_slits,)
        print(f"  step_per_slit mean={np.nanmean(step_per_slit):.6f}, std={np.nanstd(step_per_slit):.6f}")
        # 延長フレームを埋める: frame 0 = target_z, frame num_frames-1 = target_z + step*(num_frames-1)
        # frame num_frames = data[0] (既存データ) = target_z + step*num_frames
        for i in range(num_frames):
            extra_array[i, :, 0] = xyfirstFrame
            extra_array[i, :, 1] = target_z + step_per_slit * i
        self.data = extra_array
        # 検証
        z_new_first = np.nanmean(self.data[0, :, 1])
        z_boundary = np.nanmean(self.data[num_frames, :, 1])
        print(f"  result: z[0]={z_new_first:.1f} (target={target_z:.1f}), z[{num_frames}]={z_boundary:.1f}")
        self.maneuver_log(f"TimeFlowKeepingExtend_CoodinateBase_Intro_tz{target_z}_n{num_frames}")

    def applyTimeFlowKeepingExtend_CoodinateBase_Outtro(self, target_z, num_frames):
        """アウトロ延長: target_zを末尾フレームのz値としてnum_frames分アペンド。
        各スリットのステップ = (target_z - data[-1, slit, 1]) / num_frames で計算し、
        累積誤差ゼロを保証する。"""
        print(sys._getframe().f_code.co_name, f"target_z={target_z}, num_frames={num_frames}")
        extra_array = np.zeros((self.data.shape[0] + num_frames, self.data.shape[1], 2), dtype=np.float64)
        # 既存データを先頭からコピー
        extra_array[:self.data.shape[0]] = self.data
        copy_end = self.data.shape[0]
        # 各スリットのx座標は末尾フレームと同じ
        xylastFrame = self.data[-1, :, 0]
        # 各スリットごとのステップ（data[-1] → target_zへ向かう方向）
        z_current = self.data[-1, :, 1]  # shape: (num_slits,)
        step_per_slit = (target_z - z_current) / num_frames  # shape: (num_slits,)
        print(f"  step_per_slit mean={np.nanmean(step_per_slit):.6f}, std={np.nanstd(step_per_slit):.6f}")
        # 延長フレームを埋める: frame 1 = data[-1] + step*1, ..., frame num_frames = target_z
        for i in range(num_frames):
            extra_array[copy_end + i, :, 0] = xylastFrame
            extra_array[copy_end + i, :, 1] = z_current + step_per_slit * (i + 1)
        self.data = extra_array
        # 検証
        z_new_last = np.nanmean(self.data[-1, :, 1])
        z_boundary = np.nanmean(self.data[copy_end - 1, :, 1])
        print(f"  result: z[-1]={z_new_last:.1f} (target={target_z:.1f}), z[{copy_end-1}]={z_boundary:.1f}")
        self.maneuver_log(f"TimeFlowKeepingExtend_CoodinateBase_Outtro_tz{target_z}_n{num_frames}")

    def applyTimeForward(self, slide_time=None, start_frame=0, end_frame=None):
        print(sys._getframe().f_code.co_name)

        if slide_time is None:
            slide_time = self.outfps / self.recfps

        if end_frame is None:
            end_frame = self.data.shape[0]

        for k in range(start_frame, end_frame):
            self.data[k, :, 1] += slide_time * (k - start_frame)
        
        if end_frame < self.data.shape[0]:
            self.data[end_frame:,:,1] += slide_time * (end_frame - start_frame) 

        print("Slide_timeの計算後 min-max =", np.amin(self.data[:, :, -1]), np.amax(self.data[:, :, -1]))
        self.maneuver_log(f"TimeForward[{start_frame}:{end_frame}]_{slide_time}")

        #配列全体に時間の順方向の流れ（単位はslide_time）を付与する。
    def applyTimeChoppyLoop(self,slide_time=None,frequency=1,phase_shift=0,rise = 0.5,fall = 0.5 ):
        print(sys._getframe().f_code.co_name)
        if slide_time==None:slide_time=self.outfps/self.recfps
        N = self.data[:,:,1].shape[0]
        # 時間軸を生成
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # 位相のずれを考慮に入れた時間軸
        t_shifted = t * frequency + phase_shift * np.pi 
        # 三角波の生成
        triangle_wave = np.zeros_like(t_shifted)
        for i, ti in enumerate(t_shifted):
            # 周期のどの部分にいるかを計算
            mod_t = ti % (2 * np.pi)
            if mod_t < 2 * np.pi * rise:  #上昇部分
                triangle_wave[i] = (mod_t / (2 * np.pi * rise)) * 2 - 1
            else:  #下降部分
                triangle_wave[i] = -((mod_t - 2 * np.pi * rise) / (2 * np.pi * fall)) * 2 + 1
        # 0から1の範囲に収めるために三角波を再スケーリングしてフレームまいに
        scaled_triangle_wave = (triangle_wave + 1) / 2 * (N/2*slide_time)
        # scaled_triangle_wave を繰り返して形状を合わせる
        scaled_triangle_wave_expanded = np.tile(scaled_triangle_wave, (self.data.shape[1], 1)).T
        self.data[:,:,1] +=scaled_triangle_wave_expanded
        print("Slide_timeの計算後 min-max =",np.amin(self.data[:,:,-1]),np.amax(self.data[:,:,-1]))
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(slide_time))

    def applyTimeChoppyLoopB(self, slide_time=None, frequency=1, phase_shift=0, rise=0.5, fall=0.5, wave_type='triangle',blur=0):
        print(sys._getframe().f_code.co_name)
        if slide_time is None:
            slide_time = self.outfps / self.recfps
        N = self.data[:,:,1].shape[0]
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        t_shifted = t * frequency + phase_shift * np.pi

        if wave_type == 'triangle':
            # 三角波の生成
            triangle_wave = np.zeros_like(t_shifted)
            for i, ti in enumerate(t_shifted):
                mod_t = ti % (2 * np.pi)
                if mod_t < 2 * np.pi * rise:
                    triangle_wave[i] = (mod_t / (2 * np.pi * rise)) * 2 - 1
                else:
                    triangle_wave[i] = -((mod_t - 2 * np.pi * rise) / (2 * np.pi * fall)) * 2 + 1
            wave = triangle_wave
            # 頂点の時間位置を計算
            if blur != 0 :
                vertex_times = []
                period = 2 * np.pi / frequency
                for peak in np.arange(phase_shift * np.pi, t_shifted[-1], period):
                    if peak <= t_shifted[-1]:
                        vertex_times.append(peak)
                    if peak + period * rise <= t_shifted[-1]:
                        vertex_times.append(peak + period * rise)

                # t_shiftedの実際の時間範囲にマッピング
                real_time_vertex_times = [vertex_time * (N / (2 * np.pi)) for vertex_time in vertex_times]

                # 頂点の時間位置を表示
                print("頂点（実数時間）：", real_time_vertex_times)
            
        elif wave_type == 'sine':
            # サイン波の生成
            # 振幅のスケーリング (rise と fall の平均値/ 2を使用)
            # amplitude_scale = (rise + fall) / 2
            wave = np.sin(t_shifted)
        # 波形のスケーリングと適用
        scaled_wave = (wave + 1) / 2 * (N / 2 * slide_time)
        scaled_wave_expanded = np.tile(scaled_wave, (self.data.shape[1], 1)).T
        # plt.plot(scaled_wave_expanded)
        if blur != 0 :
            range_frame=blur
            scaled_wave_expanded=onedimention_LoopBlur(scaled_wave_expanded,blur)
            # for point_frame in real_time_vertex_times:
            #     scaled_wave_expanded=custom_onedimention_blur(scaled_wave_expanded,s_frame=int(point_frame-range_frame), e_frame=int(point_frame+range_frame), bl_time = blur)

        self.data[:,:,1] += scaled_wave_expanded
        print("Slide_timeの計算後 min-max =", np.amin(self.data[:,:,-1]), np.amax(self.data[:,:,-1]))
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1] + str(slide_time))

 
    #与えた軌道配列全体の時間を前半、順方向、後半、逆転して、最後にまた順方向へながれ、最初と終わりの時間差がない。そのままループ再生すればシームレスなループが作られる。
    #デフォルト周波数２hzでしか現在対応できていない。
    def applyTimeLoop(self,slide_time,freq=2,stay_time=30,intepolation_min=300,stay_time_min =30):
        print(sys._getframe().f_code.co_name)
        time_array = self.data[:,:,1]
        #サインカーブ部分最低の長さ
        # intepolation_min=15
        #sty_time部分最低の長さ
        # stay_time_min =10
        stay_time_minmargin =stay_time - stay_time_min
        if(time_array.shape[0]-stay_time*4-intepolation_min*2)<1 :
            print("ERROR")
            return
        gapinout=np.mean(time_array[0]-time_array[-1])-slide_time
        # gapinout=(time_array[0]-time_array[-1])-slide_time
        print("Slide_timeの計算前",time_array[0,0],time_array[-1,0],gapinout)        
        stay_time_extend_margin=time_array.shape[0]-intepolation_min*2 - stay_time*4
        stay_timeA=stay_time
        stay_timeB=stay_time
        if stay_time_extend_margin > 0 :
            if stay_time_extend_margin*slide_time > abs(gapinout):
                if gapinout < 0: 
                    stay_timeB = stay_time+int(abs(gapinout) / slide_time / 2)
                else :
                    stay_timeA = stay_time+int(abs(gapinout) / slide_time / 2)
            elif stay_time_extend_margin*slide_time  + stay_time_minmargin*slide_time*4 > abs(gapinout):
                lastgap=int((abs(gapinout)-stay_time_extend_margin*slide_time)/slide_time)
                if gapinout < 0: 
                    stay_timeB = stay_time+int(stay_time_extend_margin / 2)+(lastgap/4)
                    stay_timeA = stay_time-(lastgap/4)
                else :
                    stay_timeA = stay_time+int(stay_time_extend_margin / 2)+(lastgap/4)
                    stay_timeB = stay_time-(lastgap/2/2)
            else :
                # if gapinout < 0: 
                #     lastgap=abs(gapinout / slide_time)-int(stay_time_extend_margin / slide_time / 2)
                #     stay_timeB = stay_time+int(stay_time_extend_margin / slide_time / 2)+(lastgap/2/2)
                #     stay_timeA = stay_time-(lastgap/2/2)
                print("ERROR,限界のところまで拡張対応、あとはGapinoutで強制的に調整する。")
                if gapinout < 0: 
                    stay_timeB = stay_time+int(stay_time_extend_margin / 2)+stay_time_minmargin
                    stay_timeA = stay_time_min
                else :
                    lastgap=abs(gapinout / slide_time)-int(stay_time_extend_margin / slide_time / 2)
                    stay_timeA = stay_time+int(stay_time_extend_margin / 2)+stay_time_minmargin
                    stay_timeB = stay_time_min
        else :
            print("ERROR")
            return 
        #slide_timeの処理
        print("Stay_timeA:",stay_timeA)
        print("stay-timeB",stay_timeB)
        interpolation_time = time_array.shape[0] -stay_timeB*2-stay_timeA*2
        now_point=0
        for k in range(time_array.shape[0]):
            if k < stay_timeA :
                #スタート
                now_point += slide_time
                # print("start",k,now_point)
            elif k > time_array.shape[0]-stay_timeA:
                #エンド
                now_point += slide_time
                # print("end",k,now_point)
            elif k > (time_array.shape[0]/2-stay_timeB) and k < (time_array.shape[0]/2+stay_timeB):
                #中間の逆再生
                now_point -= slide_time
                # print("中間の逆再生",k,now_point)
            else:
                if k < time_array.shape[0]/2:
                    #前半部分のサインカープ
                    now_point += slide_time*math.cos(math.pi*freq*(k-stay_timeA) / interpolation_time)
                    #print("前半部分のサインカープ",k,now_point)
                else:
                    #後半部分のサインカープ
                    now_point += slide_time*math.cos(math.pi*freq*(k-(stay_timeA+stay_timeB*2+interpolation_time)) / interpolation_time)
                    # print("後半部分のサインカープ",k,now_point)
            # time_array[k]+=int(now_point)
            time_array[k]+=now_point
        gapinout=(time_array[0]-time_array[-1])-slide_time
        gapstart=slide_time-time_array[0]
        print("Slide_timeの計算後",time_array[0,0],time_array[-1,0],gapinout[0],gapstart[0])
        for k in range(time_array.shape[0]):
            # time_array[k]+=gapinout*int(k/time_array.shape[0])+gapstart
            time_array[k]+=gapinout*int(k/time_array.shape[0])+gapstart
        print("Gap計算後",time_array[0,0],time_array[-1,0],gapinout[0],gapstart[0])
        self.data[:,:,1] = time_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1])
    
    #スリットごとにZGAPがある。
    def applyTimeLoopB(self,slide_time,freq=2,stay_time=90,intepolation_min=300,stay_time_min =30):
        print(sys._getframe().f_code.co_name)
        time_array = self.data[:,:,1]
        #サインカーブ部分最低の長さ
        # intepolation_min=15
        #sty_time部分最低の長さ
        # stay_time_min =10
        stay_time_minmargin =stay_time - stay_time_min
        if(time_array.shape[0]-stay_time*4-intepolation_min*2)<1 :
            print("ERROR")
            return
        # gapinout=np.mean(time_array[0]-time_array[-1])-slide_time
        gapinout=(time_array[0]-time_array[-1])-slide_time
        print("Slide_timeの計算前",time_array[0,0],time_array[-1,0],gapinout)        
        stay_time_extend_margin= np.full(self.scan_nums,time_array.shape[0]-intepolation_min*2 - stay_time*4)
        stay_timeA=np.full(self.scan_nums,stay_time)
        stay_timeB=np.full(self.scan_nums,stay_time)
        if stay_time_extend_margin[0] > 0 :
            for i, stem in enumerate(stay_time_extend_margin):
                if stem*slide_time > abs(gapinout[i]):
                    if gapinout[i] < 0: 
                        stay_timeB[i] = stay_time+int(abs(gapinout[i]) / slide_time / 2)
                    else :
                        stay_timeA[i] = stay_time+int(abs(gapinout[i]) / slide_time / 2)
                elif stem*slide_time  + stay_time_minmargin*slide_time*4 > abs(gapinout[i]):
                    lastgap=int((abs(gapinout[i])-stem*slide_time)/slide_time)
                    if gapinout[i] < 0: 
                        stay_timeB[i] = stay_time+int(stem / 2)+(lastgap/4)
                        stay_timeA[i] = stay_time-(lastgap/4)
                    else :
                        stay_timeA[i] = stay_time+int(stem / 2)+(lastgap/4)
                        stay_timeB[i] = stay_time-(lastgap/2/2)
                else :
                    print("ERROR,限界のところまで拡張対応、あとはGapinoutで強制的に調整する。")
                    if gapinout[i] < 0: 
                        stay_timeB[i]  = stay_time+int(stem / 2)+stay_time_minmargin
                        stay_timeA[i]  = stay_time_min
                    else :
                        lastgap=abs(gapinout[i]  / slide_time)-int(stem / slide_time / 2)
                        stay_timeA[i]  = stay_time+int(stem / 2)+stay_time_minmargin
                        stay_timeB[i]  = stay_time_min
        else :
            print("ERROR")
            return 
        #slide_timeの処理
        print("Stay_timeA:",stay_timeA)
        print("stay-timeB",stay_timeB)
        for i in range(time_array.shape[1]):
            now_point=0
            for k in range(time_array.shape[0]):
                interpolation_time = time_array.shape[0] - stay_timeB[i]*2-stay_timeA[i]*2
                if k < stay_timeA[i] :
                    #スタート
                    now_point += slide_time
                    # if i % 100==0 and k%100==0 : print("start",i,k,now_point)
                elif k > time_array.shape[0]-stay_timeA[i]:
                    #エンド
                    now_point += slide_time
                    # if i % 100==0 and k%100==0 : print("end",i,k,now_point)
                elif k > (time_array.shape[0]/2-stay_timeB[i]) and k < (time_array.shape[0]/2+stay_timeB[i]):
                    #中間の逆再生
                    now_point -= slide_time
                    # if i % 100==0 and k%100==0 : print("中間の逆再生",k,now_point)
                else:
                    if k < time_array.shape[0]/2:
                        #前半部分のサインカープ
                        now_point += slide_time*math.cos(math.pi*freq*(k-stay_timeA[i]) / interpolation_time)
                        # if i % 100==0 and k%100==0 : print("前半部分のサインカープ",i,k,now_point)
                    else:
                        #後半部分のサインカープ
                        now_point += slide_time*math.cos(math.pi*freq*(k-(stay_timeA[i]+stay_timeB[i]*2+interpolation_time)) / interpolation_time)
                        # if i % 100==0 and k%100==0 : print("後半部分のサインカープ",i,k,now_point)
                time_array[k,i]+=int(now_point)
        gapinout=(time_array[0]-time_array[-1])-slide_time
        gapstart=slide_time-time_array[0]
        print("Slide_timeの計算後",time_array[0,0],time_array[-1,0],gapinout[0],gapstart[0])
        # for k in range(time_array.shape[0]):
        #     time_array[k]+=gapinout*int(k/time_array.shape[0])+gapstart
        # print("Gap計算後",time_array[0,0],time_array[-1,0],gapinout[0],gapstart[0])
        self.data[:,:,1] = time_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1])

    # 指定したスリットの時間の流れを指定した時間に固定する。
    def applyTimeClip(self,trackslit:int,cliptime=None):
        print(sys._getframe().f_code.co_name)
        if cliptime != None : 
            timegap = cliptime - self.data[0,trackslit,1]
        else:
            timegap = 0 - self.data[0,trackslit,1]
        now_point = 0
        self.data[0,:,1]+=timegap 
        for k in range(1,self.data[:,:,1].shape[0]):
            now_point = self.data[k-1,trackslit,1] - self.data[k,trackslit,1]
            self.data[k,:,1]+=now_point
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(trackslit)+"-"+str(cliptime))

    # スリットの空間位置に応じて、最大`v`で指定したフレーム数分、時間方向へずらす。mean_mode=1でself.cycle_axisを参照する。mean_mode=2で、スリット空間位置の平均に対して計算する。2023.10.10added
    def applyTimebySpace(self,v,mode=0):
        print(sys._getframe().f_code.co_name)
        for k in range(0,int(self.data.shape[0])):#Xは固定でzポイントが飛び飛びから詰まっていく。
            # self.data[k,:,1] += v*(self.data[k,:,0]/self.scan_nums)
            if mode == 0 :
                self.data[k,:,1] = self.data[k,:,1] + v*(self.data[k,:,0]/self.scan_nums)  
            elif mode == 1:
                self.data[k,:,1] = self.data[k,:,1] + v*(self.cycle_axis[k]/self.scan_nums)  
            elif mode == 2:
                self.data[k,:,1] = self.data[k,:,1] + v*(np.mean(self.data[k,:,0])/self.scan_nums) 
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(v))
    
    # スリットの空間位置に応じて、キーフレームで指定したフレーム数分、時間方向へずらす。mean_mode=1でself.cycle_axisを参照する。mean_mode=2で、スリット空間位置の平均に対して計算する。2023.10.10added
    def applyTimebyKeyframetoSpace(self,keyframes=[],mode=0):
        print(sys._getframe().f_code.co_name)
        if len(keyframes)==0:return
        for k in range(0,int(self.data.shape[0])):#Xは固定でzポイントが飛び飛びから詰まっていく。
            # self.data[k,:,1] += v*(self.data[k,:,0]/self.scan_nums)
            if mode == 0 :
                self.data[k,:,1] = self.data[k,:,1] + self.spline_interpolate(self.data[k,:,0], keyframes, 'spline')
            elif mode == 1:
                self.data[k,:,1] =  self.data[k,:,1] + self.spline_interpolate(self.cycle_axis[k], keyframes, 'spline')
                # self.data[k,:,1] = self.data[k,:,1] + v*(self.cycle_axis[k]/self.scan_nums)  
            elif mode == 2:
                self.data[k,:,1] = self.data[k,:,1] + self.spline_interpolate(np.mean(self.data[k,:,0]), keyframes, 'spline')
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1])
    

    def applySpaceFlip(self):
        print(sys._getframe().f_code.co_name)
        self.data[:, :, 0] = self.scan_nums - 1 - self.data[:, :, 0]
        self.maneuver_log((sys._getframe().f_code.co_name))
    
    def applySpaceFlat(self):
        """
        self.data[:,:,0] の空間成分（XまたはY軸）を初期状態の連番にリセットする
        """
        print(sys._getframe().f_code.co_name)
        # 初期の空間フレーム番号（0〜self.scan_nums-1）を生成
        normalFrame = np.arange(0, self.scan_nums)
        # self.data の shape に合わせてブロードキャスト（行数分だけコピー）
        self.data[:,:,0] = np.tile(normalFrame, (self.data.shape[0], 1))
        self.maneuver_log((sys._getframe().f_code.co_name))


