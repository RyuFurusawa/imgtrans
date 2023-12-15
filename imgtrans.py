import csv
import gc
import glob
import math
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
import psutil
import easing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d #3dプロットに必要
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import inspect
from typing import List
import librosa
import librosa.display
import pickle #transprocess_typeBにてタプル型の保存に必要
from scipy.interpolate import splrep, splev, interp1d #スプライン補完で使用
from scipy.special import comb
import re #正規表現（Regular Expressions）を扱うためのモジュール
import shutil #特定のボリューム（ドライブ）の残りデータ容量を調べる
# import numexpr as ne

class drawManeuver:
    imgtype = ".jpg" #.bmp or .jpg
    img_size_type = 0 #0:hw->hw 1:hw->w,w*2 2:総フレーム数分 3: square 
    outfps = 30
    recfps = 120
    progressbarsize=50
    sepVideoOut = 0 # セパレートしない場合、rawでnpアレイファイルをテンポファイルとしてハードディスクに貯めておき、全てのアレイが準備できてからレンダリングする。そのためHD容量を100GBとか普通に食う。
    memory_percent = 60 #%
    auto_visualize_out = True
    default_debugmode = False
    audio_form_out = False
    embedHistory_intoName = True

    def __init__(self,videopath:str,sd:bool,outdir:str=None,datapath:str=None,foldername_attr:str=None):
        self.VIDEO_PATH = videopath
        self.ORG_NAME= videopath.split(".")[0].rsplit("/",1)[-1]
        self.ORG_PATH=videopath.split(".")[0].rsplit("/",1)[0] if outdir==None else outdir
        # self.ORG_FNAME= videopath.split(".")[0].rsplit("/",1)[0].rsplit("/",1)[-1]
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
            self.recfps = self.cap.get(cv2.CAP_PROP_FPS)# fps
        self.data = [] if datapath == None else  np.load(datapath)
        self.scan_direction=sd 
        self.scan_nums = int(self.width) if sd % 2 == 1 else int(self.height)
        self.slit_length = int(self.height) if sd % 2 == 1 else int(self.width)
        self.sc_resetPositionMap=[]
        self.sc_rateMap = []
        self.sc_inPanMap = []
        self.sc_now_depth = []
        self.out_videopath = ""
        sd_attr = "Hslit" if sd==0 else "Vslit" #Vertical or horizontal
        self.out_name_attr =datetime.now().strftime('%Y_%m%d')+"_"+sd_attr 
        if foldername_attr != None : self.out_name_attr+="_"+foldername_attr
        self.log = 0
        self.infolog = 0
        self.sc_FNAME =self.ORG_NAME + ".AIFF"
        self.cycle_axis=[]
        #ディレクトリ作成、そのディデクトリに移動
        NPATH = self.ORG_PATH+"/"+self.ORG_NAME+"_"+self.out_name_attr 
        if os.path.isdir(NPATH)==False:
            os.makedirs(NPATH)
        os.chdir(NPATH)
        # 現在の日時を取得
        now = datetime.now()
        # 日、時間、分の形式で文字列に変換
        log_entry = now.strftime('%Y-%m-%d %H:%M')
        append_to_logfile("-")
        append_to_logfile(self.ORG_NAME+"_"+ sd_attr)
        append_to_logfile(log_entry)

    #added 2023 10/6
    def maneuver_log(self,module_name_attr):
        self.log+=1
        self.out_name_attr+="_"+module_name_attr
        if len(self.out_name_attr) > 100:#文字数が255を超えるとエラーとなる問題
            self.out_name_attr = self.out_name_attr[:97] + "..."+str(self.log)
        append_to_logfile(str(self.log)+":"+module_name_attr)
        if self.auto_visualize_out : 
            if self.default_debugmode : self.maneuver_2dplot(debugmode=True)
            else : self.maneuver_2dplot()

    def append(self,maneuver,auto_zslide=True,zslide=0):
        permit_auto_zslide=False
        if zslide == 0:
            if auto_zslide and len(self.data)>0:
                permit_auto_zslide=True
        if permit_auto_zslide : zslide = maneuver[0,:,1]-self.data[-1,:,1]
        maneuver[:,:,1]=maneuver[:,:,1]-zslide
        self.data = np.vstack((self.data,maneuver))
        self.maneuver_log(sys._getframe().f_code.co_name)
    
    #added 2023 11/25
    def wide_expand(self,add_size=300,sclip=True,zclip=True):
        # 2次元目にパディングを適用（左右に均等にパディングを追加）
        new_array = np.pad(self.data, ((0, 0), (add_size, add_size), (0, 0)), mode='constant')
        for i in range(self.data.shape[0]):
            ls_diff = self.data[i,0,0]-self.data[i,1,0]
            lz_diff = self.data[i,0,1]-self.data[i,1,1]
            rs_diff = self.data[i,-1,0]-self.data[i,-2,0]
            rz_diff = self.data[i,-1,1]-self.data[i,-2,1]
            # print(ls_diff,lz_diff,rs_diff,rz_diff)
            for k in range(add_size):
                new_array[i,add_size-k-1,0]=new_array[i,add_size-k,0]+ls_diff
                new_array[i,add_size-k-1,1]=new_array[i,add_size-k,1]+lz_diff
                new_array[i,add_size+self.data.shape[1]+k,0]=new_array[i,add_size+self.data.shape[1]+k-1,0]+rs_diff
                new_array[i,add_size+self.data.shape[1]+k,1]=new_array[i,add_size+self.data.shape[1]+k-1,1]+rz_diff
                # print(new_array[i,add_size-k,1],new_array[i,add_size+self.data.shape[1]+k,1])
        if sclip : new_array[:,:,0] = np.clip(new_array[:,:,0],0,self.scan_nums-1)
        if zclip : new_array[:,:,1] = np.clip(new_array[:,:,1],1,self.count-1)
        self.data = new_array
        self.maneuver_log(sys._getframe().f_code.co_name)

    #added 2023 11/24 DATaの鏡面反転
    def arrayReflection(self):
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
        self.data = self.data[start:end,:,:]
        self.maneuver_log(sys._getframe().f_code.co_name)

    #added 2023 11/24 
    def applyLoopBlur(self,sblur,tblur):
        array=self.data
        self.data= np.vstack((np.vstack((array,self.data)),array))
        if sblur > 0 : self.data[:,:,0]=cv2.blur(self.data[:,:,0],(1,int(sblur)))
        if tblur > 0 : self.data[:,:,1]=cv2.blur(self.data[:,:,1],(1,int(tblur)))
        self.data = self.data[array.shape[0]:array.shape[0]+array.shape[0],:,:]
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+"s"+str(sblur)+"b"+str(tblur))
    
    #applyCustomeBlurの埋め込み
    def applyConnectLoopBlur(self,sblur,tblur,connect_frame=100):
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
        if tblur > 0 :
            self.data = custom_blur(self.data,s_frame=point_frame-range_frame, e_frame=point_frame+range_frame, bl_time = tblur, dim_num=1)
        if sblur > 0 :
            self.data = custom_blur(self.data,s_frame=point_frame-range_frame, e_frame=point_frame+range_frame, bl_time = sblur, dim_num=0)
        
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+"s"+str(sblur)+"b"+str(tblur))
    
    #added 2023 10/22
    #指定したフレーム数で保管する。保管するz距離に応じで速度が変わる。
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
    


    # グローバル関数”img_size_type”に応じて、出力するイメージのサイズ（縦、横）を返す。
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
                apply_bl_time=int(bl_time/2)
                # print("BlurProcess:if:",y,apply_bl_time)
            else:
                apply_bl_time= y+1 if y < bl_time/2 else (e_frame-s_frame)-y
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
                if weightmean:
                    blur_array[y,x]= (time_array[s_frame+y,x]*wA + np.mean(time_array[int(s_frame+y-apply_bl_time):int(s_frame+y+apply_bl_time),x])*wB) / (wA+wB)
                else : blur_array[y,x]=np.mean(time_array[int(s_frame+y-apply_bl_time):int(s_frame+y+apply_bl_time),x])
                # blur_array[y,x]=np.mean(time_array[int(s_frame+y-apply_bl_time):int(s_frame+y+apply_bl_time)+1,x])
        time_array[s_frame:e_frame,:]=blur_array
        print(time_array.shape)
        self.data[:,:,dim_num]=time_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(bl_time))
        
    
    # 時間軸、空間軸、ともに変化のないフラットな配列を”frame_nums”で指定されたフレーム数分生成。
    def addFlat(self,frame_nums,z_pos=0,z_autofit=True,prepend=False):
        print(sys._getframe().f_code.co_name)
        extra_array = np.zeros((frame_nums,self.scan_nums,2),dtype=np.float64)
        normalFrame=np.arange(0,self.scan_nums)
        if len(self.data) == 0 : #replace
            zFrame=np.full(self.scan_nums,z_pos)
            for i in range(frame_nums):
                extra_array[i,:,0]=normalFrame
                if z_pos!=0 : 
                    extra_array[i,:,0]=zFrame
            self.data = extra_array
        else : 
            if z_autofit:
                z_pos = int(np.mean(self.data[-1,:,0])) if prepend == False else int(np.mean(self.data[0,:,1]))
            zFrame=np.full(self.scan_nums,z_pos)
            for i in range(frame_nums):
                extra_array[i,:,0]=normalFrame
                extra_array[i,:,0]=zFrame
            self.data = np.vstack((self.data,extra_array)) if prepend == False else np.vstack((extra_array,self.data)) 
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1]+str(frame_nums))

    # 時間方向の断面配列を”frame_nums”で指定されたフレーム数分生成。"xypoint"は0−1の範囲で指定
    def addSlicePlane(self,frame_nums,xypoint=0.5):
        print(sys._getframe().f_code.co_name)
        extra_array = np.zeros((frame_nums,self.scan_nums,2),dtype=np.float64)
        normalFrame=np.arange(0,self.scan_nums)
        xyFrame=np.full(self.scan_nums,int(self.scan_nums*xypoint))
        for i in range(frame_nums):
            extra_array[i,:,0]=xyFrame
            extra_array[i,:,1]=normalFrame
        if len(self.data) == 0 : #replace
            self.data = extra_array
        else : 
            print("error")
            return
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1]+str(frame_nums))

    # 時間軸、空間軸、ともに変化のないフラットな配列を”frame_nums”で指定されたフレーム数分生成。
    def applyTimeOblique(self,maxgap):
        print(sys._getframe().f_code.co_name)
        for i in range(self.scan_nums):
            self.data[:,i,1] += maxgap * i/self.scan_nums 
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1])

    # 時間軸、空間軸、ともに最終列の配列を”frame_nums”で指定されたフレーム数分生成して加える。
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
    def addExtend(self,addframe:int):
        print(sys._getframe().f_code.co_name)
        extra_array = np.zeros((self.data.shape[0]+addframe,self.data.shape[1],2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
        outroFrame = self.data[-1,:,:]
        extra_array[:self.data.shape[0]] = self.data
        for i in range(addframe):
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
    def timeFlowKeepingExtend(self, frame_nums: int, fade: bool = False, intro: bool = True, outro: bool = True,fade_speed=0,fade_type="inout"):
        print(sys._getframe().f_code.co_name)
        if intro == False and outro == False: return

        if intro == False or outro == False:
            extra_array = np.zeros((self.data.shape[0] + frame_nums, self.data.shape[1], 2), dtype=np.float64)    
        else:
            extra_array = np.zeros((self.data.shape[0] + frame_nums * 2, self.data.shape[1], 2), dtype=np.float64)
        xyfirstFrame = self.data[0, :, 0]
        xylastFrame = self.data[-1, :, 0]
        zfirstDiff = self.data[1, :, 1] - self.data[0, :, 1]
        zlastDiff = self.data[-2, :, 1] - self.data[-1, :, 1]
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
            if intro:
                for n in range(introEaseArray.shape[0]):
                    if fade_type=="inout":
                        introEaseArray[n] = easing.inOutQuad(n, zfirstDiff, -fade_speed_array-zfirstDiff, introEaseArray.shape[0])
                    elif fade_type=="in":
                        introEaseArray[n] = easing.inQuad(n, zfirstDiff, -fade_speed_array-zfirstDiff, introEaseArray.shape[0])
                    elif fade_type=="out":
                        introEaseArray[n] = easing.outQuad(n, zfirstDiff, -fade_speed_array-zfirstDiff, introEaseArray.shape[0])
            if outro:
                for n in range(outroEaseArray.shape[0]):
                    if fade_type=="inout":
                        outroEaseArray[n] = easing.inOutQuad(n, zlastDiff, fade_speed_array-zlastDiff, outroEaseArray.shape[0])
                    elif fade_type=="in":
                        outroEaseArray[n] = easing.inOutQuad(n, zlastDiff, fade_speed_array-zlastDiff, outroEaseArray.shape[0])
                    elif fade_type=="out":
                        outroEaseArray[n] = easing.inOutQuad(n, zlastDiff, fade_speed_array-zlastDiff, outroEaseArray.shape[0])

        for i in range(frame_nums):
            if outro:
                extra_array[copy_end + i, :, 0] = xylastFrame
                if fade:
                    extra_array[copy_end + i, :, 1] = extra_array[copy_end + i - 1, :, 1] - outroEaseArray[i]
                else:
                    extra_array[copy_end + i, :, 1] = extra_array[copy_end + i - 1, :, 1] - zlastDiff
            if intro:
                extra_array[frame_nums - i - 1, :, 0] = xyfirstFrame
                if fade:
                    extra_array[frame_nums - i - 1, :, 1] = extra_array[frame_nums - i, :, 1] - introEaseArray[i]
                else:
                    extra_array[frame_nums - i - 1, :, 1] = extra_array[frame_nums - i, :, 1] - zfirstDiff
        return extra_array


    # 与えた軌道配列に、延長させたフレームをプリペンド、アペンドする。XYフレームそれぞれ最終フレームと最初のフレームと同じデータで延長させる。Z(アウト時間）に関しては最終の変化量を維持して延長させる。fade引数をTrueでスピード０に落ち着かせる。
    def applyTimeFlowKeepingExtend(self,frame_nums:int,fade:bool=False,intro: bool = True, outro: bool = True,fade_speed=0,fade_type="inout"):
        print(sys._getframe().f_code.co_name)
        self.data = self.timeFlowKeepingExtend(frame_nums,fade,intro,outro,fade_speed,fade_type)
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1])
            
    #配列全体に時間の順方向の流れ（単位はslide_time）を付与する。
    def applyTimeForward(self,slide_time=None):
        print(sys._getframe().f_code.co_name)
        if slide_time==None:slide_time=self.outfps/self.recfps
        for k in range(self.data[:,:,1].shape[0]):
            self.data[k,:,1] += slide_time*k
        print("Slide_timeの計算後 min-max =",np.amin(self.data[:,:,-1]),np.amax(self.data[:,:,-1]))
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(slide_time))
    
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

    def applyTimeChoppyLoopB(self, slide_time=None, frequency=1, phase_shift=0, rise=0.5, fall=0.5, wave_type='triangle'):
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
        elif wave_type == 'sine':
             # サイン波の生成
        # 振幅のスケーリング (rise と fall の平均値/ 2を使用)
            # amplitude_scale = (rise + fall) / 2
            wave = np.sin(t_shifted)/ 2
        # 波形のスケーリングと適用
        scaled_wave = (wave + 1) / 2 * (N / 2 * slide_time)
        scaled_wave_expanded = np.tile(scaled_wave, (self.data.shape[1], 1)).T
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
            time_array[k]+=int(now_point)
        gapinout=(time_array[0]-time_array[-1])-slide_time
        gapstart=slide_time-time_array[0]
        print("Slide_timeの計算後",time_array[0,0],time_array[-1,0],gapinout[0],gapstart[0])
        for k in range(time_array.shape[0]):
            time_array[k]+=gapinout*int(k/time_array.shape[0])+gapstart
        print("Gap計算後",time_array[0,0],time_array[-1,0],gapinout[0],gapstart[0])
        self.data[:,:,1] = time_array
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1])
    
    #gスリットごとにZGAPの差がある。
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
    def zCenterArange(self):
        print(sys._getframe().f_code.co_name)
        ediff = self.count-np.amax(self.data[:,:,1])
        idiff = np.amin(self.data[:,:,1])
        if (ediff + idiff) > 0 :
            self.data[:,:,1] -= idiff #0基準に戻す
            self.data[:,:,1] += int((idiff+ediff)/2)
            self.maneuver_log("zCenterAranged")
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
                print("zp range-調整後:",np.amin(self.data[:,:,1]),np.amax(self.data[:,:,1]))
                add_attr+="_timeSlide"+str(int(add_frame))+"f"
            if np.amax(self.data[:,:,1]) > (self.count-subtract_count):
                diffframe=np.amax(self.data[:,:,1])-(self.count-subtract_count)
                if diffframe < np.amin(self.data[:,:,1]):
                    add_frame = diffframe*-1
                    self.data[:,:,1] += add_frame
                    print("zp range-調整後:",np.amin(self.data[:,:,1]),np.amax(self.data[:,:,1]))
                    add_attr+="_timeSlide"+str(int(add_frame))+"f"
                else:
                    self.data[:,:,1] += np.amin(self.data[:,:,1])*-1
                    scale_rate=(self.count-subtract_count) /np.amax(self.data[:,:,1])
                    self.data[:,:,1]=self.data[:,:,1]*scale_rate
                    add_attr+="_timeScaling"+str(scale_rate)
            self.maneuver_log(add_attr)
        else : print("check ok!!!")

     #一番初めのフレームの中心のスリットの参照時間を、指定した時間にセットする。それに合わせて全体に対してスライドさせて調節する。baseframe＝ー1で最終フレームを軸とする
    def applyTimeSlide(self,setframe:int,baseframe:int=0):    
        print(sys._getframe().f_code.co_name)
        deff = setframe - self.data[baseframe,int(self.scan_nums/2),1]
        self.data[:,:,1]+=deff
        print("zp range-調整後:",np.amin(self.data[:,:,1]),np.amax(self.data[:,:,1]))
        self.maneuver_log((sys._getframe().f_code.co_name).split("apply")[1]+str(baseframe)+"->"+str(setframe))
        
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

     #指定したZアレイと最初フレームの差分を計算して、差分があれば、差分を埋め合わせるように、全てのフレームにたいして、調整する。
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
    def applyOutFix(self,target_z_array):
        print(sys._getframe().f_code.co_name)
        gap=(self.data[-1,:,1]-target_z_array)
        if abs(np.amax(gap)) > 1:
            print("Gapあり",np.amax(gap))
            for k in range(self.data.shape[0]):
                self.data[k,:,1]-=gap*(k/self.data.shape[0])
        self.maneuver_log(sys._getframe().f_code.co_name)


    def maneuver_CSV_out(self,thread_num=None):
        print(sys._getframe().f_code.co_name)
        if thread_num != None : self.info_setting(thread_num)
        else:
            if self.infolog == 0 :self.info_setting() 
            elif self.infolog != self.log :self.info_setting(self.sc_resetPositionMap.shape[0])
        print(sys._getframe().f_code.co_name)
        j=self.sc_resetPositionMap.shape[0]
        #AEでの描画よう
        np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_ResetP_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_resetPositionMap),delimiter=',')
        np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_Rate_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_rateMap),delimiter=',')
        np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_inPanMap_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_inPanMap),delimiter=',')
        # np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_nowDepth_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_now_depth),delimiter=',')
        
        # 表計算ソフトで作図用
        # np.savetxt(self.ORG_NAME+"_"+self.out_name_attr+'_ResetP-transpose.csv',self.sc_resetPositionMap.transpose(),delimiter=',')
        # np.savetxt(self.ORG_NAME+"_"+self.out_name_attr+'_Rate-transpose.csv',self.sc_rateMap.transpose(),delimiter=',')
        # np.savetxt(self.ORG_NAME+"_"+self.out_name_attr+'_inPanMap-transpose.csv',self.sc_inPanMap.transpose(),delimiter=',')
        
    def info_setting(self,thread_num=20):
        print("info_setting,分割:",np.ceil(self.data.shape[1]/(thread_num-1)).astype(int))
        self.sc_resetPositionMap = np.vstack((self.data[:,::np.ceil(self.data.shape[1]/(thread_num-1)).astype(int),1].transpose(),self.data[:,-1,1]))#transpose()で次元入れ替え、"[::a]"で間引き
        self.sc_rateMap = np.zeros(self.sc_resetPositionMap.shape,np.float32)#再生レートの配列
        self.sc_inPanMap = np.vstack((self.data[:,::np.ceil(self.data.shape[1]/(thread_num-1)).astype(int),0].transpose(),self.data[:,-1,0]))#入力panの配列
        #実験用。リバーブなどに適応することをイメージ
        self.sc_now_depth =np.zeros(self.data.shape[0],np.float32)#フレーム内に収まる時間の幅
        for i in range(self.sc_resetPositionMap.shape[0]):
            for k in range(self.sc_resetPositionMap.shape[1]):
                if((k+1)<self.sc_resetPositionMap.shape[1]):
                    #前後の差分から変化率を計測して再生レートを計算
                    self.sc_rateMap[i,k]=(self.sc_resetPositionMap[i,k+1]-self.sc_resetPositionMap[i,k])/(self.recfps/self.outfps)
                else:
                    self.sc_rateMap[i,k]=self.sc_rateMap[i,k-1]
        #実験用。リバーブなどに適応することをイメージ
        for i in range(self.sc_now_depth.shape[0]):
            self.sc_now_depth[i]=abs(np.amax(self.data[i,:,1])-np.amin(self.data[i,:,1]))
        self.infolog = self.log
    
    # supercolliderに読み込ませるサウンドプロセスコードを出力する。
    def scd_out(self,thread_num=None,audio_path=None):
        print(sys._getframe().f_code.co_name)
        if audio_path == None:
            audio_path = self.ORG_PATH+"/"+self.sc_FNAME
        if thread_num != None : self.info_setting(thread_num)
        else:
            if self.infolog == 0 :self.info_setting() 
            elif self.infolog != self.log :self.info_setting(self.sc_resetPositionMap.shape[0])
        j=self.sc_resetPositionMap.shape[0]
        if self.audio_form_out:
            # 音声データの読み込み
            start_sec=np.min(self.data[:,:,1])/self.recfps
            duration_sec=(np.max(self.data[:,:,1])-np.min(self.data[:,:,1]))/self.recfps
            y, sr = librosa.load(audio_path,offset=start_sec,duration=duration_sec)
            fig,ax= plt.subplots(figsize=(8, 2),tight_layout=True)

            librosa.display.waveshow(y, sr=sr,color='#777777',offset=start_sec)
            plt.xlabel('')
            ax.set_facecolor('none')  # 背景を透明に設定
            ax.yaxis.set_tick_params(color='none',labelcolor='none')
            ax.xaxis.set_tick_params(labelsize=9,color="none",labelcolor="#777777")         
            # 外枠の色を変更
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.savefig('waveform.png',transparent=True)  # 画像として保存
            # plt.show()
            plt.clf()

        #csvで出力する。
        csvoutstr = self.ORG_NAME+"_"+self.out_name_attr 
        if self.embedHistory_intoName == False :
            csvoutstr = self.ORG_NAME+"_process"+str(self.log)
        np.savetxt(csvoutstr +'_ResetP_'+str(j)+'thread.csv',self.sc_resetPositionMap,delimiter=',')
        np.savetxt(csvoutstr +'_Rate_'+str(j)+'thread.csv',self.sc_rateMap,delimiter=',')
        np.savetxt(csvoutstr +'_inPanMap_'+str(j)+'thread.csv',self.sc_inPanMap,delimiter=',')
        np.savetxt(csvoutstr +'_nowDepth_'+str(j)+'thread.csv',self.sc_now_depth,delimiter=',')
        NOW_DIR=os.getcwd()
        ORG_PATH=NOW_DIR.rsplit("/",1)[0]
        sc_load_str ='//映像の3次元マップ情報を読み込む\r r=["'+NOW_DIR+"/"+csvoutstr +'_ResetP_'+str(j)+'thread.csv",\r"'+NOW_DIR+"/"+csvoutstr +'_Rate_'+str(j)+'thread.csv",\r"'+NOW_DIR+"/"+csvoutstr +'_inPanMap_'+str(j)+'thread.csv",\r"'+NOW_DIR+"/"+csvoutstr +'_nowDepth_'+str(j)+'thread.csv",\r"'+audio_path+'"];\r'
        sc_simplePlay_setting_str = '(//AudioSetting - Rewrite it to suit your environment- \nd=ServerOptions.devices[4];//Run "ServerOptions.devices" to see the list of devices and rewrite this number.\nServer.default.options.outDevice_(d);\ns.options.numOutputBusChannels=2;\ns.boot;\ns.meter;\ns.plotTree;\ns.scope;\n\nSynthDef(\simplePlay, {\narg bufL = 0 , bufR = 1,trig = 0, rate=1, pan=0.5, inpan=0.5,resetp=1000,amp=0.1,lag=(1/'+str(self.outfps)+'),out=0,v=0.3;\n var sigL,sigR;\n sigL = PlayBuf.ar(1,bufL, BufRateScale.kr(bufL) *  Lag2UD.kr(rate,lag,lag), trigger:Impulse.kr(trig), startPos:resetp*BufSampleRate.kr(bufL), loop:1);\n    sigR = PlayBuf.ar(1,bufR, BufRateScale.kr(bufR) *  Lag2UD.kr(rate,lag,lag), trigger:Impulse.kr(trig), startPos:resetp*BufSampleRate.kr(bufR), loop:1); \n //2ch out \n  Out.ar(out, sigL * [1*pan,1*(1-pan)]* Lag2UD.kr(inpan,lag,lag)*amp); \n Out.ar(out, sigR * [1*(1-pan),1*pan]* Lag2UD.kr(inpan,lag,lag)*amp);\n }).add;\n\n'
        sc_grain_setting_str = '(//AudioSetting - Rewrite it to suit your environment- \nd=ServerOptions.devices[4];//Run "ServerOptions.devices" to see the list of devices and rewrite this number.\nServer.default.options.outDevice_(d)\ns.options.numOutputBusChannels=2;\ns.boot;\ns.meter;\ns.plotTree;\ns.scope;\n\n//////////////////////////////////\nSynthDef.new("bufgrainLR",{\n    arg bufL = 0 , bufR = 1,out=0,rate=1,dur=0.1,trig=0,resetp=0.1,pan=0.5,inpan=0.5,ampl=1,ampr=1,amp=1,lag=(1/'+str(self.outfps)+'),v=0.3;\n        var sigL,sigR;\n    sigL = GrainBuf.ar(\n        2,\n        trigger:Impulse.ar('+str(self.outfps)+'),\n        dur:dur,\n        sndbuf:bufL,\n        rate:1,//MouseX.kr(0.5,2.1)\n        pos:(Phasor.ar(Impulse.ar(trig),Lag2UD.kr(rate,lag,lag)*BufRateScale.ir(bufL),0,BufSamples.ir(bufL),resetp*BufSamples.ir(bufL))) / BufSamples.ir(bufL), //(Phasor.ar(0,MouseY.kr(0.1,2,1)*BufRateScale.ir(b),0,BufSamples.ir(b)-1)+LFNoise1.ar(100).bipolar(0.0*SampleRate.ir)) / BufSamples.ir(b)\n        interp:2,\n        pan:0,\n        envbufnum:-1,\n        maxGrains:512);\n    sigR = GrainBuf.ar(\n        2,\n        trigger:Impulse.ar('+str(self.outfps)+'),\n        dur:dur,\n        sndbuf:bufR,\n        rate:1,//MouseX.kr(0.5,2.1)\n        pos:(Phasor.ar(Impulse.ar(trig),Lag2UD.kr(rate,lag,lag)*BufRateScale.ir(bufR),0,BufSamples.ir(bufR),resetp*BufSamples.ir(bufR))) / BufSamples.ir(bufR), //(Phasor.ar(0,MouseY.kr(0.1,2,1)*BufRateScale.ir(b),0,BufSamples.ir(b)-1)+LFNoise1.ar(100).bipolar(0.0*SampleRate.ir)) / BufSamples.ir(b)\n        interp:2,\n        pan:0,\n        envbufnum:-1,\n        maxGrains:512);\n    //2ch out\n    Out.ar(out, sigL * [1*pan,1*(1-pan)]* Lag2UD.kr(inpan,lag,lag)*amp);\n    Out.ar(out, sigR * [1*(1-pan),1*pan]* Lag2UD.kr(inpan,lag,lag)*amp);\n    //4ch out\n    // Out.ar(out, sigL * max(0,(1-[abs(1-pan),abs(2/3-pan),abs(1/3-pan),abs(0-p)]-v) / v)* Lag2UD.kr(inpan,lag,lag)*amp);\n    // Out.ar(out, sigR * max(0,(1-[abs(1-pan),abs(2/3-pan),abs(1/3-pan),abs(0-p)]-v) / v)* Lag2UD.kr(inpan,lag,lag)*amp);\n}).add;\n//////////////////////////////////'
        sc_rev_setting_str = '//////////////////////////////////\nSynthDef("preDelay", {\n    ReplaceOut.ar(\n        6,\n        DelayN.ar(In.ar(4, 2), 0.048, 0.048)\n    )\n}).add;\n\nSynthDef("combs", { arg dtime=0.1;\n    ReplaceOut.ar(\n        8,\n        Mix.arFill(7, { CombL.ar(In.ar(6, 2), dtime, LFNoise1.kr(Rand(0, dtime), 0.04, 0.05), 15) })\n    )\n}).add;\n\nSynthDef("allpass", { arg gain = 0.2;\n    var source;\n    source = In.ar(8, 2);\n    4.do({ source = AllpassN.ar(source, 0.050, [Rand(0, 0.05), Rand(0, 0.05)], 1) });\n    ReplaceOut.ar(\n        10,\n        source * gain\n    )\n}).add;\n\nSynthDef("theMixer", { arg gain = 1,dry=1.0;\n    ReplaceOut.ar(\n        0,\n        Mix.ar([In.ar(4, 2)*dry, In.ar(10, 2)]) * gain\n    )\n}).add;\n//////////////////////////////////'
        #'W'モードで書き出し。 同名のファイルが存在する場合にもファイルの内容が上書きされます。それを避ける場合は"x"。'x' モードは、ファイルが存在しない場合にのみ書き込みを行い、既存のファイルがある場合はエラーを発生させます。
        with open(csvoutstr +'SC_Grain-'+str(j)+'voices.scd','w') as f:
            f.write(sc_grain_setting_str)
            f.write(sc_load_str)
            f.write('\n')
            f.write('x = CSVFileReader.readInterpret(r[0]).postcs;//ResetP \ry = CSVFileReader.readInterpret(r[1]).postcs;//Rate \rz = CSVFileReader.readInterpret(r[2]).postcs;//inPanMap \rc = CSVFileReader.readInterpret(r[3]).postcs;//nowdepth')
            f.write('\n')
            f.write('a = Buffer.readChannel(s,r[4],channels:[1]);\rb = Buffer.readChannel(s,r[4],channels:[0]);\r)')
            f.write('\n')
            f.write('(\r//Grainチャンネルミックスをrecording, LR分析バージョン')
            f.write('\n')
            # f.write('"open -a \'QuickTime Player\'".unixCmd;')
            f.write('"open -a \'QuickTime Player\' \''+NOW_DIR+"/"+csvoutstr +'.mp4'+'\' ".unixCmd;')
            f.write('\n')
            f.write('s.record("'+ NOW_DIR + "/" + self.ORG_NAME + "_" + self.out_name_attr +'_Grain-'+ str(j) + 'voices.aiff");')
            f.write('\n')
            f.write('fork{var cha='+str(j)+',num=x[0].size,recfps='+str(self.recfps)+',trig=0.333')
            for i in range (self.sc_resetPositionMap.shape[0]):f.write(',gs'+str(i))
            f.write(';')
            f.write('\n')
            for i in range (self.sc_resetPositionMap.shape[0]):
                if i == 0 :
                    f.write('    gs'+str(i)+' = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",0,"amp",1/cha,"resetp",x['+str(i)+'].[0]/((b.numFrames / b.sampleRate)*recfps)]);\r')
                elif i == self.sc_resetPositionMap.shape[0]-1:
                    f.write('    gs'+str(i)+' = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",1,"amp",1/cha,"resetp",x['+str(i)+'].[0]/((b.numFrames / b.sampleRate)*recfps)]);\r')
                else:
                    f.write('    gs'+str(i)+' = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",'+str(i/(j-i))+',"amp",1/cha,"resetp",x['+str(i)+'].[0]/((b.numFrames / b.sampleRate)*recfps)]);\r')
            f.write('    num.do{arg i;\r        i.post;"/".post;num.postln;\r        (1/'+str(self.outfps)+').yield;\r')
            for i in range (self.sc_resetPositionMap.shape[0]):
                f.write('\r        gs'+str(i)+'.set("resetp",x['+str(i)+'].[i]/((b.numFrames / b.sampleRate)*recfps));\r')
                f.write('        gs'+str(i)+'.set("rate",y['+str(i)+'].[i]);\r')
                f.write('        gs'+str(i)+'.set("inpan",z['+str(i)+'].[i]/3840);\r')
            f.write('        };\r    };\r)')

        with open(csvoutstr +'SC_Play-'+str(j)+'voices.scd','w') as f:
            f.write(sc_simplePlay_setting_str)
            f.write('\n')
            f.write(sc_load_str)
            f.write('\n')
            f.write('x = CSVFileReader.readInterpret(r[0]).postcs;//ResetP \ry = CSVFileReader.readInterpret(r[1]).postcs;//Rate \rz = CSVFileReader.readInterpret(r[2]).postcs;//inPanMap \rc = CSVFileReader.readInterpret(r[3]).postcs;//nowdepth')
            f.write('\n')
            f.write('a = Buffer.readChannel(s,r[4],channels:[1]);\rb = Buffer.readChannel(s,r[4],channels:[0]);\r)')
            f.write('\n')
            f.write('(\r// PLAY BUF, LR分析バージョン')
            f.write('\n')
            f.write('"open -a \'QuickTime Player\' \''+NOW_DIR+"/"+csvoutstr +'.mp4'+'\' ".unixCmd;')
            f.write('\n')
            f.write('s.record("'+ NOW_DIR + "/" + self.ORG_NAME + "_" + self.out_name_attr +'_Play-'+ str(j) + 'voices.aiff");')
            f.write('\n')
            f.write('fork{\r    var cha='+str(j)+',num=x[0].size,fps='+str(self.recfps)+',trig=1')
            for i in range (self.sc_resetPositionMap.shape[0]):f.write(',gs'+str(i))
            f.write(';\n')
            for i in range (self.sc_resetPositionMap.shape[0]):
                if i == 0 :
                    f.write('    gs'+str(i)+' = Synth("simplePlay",["bufL",a,"bufR",b,"rate",1,"trig",trig+rand(0.05),"pan",0,"amp",1/cha,"resetp",x['+str(i)+'].[0]/fps]);\r')
                elif i == self.sc_resetPositionMap.shape[0]-1:
                    f.write('    gs'+str(i)+' = Synth("simplePlay",["bufL",a,"bufR",b,"rate",1,"trig",trig+rand(0.05),"pan",1,"amp",1/cha,"resetp",x['+str(i)+'].[0]/fps]);\r')
                else:
                    f.write('    gs'+str(i)+' = Synth("simplePlay",["bufL",a,"bufR",b,"rate",1,"trig",trig+rand(0.05),"pan",'+str(i/(j-1))+',"amp",1/cha,"resetp",x['+str(i)+'].[0]/fps]);\r')
            f.write('    num.do{arg i;\r        i.post;"/".post;num.postln;\r        (1/'+str(self.outfps)+').yield;\r')
            for i in range (self.sc_resetPositionMap.shape[0]):
                f.write('\r        gs'+str(i)+'.set("resetp",x['+str(i)+'].[i]/fps);\r')
                f.write('        gs'+str(i)+'.set("rate",y['+str(i)+'].[i]);\r')
                f.write('        gs'+str(i)+'.set("inpan",z['+str(i)+'].[i]/3840);\r')
            f.write('        };\r    };\r)')

        with open(csvoutstr +'SC_Rev-Play-'+str(j)+'voices.scd','w') as f:
            f.write(sc_simplePlay_setting_str)
            f.write(sc_rev_setting_str)
            f.write(sc_load_str)
            f.write('\n')
            f.write('x = CSVFileReader.readInterpret(r[0]).postcs;//ResetP \ry = CSVFileReader.readInterpret(r[1]).postcs;//Rate \rz = CSVFileReader.readInterpret(r[2]).postcs;//inPanMap \rc = CSVFileReader.readInterpret(r[3]).postcs;//nowdepth')
            f.write('\n')
            f.write('a = Buffer.readChannel(s,r[4],channels:[1]);\rb = Buffer.readChannel(s,r[4],channels:[0]);\r)')
            f.write('\n')
            f.write('(\r// SimplePlay-Reverbバージョン')
            f.write('\n')
            f.write('"open -a \'QuickTime Player\' \''+NOW_DIR+"/"+csvoutstr +'.mp4'+'\' ".unixCmd;')
            f.write('\n')
            f.write('s.record("'+ NOW_DIR + "/" + self.ORG_NAME + "_" + self.out_name_attr +'_Rev-Play-'+ str(j) + 'voices.aiff");')
            f.write('\n')
            f.write('fork{var cha='+str(j)+',num=x[0].size,recfps='+str(self.recfps)+',trig=0,delaych,combch,mix,zmax='+str(np.amax(self.sc_now_depth))+',maxdtime=5')
            for i in range (self.sc_resetPositionMap.shape[0]):f.write(',gs'+str(i))
            f.write(';')
            f.write('\n')
            for i in range (self.sc_resetPositionMap.shape[0]):
                if i == 0 :
                    f.write('    gs'+str(i)+' = Synth.head(s,"simplePlay",["bufL",a,"bufR",b,"rate",1,"trig",trig+rand(0.05),"pan",0,"amp",1/cha,"resetp",x['+str(i)+'].[0]/recfps,"out",4]);\r')
                elif i == self.sc_resetPositionMap.shape[0]-1:
                    f.write('    gs'+str(i)+' = Synth.head(s,"simplePlay",["bufL",a,"bufR",b,"rate",1,"trig",trig+rand(0.05),"pan",1,"amp",1/cha,"resetp",x['+str(i)+'].[0]/recfps,"out",4]);\r')
                else:
                    f.write('    gs'+str(i)+' = Synth.head(s,"simplePlay",["bufL",a,"bufR",b,"rate",1,"trig",trig+rand(0.05),"pan",'+str(i/(j-1))+',"amp",1/cha,"resetp",x['+str(i)+'].[0]/recfps,"out",4]);\r')
            f.write('    Synth.tail(s, "preDelay");\r        combch=Synth.tail(s, "combs");\r        delaych=Synth.tail(s, "allpass");\r        mix=Synth.tail(s, "theMixer");\r')
            f.write('    num.do{arg i;\r        i.post;"/".post;num.postln;\r        (1/'+str(self.outfps)+').yield;\r')
            for i in range (self.sc_resetPositionMap.shape[0]):
                f.write('\r        gs'+str(i)+'.set("resetp",x['+str(i)+'].[i]/((b.numFrames / b.sampleRate)*recfps));\r')
                f.write('        gs'+str(i)+'.set("rate",y['+str(i)+'].[i]);\r')
                f.write('        gs'+str(i)+'.set("inpan",z['+str(i)+'].[i]/3840);\r')
            f.write('\r        if (zmax > (maxdtime*recfps),{combch.set("dtime",(c[i][0]/zmax)*maxdtime);},{combch.set("dtime",c[i][0]/recfps)});\r')
            f.write('        delaych.set("gain",(c[i][0] / zmax)*0.2);\r')
            f.write('        mix.set("dry",1-((c[i][0] / zmax)*0.2));\r')
            f.write('        };\r    };\r)')

        with open(csvoutstr +'SC_Rev-Grain-'+str(j)+'voices.scd','w') as f:
            f.write(sc_grain_setting_str)
            f.write(sc_rev_setting_str)
            f.write(sc_load_str)
            f.write('\n')
            f.write('x = CSVFileReader.readInterpret(r[0]).postcs;//ResetP \ry = CSVFileReader.readInterpret(r[1]).postcs;//Rate \rz = CSVFileReader.readInterpret(r[2]).postcs;//inPanMap \rc = CSVFileReader.readInterpret(r[3]).postcs;//nowdepth')
            f.write('\n')
            f.write('a = Buffer.readChannel(s,r[4],channels:[1]);\rb = Buffer.readChannel(s,r[4],channels:[0]);\r)')
            f.write('\n')
            f.write('(\r//Grain-Reverbバージョン')
            f.write('\n')
            f.write('"open -a \'QuickTime Player\' \''+NOW_DIR+"/"+csvoutstr +'.mp4'+'\' ".unixCmd;')
            f.write('\n')
            f.write('s.record("'+ NOW_DIR + "/" + self.ORG_NAME + "_" + self.out_name_attr +'_Rev-Grain-'+ str(j) + 'voices.aiff");')
            f.write('\n')
            f.write('fork{var cha='+str(j)+',num=x[0].size,recfps='+str(self.recfps)+',trig=0,delaych,combch,mix,zmax='+str(np.amax(self.sc_now_depth))+',maxdtime=5')
            for i in range (self.sc_resetPositionMap.shape[0]):f.write(',gs'+str(i))
            f.write(';')
            f.write('\n')
            for i in range (self.sc_resetPositionMap.shape[0]):
                if i == 0 :
                    f.write('    gs'+str(i)+' = Synth.head(s,"bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",0,"amp",1/cha,"resetp",x['+str(i)+'].[0]/((b.numFrames / b.sampleRate)*recfps),"out",4]);\r')
                elif i == self.sc_resetPositionMap.shape[0]-1:
                    f.write('    gs'+str(i)+' = Synth.head(s,"bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",1,"amp",1/cha,"resetp",x['+str(i)+'].[0]/((b.numFrames / b.sampleRate)*recfps),"out",4]);\r')
                else:
                    f.write('    gs'+str(i)+' = Synth.head(s,"bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",'+str(i/(j-1))+',"amp",1/cha,"resetp",x['+str(i)+'].[0]/((b.numFrames / b.sampleRate)*recfps),"out",4]);\r')
            f.write('    Synth.tail(s, "preDelay");\r        combch=Synth.tail(s, "combs");\r        delaych=Synth.tail(s, "allpass");\r        mix=Synth.tail(s, "theMixer");\r')
            f.write('    num.do{arg i;\r        i.post;"/".post;num.postln;\r        (1/'+str(self.outfps)+').yield;\r')
            for i in range (self.sc_resetPositionMap.shape[0]):
                f.write('\r        gs'+str(i)+'.set("resetp",x['+str(i)+'].[i]/((b.numFrames / b.sampleRate)*recfps));\r')
                f.write('        gs'+str(i)+'.set("rate",y['+str(i)+'].[i]);\r')
                f.write('        gs'+str(i)+'.set("inpan",z['+str(i)+'].[i]/3840);\r')
            f.write('\r        if (zmax > (maxdtime*recfps),{combch.set("dtime",(c[i][0]/zmax)*maxdtime);},{combch.set("dtime",c[i][0]/recfps)});\r')
            f.write('        delaych.set("gain",(c[i][0] / zmax)*0.1);\r')
            f.write('        mix.set("dry",1-((c[i][0] / zmax)*0.2));\r')
            f.write('        };\r    };\r)')

    def maneuver_2dplot(self, thread_num=None, debugmode=False,normal_line_draw=False, w_inc=5, h_inc=9,plinewidth=1.0,individual_output=False,timeflow_scaling=True,palpha=1.0,x_positions=[],y_positions=[]):
        print(sys._getframe().f_code.co_name)
        if thread_num != None: 
            self.info_setting(thread_num)
        else:
            if self.infolog == 0:
                self.info_setting() 
            elif self.infolog != self.log:
                self.info_setting(self.sc_resetPositionMap.shape[0])
                
        j = self.sc_resetPositionMap.shape[0]
        cmap = LinearSegmentedColormap.from_list('original', [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)], N=j)
        color_map = plt.get_cmap(cmap)

        plt.style.use("bmh")
        mpl.rcParams['axes.facecolor'] = (1,1,1,0)
        mpl.rcParams['axes.edgecolor'] = (0,0,0,0.5)
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['grid.color'] = "#aaaaaa"
        
        if individual_output:
            fig1, ax1 = plt.subplots()   # 新しい図とaxesを作成
            for n in range(j):
                ax1.plot(self.sc_inPanMap[n], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                # ax1.plot(self.sc_inPanMap[n], color="red",linewidth=plinewidth)
            if self.scan_direction == 1:
                ax1.set_ylabel('Space(X) Flow(px)')
            else:
                ax1.set_ylabel('Space(Y) Flow(px)')
            fig1.set_size_inches(w_inc, h_inc/3)
            fig1.tight_layout()
            fig1.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread_SpaceFlow.png', dpi=300, transparent=True)
            fig2, ax2 = plt.subplots()
            for n in range(j):
                ax2.plot(self.sc_resetPositionMap[n], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                # ax2.plot(self.sc_resetPositionMap[n], color="red",linewidth=plinewidth)
            ax2.set_ylabel('Time Flow(frame)')
            # Y軸の最大値を設定
            if timeflow_scaling != True : ax2.set_ylim(top=self.count)
            fig2.set_size_inches(w_inc, h_inc/3)
            fig2.tight_layout()
            fig2.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread_TimeFlow.png', dpi=300, transparent=True)

            fig3, ax3 = plt.subplots()
            for n in range(j):
                ax3.plot(self.sc_rateMap[n], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                # ax3.plot(self.sc_rateMap[n], color="red",linewidth=plinewidth)
            ax3.set_ylabel('Play Rate')
            if normal_line_draw :
                ax3.axhline(y=1.0, color='black', linestyle='-',linewidth=0.3)
                # y軸に1.0を強調表示
                current_ticks = ax3.get_yticks()
                new_ticks = np.append(current_ticks, 1.0)
                ax3.set_yticks(new_ticks)

            fig3.set_size_inches(w_inc, h_inc/3)
            fig3.tight_layout()
            fig3.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread_PlayRate.png', dpi=300, transparent=True)
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(3, 1, 1)
            for n in range(j):
                ax1.plot(self.sc_inPanMap[n], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                # ax1.plot(self.sc_inPanMap[n], color="red",linewidth=plinewidth,alpha=palpha)
            if self.scan_direction == 1:
                ax1.set_ylabel('Space(X) Flow(px)')
            else:
                ax1.set_ylabel('Space(Y) Flow(px)')

            ax2 = fig.add_subplot(3, 1, 2)
            for n in range(j):
                ax2.plot(self.sc_resetPositionMap[n], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                # ax2.plot(self.sc_resetPositionMap[n], color="red",linewidth=plinewidth,alpha=palpha)
            # Y軸の最大値を設定
            if timeflow_scaling != True : ax2.set_ylim(top=self.count)
            ax2.set_ylabel('Time Flow(frame)')

            ax3 = fig.add_subplot(3, 1, 3)
            for n in range(j):
                ax3.plot(self.sc_rateMap[n], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                # ax3.plot(self.sc_rateMap[n], color="red",linewidth=plinewidth,alpha=palpha)
            ax3.set_ylabel('Play Rate')
            if normal_line_draw :
                ax3.axhline(y=1.0, color='black', linestyle='-',linewidth=palpha)
                # y軸に1.0を強調表示
                current_ticks = ax3.get_yticks()
                new_ticks = np.append(current_ticks, 1.0)
                ax3.set_yticks(new_ticks)
            # 各x_positionに対して縦線を描画
            for x_position in x_positions:
                # ax1.axvline(x=x_position, color='black', linestyle='-', linewidth=1.0)
                ax2.axvline(x=x_position, color='black', linestyle='-', linewidth=1.0)
                # ax3.axvline(x=x_position, color='black', linestyle='-', linewidth=1.0)

            for y_position in y_positions:
                # ax1.axhline(x=x_position, color='black', linestyle='-', linewidth=1.0)
                ax2.axhline(y=y_position, color='black', linestyle='-', linewidth=1.0)
                # ax3.axhline(x=x_position, color='black', linestyle='-', linewidth=1.0)
                    

            fig.set_size_inches(w_inc, h_inc)
            plt.tight_layout()
            if len(x_positions)>0 or len(y_positions)>0:
                plt.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread-renderSeparation.png', dpi=300, transparent=True)
            else:
                plt.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread.png', dpi=300, transparent=True)

        if debugmode: 
            plt.show()
        plt.clf()
        plt.close('all')

    def maneuver_3dplot(self,thread_num=None,zRangeFix=False,outFrame_nums=50,out_fps=10):
        print(sys._getframe().f_code.co_name)
        if thread_num != None : self.info_setting(thread_num)
        else:
            if self.infolog == 0 :self.info_setting() 
            elif self.infolog != self.log :self.info_setting(self.sc_resetPositionMap.shape[0])
        j=self.sc_resetPositionMap.shape[0]
        cmap = LinearSegmentedColormap.from_list('original',[(0.0,1.0,0.0),(1.0,0.0,0.0)], N=j)
        color_map = plt.get_cmap(cmap)
        zMin=np.amin(self.data[:,:,1])
        zMax=np.amax(self.data[:,:,1])
        zRange=zMax-zMin
        xMin=0
        xMax=self.scan_nums if self.scan_direction%2 == 1  else self.slit_length 
        yMax =self.slit_length  if self.scan_direction%2 == 1 else self.scan_nums
        drawLineNum=j
        if self.data.shape[0]< outFrame_nums:
            outFrame_nums= self.data.shape[0]
        nsteps = math.floor(self.data.shape[0]/outFrame_nums)
        if os.path.isdir('3dPlot_seq')==False:
            os.makedirs('3dPlot_seq')#ディレクトリ作成
        # styles=plt.style.available
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.view_init(elev=25, azim=-40)#3d視点設定。elevは仰角（上下の角度）を表し、azimは方位角（左右の角度）
        for i in range(0,self.data.shape[0],nsteps):
            for k in range(drawLineNum):
                    rp=k*math.ceil(self.data.shape[1]/(drawLineNum-1)) if k != (drawLineNum-1) else (self.data.shape[1]-1)
                    # print(rp)
                    if self.scan_direction%2==1:
                        ax.plot([self.data[i,rp,0],self.data[i,rp,0]],[self.data[i,rp,1],self.data[i,rp,1]],[self.slit_length,0],color=color_map(k))
                    else :
                        ax.plot([self.slit_length,0],[self.data[i,rp,1],self.data[i,rp,1]],[self.data[i,rp,0],self.data[i,rp,0]],color=color_map(k))
            plt.style.use("bmh")
            mpl.rcParams['axes.facecolor'] = (1,1,1,0)
            mpl.rcParams['axes.edgecolor'] = (1,1,1,0)
            mpl.rcParams['axes.labelsize']= 12
            mpl.rcParams['axes3d.grid']=True
            mpl.rcParams['grid.color']="#dddddd"
            if zRangeFix:ax.set_ylim(zMin,zMax) # z軸固定
            ax.set_xlim(xMin, xMax) # x軸固定
            ax.set_xticks(np.arange(0,xMax+1,math.floor(xMax/2)))
            ax.set_yticks(np.arange(zMin,zMax+1,closest_value(500,50000,10,zRange/3)))
            ax.set_zticks(np.arange(0,yMax+1,math.floor(yMax/2)))
            # ax.w_xaxis.line.set_color('#888888') py3.11でNG
            # ax.w_yaxis.line.set_color('#888888')
            # ax.w_zaxis.line.set_color('#888888')
            ax.xaxis.line.set_color('#888888')
            ax.yaxis.line.set_color('#888888')
            ax.zaxis.line.set_color('#888888')
            ax.zaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")
            ax.xaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")
            ax.yaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")
            plt.savefig('3dPlot_seq/'+ self.ORG_NAME+"_process"+str(self.log)+'_3dPlot_'+str(j)+'-'+str(i)+'.png',dpi=200,transparent=True,bbox_inches='tight')
            plt.cla()
        plt.close()
        # fig.set_size_inches(8,7)  # 幅: 8インチ、高さ: 7インチ
        img=cv2.imread('3dPlot_seq/'+ self.ORG_NAME+"_process"+str(self.log)+'_3dPlot_'+str(j)+'-'+str(nsteps)+'.png')
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
        # fourcc = cv2.VideoWriter_fourcc(*'pngv')  #アルファチャンネル付き
        vname=self.ORG_NAME+"_"+self.out_name_attr +'_3dPlot.mp4'
        if self.embedHistory_intoName :
            max_length=230
            if len(vname) > max_length :
                self.embedHistory_intoName = False
        if self.embedHistory_intoName == False :
            vname=self.ORG_NAME+"_process"+str(self.log)+'_3dPlot.mp4'
        video = cv2.VideoWriter(vname,fourcc,out_fps,(int(img.shape[1]),int(img.shape[0])),1) 
        for i in range(0,self.data.shape[0],nsteps):
            readimg=cv2.imread('3dPlot_seq/'+self.ORG_NAME+"_process"+str(self.log)+'_3dPlot_'+str(j)+'-'+str(i)+'.png')
            if img.shape != readimg.shape:
                readimg=cv2.resize(readimg,(img.shape[1],img.shape[0]))
            video.write(readimg)
        video.release()

    def data_save(self):
        np.save(self.ORG_NAME+"_"+self.out_name_attr+'_raw.npy',self.data)

    # 映像のレンダリング
    def transprocess(self,separate_num=None,sep_start_num=0,sep_end_num=None,out_type=1,XY_TransOut=False,render_mode=0,seqrender=False,title_atr:str=None, tmp_type_para=0,del_data=True,auto_memory_clear=True,memory_report=False,tmp_save = False ):
        #self.outfpsはグローバルで定義
        color_depth = 8
        num_channels = 3
        file_size_bytes = (self.data.shape[0]*self.slit_length*self.data.shape[1] * color_depth * num_channels) / 8
        # Convert the file size to megabytes (1 MB = 1024 * 1024 bytes)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        if separate_num == None :
            separate_num = math.ceil(file_size_megabytes /((psutil.virtual_memory().available/(1024)**2)* (self.memory_percent/100)))
        print("separate_num=",separate_num,"active memory=",(psutil.virtual_memory().available/(1024)**2),"mb")
        if sep_end_num == None:sep_end_num = separate_num
        runFirstTime = time.time()
        XY_Name = "Y" if self.scan_direction%2 == 0 else "X"
        videostr = self.ORG_NAME+"_"+self.out_name_attr if render_mode == 0 else self.ORG_NAME+"_"+self.out_name_attr+"("+str(sep_start_num)+"-"+str(sep_end_num)+"sep)"
        if self.embedHistory_intoName == False :
            videostr = self.ORG_NAME+"_process"+str(self.log)
        if title_atr != None : videostr+=title_atr
        append_to_logfile("transprocess:"+str(separate_num)+"separate")
        append_to_logfile("active memory="+str(psutil.virtual_memory().available/(1024)**2)+"mb")
        append_to_logfile("memory_usage="+str(((file_size_megabytes / separate_num) / (psutil.virtual_memory().available/(1024)**2))*100)+"%")
        append_to_logfile(videostr)
        rotate_direction = False
        print("framecount=",self.data.shape[0],"(",self.data.shape[0]/self.recfps,"sec)",XY_Name+"(out)=",self.data.shape[1],"refer(in"+XY_Name+"-out"+XY_Name+"-ZP)=",self.data.shape[2])
        print("z min-max =",np.amin(self.data[:,:,-1]),np.amax(self.data[:,:,-1]))
        append_to_logfile("outfps="+str(self.outfps))
        append_to_logfile("framecount="+str(self.data.shape[0]))
        append_to_logfile("z min-max ="+str(np.amin(self.data[:,:,-1]))+"-"+str(np.amax(self.data[:,:,-1])))
        if np.amin(self.data[:,:,-1])<0:
            print("z<0,error")
            return
        if np.amax(self.data[:,:,-1])>self.count:
            print("z>video_count,error,data_zmax:",np.amax(self.data[:,:,-1])," > count:",self.count)
            return
        #audioようにfloatで計算していたのをintへ戻す。この方法だと小数点以下は切り捨て
        wr_array = self.data[:,:,1:].astype(np.int32) if self.data.shape[2] == 3 else self.data.astype(np.int32)

        # 分割の切れ目の数値を格納する配列を初期化
        cut_points = []

        # 切れ目を計算して配列に追加
        for i in range(1,separate_num):
            cut = wr_array.shape[0] * i // separate_num  # 整数の除算を使用して切れ目を計算
            cut_points.append(cut)
        self.maneuver_2dplot(x_positions=cut_points)
        
        if del_data :
            #メモリの最適化のため
            del self.data
        
        if self.sepVideoOut != 1:  #sepVideoOut　はglobal関数。セパレートしない場合、rawでnpアレイファイルをテンポファイルとしてハードディスクに貯めておき、全てのアレイが準備できてからレンダリングする。そのためHD容量を100GBとか普通に食う。
            if os.path.isdir("tmp")==False:
                os.makedirs("tmp")
            if os.path.isdir("img")==False and out_type != 1 :
                os.makedirs("img")
        if seqrender:
            # 配列をソートし、重複を省略した一時配列を作成
            sorted_unique_array = np.sort(np.unique(wr_array[:,:,1]))
            print(len(sorted_unique_array),"枚の画像の保存")
            seq_rom=(len(sorted_unique_array)*self.slit_length*self.data.shape[1] * color_depth * num_channels)  / (8 * 1024 * 1024 *1000)
            disk_remain =shutil.disk_usage(os.getcwd())[2]/(1024*1024*1000)
            print(seq_rom,"mb容量が必要、", file_size_megabytes,"mb=書き出し用に必要な容量。",disk_remain,"mb容量の保存可能")
            if disk_remain > (seq_rom + file_size_megabytes):
                save_video_frames(self.VIDEO_PATH, img_format='npy',frame_array=sorted_unique_array)
                append_to_logfile("npy sequence buffered "+ str(round(time.time()-runFirstTime,2))+"sec")
            else:
                print("容量が足りない")
                seqrender=False
            
        for s in range(sep_start_num,sep_end_num):
            # メモリ統計を格納するためのリスト
            if memory_report : memory_stats = []
            if self.scan_direction % 2 == 1:
                img_shape = (int(self.height), int(wr_array.shape[1]), 3)
            else:
                img_shape = (int(wr_array.shape[1]), int(self.width), 3)
            base_images_frame_num = wr_array.shape[0]//separate_num
            images_frame_num = base_images_frame_num
            if s == sep_end_num-1:
                images_frame_num +=  wr_array.shape[0] % separate_num
            
            print("img-declare",int(s*wr_array.shape[0]/separate_num),"->",int((s+1)*wr_array.shape[0]/separate_num))
            images = np.zeros((images_frame_num, *img_shape), dtype=np.uint8)
            seq_read_s=int(s*base_images_frame_num)
            seq_read_e=int(seq_read_s+images_frame_num)#分割できなかった余剰フレームは最後の演算の時に加える

            #ビデオの設定
            if self.sepVideoOut == 1 :
                print("video-preference")
                fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
                if self.scan_direction%2==1 :
                    video = cv2.VideoWriter(videostr +'sep_index='+str(s)+'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'sep_index='+str(s)+'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
                else :
                    video = cv2.VideoWriter(videostr +'sep_index='+str(s)+'.mp4', fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'sep_index='+str(s)+'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
            interval_first = None
            block_num = 0
            minz = np.amin(wr_array[seq_read_s:seq_read_e,:,1])
            maxz = np.amax(wr_array[seq_read_s:seq_read_e,:,1])
            print("minz",minz)
            print("maxz",maxz)
            print("depth=",np.amin(self.sc_now_depth[seq_read_s:seq_read_e]),np.amax(self.sc_now_depth[seq_read_s:seq_read_e]))
            
            totalnum=maxz-minz
            totalslits=wr_array.shape[0]/separate_num*wr_array.shape[1]
            slitscounter=0
            if totalnum == 0 : totalnum=1
            progresscale=self.progressbarsize/totalnum
            progressAllScale=self.progressbarsize/(round(self.count))
            writingTotalNum=int(wr_array.shape[0]/separate_num)
            writingScale = self.progressbarsize/writingTotalNum
            if minz < 0 :
                print("z<0,error")
                return
            num = minz
            print("count",self.count)
            print("processing-Frames",writingTotalNum)
            # print("num=",num)
            if self.cap == None :
                sstime = time.time()
                # 画像ファイルのリストを取得
                image_files = [f for f in os.listdir(self.ORG_PATH +"/"+ self.ORG_NAME) if f.endswith(('.png', '.jpg', '.tif' ,'.jpeg', '.bmp','.npy'))]
                image_files.sort()  # ソートして順番を保証
                # 配列をソートし、重複を省略した一時配列を作成
                sorted_unique_array = np.sort(np.unique(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1]))
            
                for num in sorted_unique_array : 
                    stime = time.time()
                    image_path = os.path.join(self.ORG_PATH +"/"+ self.ORG_NAME, image_files[num])
                    frame = np.load(image_path) if image_path.endswith('.npy') else cv2.imread(image_path)
                    if self.scan_direction%2 == 0 :
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wr =np.array(wr_array[i])
                            indices = np.where(wr[:, 1] == num)
                            for p in indices[0]:
                                images[i,p,:]= frame[wr[p,0],:]
                            slitscounter += len(indices[0])
                    else :
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wr =np.array(wr_array[i])
                            indices = np.where(wr[:, 1] == num)
                            for p in indices[0]:
                                images[i,:,p]=frame[:,wr[p,0]]
                            slitscounter += len(indices[0])
                    etime = time.time()
                    Interval = etime - stime
                    if interval_first == None :
                        interval_first = Interval
                    elif Interval < interval_first :
                        interval_first=Interval
                    #progressbar
                    bar= '■'*int((num-minz)*progresscale) + "."*int((totalnum-(num-minz))*progresscale)
                    print(f"\r\033[K[\033[33m{bar}\033[39m] frame{(num-minz)/totalnum*100:.02f}%({minz}>{num}>{maxz}) Slits{slitscounter/totalslits*100:.02f}%({slitscounter}/{int(totalslits)}) : {round(Interval,2)}({round(interval_first,2)})sec/f",end="")
                print("\r")
                lbstime = time.time()
                Interval=lbstime-sstime
                if self.sepVideoOut == 1 or separate_num == 1:
                    print("video-preference",round(Interval,2))
                    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
                    self.out_videopath=videostr +'.mp4'
                    if self.scan_direction%2 == 1 :
                        video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
                    else :
                        video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
                else : print("file Writing",round(Interval,2))
                if tmp_type_para :
                    for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                        wstime = time.time()
                        if out_type != 1:
                            img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                            exec('cv2.imwrite(img_name,img%d)' %(i))
                        elif out_type != 0 :
                            if self.sepVideoOut == 1 or separate_num == 1:
                                if XY_TransOut :
                                    if rotate_direction: exec('video.write(img%d.transpose(1,0,2)[:,::-1])' %(i))
                                    else : exec('video.write(img%d.transpose(1,0,2)[::-1])' %(i))
                                else: exec('video.write(img%d)' %(i))
                            else :
                                tmp_name="tmp/"+self.ORG_NAME+"_"+str(i)
                                exec('np.save(tmp_name,img%d)' %(i))
                        exec("del img%d"%(i))
                        gc.collect()
                        wetime = time.time()
                        knterval = round(wetime-wstime,2)
                        ci=i-int(s*wr_array.shape[0]/separate_num)
                        bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                        if self.sepVideoOut == 1 or separate_num == 1:
                            print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                        else :
                            print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                    if self.sepVideoOut == 1  or separate_num == 1 : video.release()
                else : 
                    if out_type == 1 and (self.sepVideoOut != 1 or separate_num != 1):
                        tmp_name="tmp/"+self.ORG_NAME+"_sep-"+str(s)
                        np.save(tmp_name,images)
                        print("npydata_saved")
                    else : 
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wstime = time.time()
                            if out_type != 1:
                                img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                                cv2.imwrite(img_name,images[i])
                            if out_type != 0 :
                                if XY_TransOut :
                                    if rotate_direction: video.write(images[i].transpose(1,0,2)[:,::-1])
                                    else : video.write(images[i].transpose(1,0,2)[::-1])
                                else: video.write(images[i])
                            gc.collect()
                            wetime = time.time()
                            knterval = round(wetime-wstime,2)
                            ci=i-int(s*wr_array.shape[0]/separate_num)
                            bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                            if self.sepVideoOut == 1 or separate_num == 1:
                                print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                            else :
                                print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                        if self.sepVideoOut == 1  or separate_num == 1 : video.release()
            
                print("done:",s+1,"/",separate_num)
                append_to_logfile("done:"+str(s+1)+"/"+str(separate_num)+" "+str(round(time.time()-sstime,2))+"sec wrote-Slits:"+str(int(totalslits))+"("+str(round(totalslits/(wr_array.shape[0]*wr_array.shape[1])*100,2))+"%) scan-Frames:"+str(minz)+"->"+str(maxz))
                print("\r")
                gc.collect()
            else : 
                if seqrender:
                    sstime = time.time()
                    for num in sorted_unique_array : 
                        stime = time.time()
                        image_path = "seq/frames_"+str(num)+".npy"
                        frame = np.load(image_path) if image_path.endswith('.npy') else cv2.imread(image_path)
                        if self.scan_direction%2 == 0 :
                            for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                                wr =np.array(wr_array[i])
                                indices = np.where(wr[:, 1] == num)
                                for p in indices[0]:
                                    images[i,p,:] = frame[wr[p,0],:]
                                slitscounter += len(indices[0])
                        else :
                            for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                                wr =np.array(wr_array[i])
                                indices = np.where(wr[:, 1] == num)
                                for p in indices[0]:
                                    images[i,:,p] = frame[:,wr[p,0]]
                                slitscounter += len(indices[0])
                        etime = time.time()
                        Interval = etime - stime
                        if interval_first == None :
                            interval_first = Interval
                        elif Interval < interval_first :
                            interval_first=Interval
                        #progressbar
                        bar= '■'*int((num-minz)*progresscale) + "."*int((totalnum-(num-minz))*progresscale)
                        print(f"\r\033[K[\033[33m{bar}\033[39m] frame{(num-minz)/totalnum*100:.02f}%({minz}>{num}>{maxz}) Slits{slitscounter/totalslits*100:.02f}%({slitscounter}/{int(totalslits)}) : {round(Interval,2)}({round(interval_first,2)})sec/f",end="")
                    print("\r")
                    lbstime = time.time()
                    Interval=lbstime-sstime
                    if self.sepVideoOut == 1 or separate_num == 1:
                        print("video-preference",round(Interval,2))
                        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
                        self.out_videopath=videostr +'.mp4'
                        if self.scan_direction%2 == 1 :
                            video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
                        else :
                            video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
                    else : print("file Writing",round(Interval,2))
                    if tmp_type_para :
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wstime = time.time()
                            if out_type != 1:
                                img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                                cv2.imwrite(img_name,images[i])
                            elif out_type != 0 :
                                if self.sepVideoOut == 1 or separate_num == 1:
                                    if XY_TransOut :
                                        if rotate_direction: video.write(images[i].transpose(1,0,2)[:,::-1])
                                        else : video.write(images[i].transpose(1,0,2)[::-1])
                                    else: video.write(images[i])
                                else :
                                    tmp_name="tmp/"+self.ORG_NAME+"_"+str(i)
                                    np.save(tmp_name,images[i])
                            gc.collect()
                            wetime = time.time()
                            knterval = round(wetime-wstime,2)
                            ci=i-int(s*wr_array.shape[0]/separate_num)
                            bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                            if self.sepVideoOut == 1 or separate_num == 1:
                                print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                            else :
                                print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                        if self.sepVideoOut == 1  or separate_num == 1 : video.release()
                    else :
                        if out_type == 1 and (self.sepVideoOut != 1 or separate_num != 1):
                            tmp_name="tmp/"+self.ORG_NAME+"_sep-"+str(s)
                            np.save(tmp_name,images)
                            print("npydata_saved")
                        else : 
                            for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                                wstime = time.time()
                                if out_type != 1:
                                    img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                                    cv2.imwrite(img_name,images[i])
                                if out_type != 0 :
                                    if XY_TransOut :
                                        if rotate_direction: video.write(images[i].transpose(1,0,2)[:,::-1])
                                        else : video.write(images[i].transpose(1,0,2)[::-1])
                                    else: video.write(images[i])
                                gc.collect()
                                wetime = time.time()
                                knterval = round(wetime-wstime,2)
                                ci=i-int(s*wr_array.shape[0]/separate_num)
                                bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                                if self.sepVideoOut == 1 or separate_num == 1:
                                    print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                                else :
                                    print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                            if self.sepVideoOut == 1  or separate_num == 1 : video.release()
                    print("done:",s+1,"/",separate_num)
                    append_to_logfile("done:"+str(s+1)+"/"+str(separate_num)+" "+str(round(time.time()-sstime,2))+"sec wrote-Slits:"+str(int(totalslits))+"("+str(round(totalslits/(wr_array.shape[0]*wr_array.shape[1])*100,2))+"%) scan-Frames:"+str(minz)+"->"+str(maxz))
                    print("\r")
                    gc.collect()
                else :
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, num)
                    sstime = time.time()
                    Interval=0
                    while(self.cap.isOpened()):
                        ret, frame = self.cap.read()
                        reedfull_bool = num < (maxz+1)
                        if ret == True and reedfull_bool :
                            stime = time.time()
                            if (num > minz or num == minz):
                                if self.scan_direction%2 == 0 :
                                    for i in range(seq_read_s,seq_read_e):
                                        wr =np.array(wr_array[i])
                                        indices = np.where(wr[:, 1] == num)
                                        for p in indices[0]:
                                            images[i-seq_read_s,p,:] = frame[wr[p,0],:]
                                        slitscounter += len(indices[0])
                                else :
                                    for i in range(seq_read_s,seq_read_e):
                                        wr =np.array(wr_array[i])
                                        indices = np.where(wr[:, 1] == num)
                                        for p in indices[0]:
                                            images[i-seq_read_s,:,p] = frame[:,wr[p,0]]
                                        slitscounter += len(indices[0])
                            num += 1
                            etime = time.time()
                            newterval=(etime - stime)/(len(indices[0])+1)*1000
                            if Interval*2.5 < newterval and Interval!=0 and auto_memory_clear : 
                                gc.collect()
                                print("memoryclear", newterval ,Interval)
                            Interval = newterval
                            # if interval_first == None :
                            #     interval_first = Interval
                            # elif Interval < interval_first :
                            #     interval_first=Interval
                            #progressbar
                            bar= '■'*int((num-minz)*progresscale) + "."*int((totalnum-(num-minz))*progresscale)
                            # print(f"\r\033[K[\033[33m{bar}\033[39m] frame{(num-minz)/totalnum*100:.02f}%({minz}>{num}>{maxz}) Slits{slitscounter/totalslits*100:.02f}%({slitscounter}/{int(totalslits)}) : {round(Interval,2)}({round(interval_first,2)})sec/f",end="")
                            print(f"\r\033[K[\033[33m{bar}\033[39m] frame{(num-minz)/totalnum*100:.02f}%({minz}>{num}>{maxz}) Slits{slitscounter/totalslits*100:.03f}%({slitscounter}/{int(totalslits)}) : {round(Interval,2)}ms/slit",end="")
                            # if num%(totalnum//100)==0:
                            if memory_report : 
                                vmem = psutil.virtual_memory()
                                memory_stats.append([
                                    Interval,sys.getsizeof(images),vmem.total, vmem.available, vmem.used, vmem.percent,
                                    vmem.active, vmem.inactive, vmem.wired, vmem.free
                                ])
                        else:
                            print("\r")
                            lbstime = time.time()
                            Interval=lbstime-sstime
                            if self.sepVideoOut == 1 or separate_num == 1:
                                print("video-preference",round(Interval,2))
                                fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
                                self.out_videopath=videostr +'.mp4'
                                if self.scan_direction%2 == 1 :
                                    video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
                                else :
                                    video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
                            else : print("file Writing",round(Interval,2))

                            if tmp_type_para :
                                for i in range(seq_read_s,seq_read_e):
                                    wstime = time.time()
                                    if out_type != 1:
                                        img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                                        cv2.imwrite(img_name,images[i-seq_read_s])
                                    elif out_type != 0 :
                                        if self.sepVideoOut == 1 or separate_num == 1:
                                            if XY_TransOut :
                                                if rotate_direction: video.write(images[i].transpose(1,0,2)[:,::-1])
                                                else : video.write(images[i].transpose(1,0,2)[::-1])
                                            else: video.write(images[i-seq_read_s])
                                        else :
                                            tmp_name="tmp/"+self.ORG_NAME+"_"+str(i)
                                            np.save(tmp_name,images[i-seq_read_s])
                                    gc.collect()
                                    wetime = time.time()
                                    knterval = round(wetime-wstime,2)
                                    ci=i-int(s*wr_array.shape[0]/separate_num)
                                    bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                                    if self.sepVideoOut == 1 or separate_num == 1:
                                        print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                                    else :
                                        print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                                if self.sepVideoOut == 1  or separate_num == 1 : video.release()
                            else:
                                # if out_type == 1 and (self.sepVideoOut != 1 or separate_num != 1):
                                if out_type != 0 and self.sepVideoOut != 1 and separate_num != 1:
                                    wstime = time.time()
                                    tmp_name="tmp/"+self.ORG_NAME+"_sep-"+str(s)
                                    np.save(tmp_name,images)
                                    knterval = round(time.time()-wstime,2)
                                    print("tmp-npydata_saved",images.shape,knterval,"sec")
                                else : 
                                    print("direct render")
                                    for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                                        wstime = time.time()
                                        if out_type != 1:
                                            img_name="img/"+self.ORG_NAME+"_"+str(i)+"p"+self.imgtype 
                                            cv2.imwrite(img_name,images[i])
                                        if out_type != 0 :
                                            if XY_TransOut :
                                                if rotate_direction: video.write(images[i].transpose(1,0,2)[:,::-1])
                                                else : video.write(images[i].transpose(1,0,2)[::-1])
                                            else: video.write(images[i])
                                        gc.collect()
                                        wetime = time.time()
                                        knterval = round(wetime-wstime,2)
                                        ci=i-int(s*wr_array.shape[0]/separate_num)
                                        bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                                        if self.sepVideoOut == 1 or separate_num == 1:
                                            print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                                        else :
                                            print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                                    if self.sepVideoOut == 1  or separate_num == 1 : video.release()
                            break
                    del images
                    print("done:",s+1,"/",separate_num)
                    render_rate = round((time.time()-sstime)/ ((wr_array.shape[0]/separate_num) / 30),2)
                    append_to_logfile("done:"+str(s+1)+"/"+str(separate_num)+" "+str(round(time.time()-sstime,2))+"sec wrote-Slits:"+str(int(totalslits))+"("+str(round(totalslits/(wr_array.shape[0]*wr_array.shape[1])*100,2))+"%) scan-Frames:"+str(minz)+"->"+str(maxz)+" render_rate="+str(render_rate)+" memory="+str(psutil.virtual_memory().percent)+"%")
                    # append_to_logfile(str(psutil.virtual_memory()))
                    print(psutil.virtual_memory())
                    print("\r")
                    if memory_report :
                        with open('memory_stats'+self.ORG_NAME+'-'+str(s+1)+'-'+str(separate_num)+'.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            # CSV ヘッダー
                            writer.writerow([
                                'render_time','images_size','total', 'available', 'used', 'percent', 
                                'active', 'inactive', 'wired', 'free'
                            ])
                            # データの書き込み
                            writer.writerows(memory_stats)
                    gc.collect()
        if self.cap != None and  seqrender == False: self.cap.release()
        if self.sepVideoOut != 1 and out_type != 0 and separate_num != 1:
            print("video-preference")
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
            self.out_videopath=videostr +'.mp4'
            if self.scan_direction%2==1 :
                video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
            else :
                video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
            # video = cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(self.width),int(self.height)),1) if XY_TransOut == False else cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(self.height),int(self.width)),1) 
            if tmp_type_para : 
                if render_mode == 0:    
                    render_start=0
                    render_end=wr_array.shape[0]
                elif render_mode == 1:
                    render_start=int(sep_start_num*wr_array.shape[0]/separate_num)
                    render_end= int(sep_end_num*wr_array.shape[0]/separate_num)
                for i in range(render_start,render_end):
                    if XY_TransOut :
                        if rotate_direction:
                            last_img=np.load("tmp/"+self.ORG_NAME+"_"+str(i)+".npy").transpose(1,0,2)[:,::-1]
                        else :
                            last_img=np.load("tmp/"+self.ORG_NAME+"_"+str(i)+".npy").transpose(1,0,2)[::-1]
                    else:
                        last_img=np.load("tmp/"+self.ORG_NAME+"_"+str(i)+".npy")             
                    video.write(last_img)
                    os.remove("tmp/"+self.ORG_NAME+"_"+str(i)+".npy")
                    ci=i
                    bar= '■'*int(ci*self.progressbarsize/wr_array.shape[0])+ "."*int((wr_array.shape[0]-ci)*self.progressbarsize/wr_array.shape[0])
                    print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/wr_array.shape[0]*100:.02f}%",end="")
            else:
                if render_mode == 0:
                    s_start = sep_start_num
                    s_end = separate_num
                elif render_mode == 1:
                    s_start = sep_start_num
                    s_end = sep_end_num
                total_last_imagecounts=0
                for s in range(s_start,s_end):
                    last_images = np.load("tmp/"+self.ORG_NAME+"_sep-"+str(s)+".npy")
                    # print(last_images.shape[0])
                    for i in range(last_images.shape[0]):
                        if XY_TransOut :
                            if rotate_direction:
                                video.write(last_images[i].transpose(1,0,2)[:,::-1])
                            else :
                                video.write(last_images[i].transpose(1,0,2)[::-1])
                        else:
                            video.write(last_images[i])
                        ci=i+total_last_imagecounts
                        bar= '■'*int(ci*self.progressbarsize/wr_array.shape[0])+ "."*int((wr_array.shape[0]-ci)*self.progressbarsize/wr_array.shape[0])
                        print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/wr_array.shape[0]*100:.02f}%",end="")
                    if tmp_save != True : os.remove("tmp/"+self.ORG_NAME+"_sep-"+str(s)+".npy")
                    total_last_imagecounts += last_images.shape[0]
            video.release()
            if tmp_save != True : os.rmdir("tmp")
        runOverTime = time.time()
        lnterval = round(runOverTime-runFirstTime,2)
        print("All Done",lnterval,"sec")
        append_to_logfile("All Done"+str(lnterval)+"sec")
        
    def pretransprocess(self,outnums=100,XY_TransOut=False):
        #self.outfpsはグローバルで定義
        out_type=1
        separate_num=1
        sep_start_num=0
        sep_end_num=1
        runFirstTime = time.time()
        XY_Name = "Y" if self.scan_direction%2 == 0 else "X"
        videostr = self.ORG_NAME+"_"+self.out_name_attr 
        if self.embedHistory_intoName == False :
            videostr = self.ORG_NAME+"_process"+str(self.log)
        rotate_direction = False
        if np.amin(self.data[:,:,-1])<0:
            print("z<0,error")
            return
        if np.amax(self.data[:,:,-1])>self.count:
            print("z>video_count,error",np.amax(self.data[:,:,-1]),self.count)
            return
        #audioようにfloatで計算していたのをintへ戻す。この方法だと小数点以下は切り捨て
        dim=int(self.data.shape[0] / outnums)
        beforefps=self.outfps
        self.outfps=10
        wr_array =  self.data[::dim].astype(np.int32)
        # self.data=wr_array
        s=0
        for i in range(0,int(wr_array.shape[0])):
            if self.scan_direction%2==1 :
                exec("img%d =  np.zeros((int(self.height),int(wr_array.shape[1]),3),np.uint8)" % (i))
            else :
                exec("img%d =  np.zeros((int(wr_array.shape[1]),int(self.width),3),np.uint8)" % (i))
        #ビデオの設定
        if self.sepVideoOut == 1 :
            print("video-preference")
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
            if self.scan_direction%2==1 :
                video = cv2.VideoWriter(videostr +'_pre.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'sep_index='+str(s)+'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
            else :
                video = cv2.VideoWriter(videostr +'_pre.mp4', fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'sep_index='+str(s)+'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
        interval_first = None
        block_num = 0
        print("minz=",np.amin(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1]))
        print("maxz",np.amax(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1]))
        minz = np.amin(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1])
        maxz = np.amax(wr_array[int(s*wr_array.shape[0]/separate_num):int((s+1)*wr_array.shape[0]/separate_num),:,1])
        totalnum=maxz-minz
        totalslits=wr_array.shape[0]/separate_num*wr_array.shape[1]
        slitscounter=0
        if totalnum == 0 : totalnum=1
        progresscale=self.progressbarsize/totalnum
        progressAllScale=self.progressbarsize/(round(self.count))
        writingTotalNum=int(wr_array.shape[0]/separate_num)
        writingScale = self.progressbarsize/writingTotalNum
        if minz < 0 :
            print("z<0,error")
            return
        num = minz
        print("count",self.count)
        print("writingTotalNum",writingTotalNum)
        print("num=",num)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, num)
        sstime = time.time()
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            reedfull_bool = num < (maxz+1)
            if ret == True and reedfull_bool :
                stime = time.time()
                if (num > minz or num == minz):
                    if self.scan_direction%2 == 0 :
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wr =np.array(wr_array[i])
                            indices = np.where(wr[:, 1] == num)
                            for p in indices[0]:
                                exec("img%d[p,:] = frame[wr[p,0],:]" % (i))
                            slitscounter += len(indices[0])
                    else :
                        for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                            wr =np.array(wr_array[i])
                            indices = np.where(wr[:, 1] == num)
                            for p in indices[0]:
                                exec("img%d[:,p] = frame[:,wr[p,0]]" % (i))
                            slitscounter += len(indices[0])
                num += 1
                etime = time.time()
                Interval = etime - stime
                if interval_first == None :
                    interval_first = Interval
                elif Interval < interval_first :
                    interval_first=Interval
                #progressbar
                bar= '■'*int((num-minz)*progresscale) + "."*int((totalnum-(num-minz))*progresscale)
                print(f"\r\033[K[\033[33m{bar}\033[39m] frame{(num-minz)/totalnum*100:.02f}%({minz}>{num}>{maxz}) Slits{slitscounter/totalslits*100:.02f}%({slitscounter}/{int(totalslits)}) : {round(Interval,2)}({round(interval_first,2)})sec/f",end="")
            else:
                print("\r")
                lbstime = time.time()
                Interval=lbstime-sstime
                if self.sepVideoOut == 1 or separate_num == 1:
                    print("video-preference",round(Interval,2))
                    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
                    self.out_videopath=videostr +'.mp4'
                    if self.scan_direction%2 == 1 :
                        video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(wr_array.shape[1]),int(self.height)),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(self.height),int(wr_array.shape[1])),1)
                    else :
                        video = cv2.VideoWriter(self.out_videopath, fourcc, self.outfps,(int(self.width),int(wr_array.shape[1])),1) if XY_TransOut == False else  cv2.VideoWriter(videostr +'.mp4', fourcc, self.outfps,(int(wr_array.shape[1]),int(self.width)),1)
                else : print("file Writing",round(Interval,2))
                for i in range(int(s*wr_array.shape[0]/separate_num),int((s+1)*wr_array.shape[0]/separate_num)):
                    wstime = time.time()
                    if XY_TransOut :
                        if rotate_direction: exec('video.write(img%d.transpose(1,0,2)[:,::-1])' %(i))
                        else : exec('video.write(img%d.transpose(1,0,2)[::-1])' %(i))
                    else: exec('video.write(img%d)' %(i))
                    exec("del img%d"%(i))
                    gc.collect()
                    wetime = time.time()
                    knterval = round(wetime-wstime,2)
                    ci=i-int(s*wr_array.shape[0]/separate_num)
                    bar= '■'*int(ci*writingScale)+ "."*int((writingTotalNum-ci)*writingScale)
                    if self.sepVideoOut == 1 or separate_num == 1:
                        print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                    else :
                        print(f"\r\033[K[\033[32m{bar}\033[39m] {ci/writingTotalNum*100:.02f}% ({knterval:.02f})sec/f",end="")
                if self.sepVideoOut == 1  or separate_num == 1 : video.release()
                break
        print("done:",s+1,"/",separate_num)
        print("\r")
        gc.collect()
        self.cap.release()
        runOverTime = time.time()
        lnterval = round(runOverTime-runFirstTime,2)
        print("All Done",lnterval,"sec")
        self.outfps=beforefps

    # レンダリング済みの映像ファイルと、軌道配列から、軌道の3D可視化アニメーションを作成する。
    def animationout(self,outFrame_nums=100,drawLineNum=250,dpi=200,out_fps=10):#outFrame_nums=50,out_fps=10
        runFirstTime = time.time()
        if self.data.shape[0]< outFrame_nums:
            outFrame_nums = self.data.shape[0]
        cap = cv2.VideoCapture(self.out_videopath)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)# 幅
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)# 高さ
        # count = cap.get(cv2.CAP_PROP_FRAME_COUNT)# 総フレーム数
        # fps = cap.get(cv2.CAP_PROP_FPS)# fps
        #audioようにfloatで計算していたのをintへ戻す。この方法だと小数点以下は切り捨て
        wr_array = self.data.astype(np.int32)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.style.use("bmh")
        ax.view_init(elev=25, azim=-40)
        mpl.rcParams['axes.facecolor'] = (1,1,1,0)
        mpl.rcParams['axes.edgecolor'] = (1,1,1,0)
        mpl.rcParams['axes.labelsize']= 12
        mpl.rcParams['axes3d.grid']=False
        mpl.rcParams['grid.color']="#dddddd"
        # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        xMin=0
        xMax=self.scan_nums if self.scan_direction == 1  else self.slit_length 
        # xMax=self.scan_nums
        zMin=np.amin(wr_array[:,:,1])
        zMax=np.amax(wr_array[:,:,1])
        ax.set_xlim(xMin, xMax) # x軸固定
        ax.set_ylim(zMin, zMax) # z軸固定
        yMin=0
        yMax =self.slit_length  if self.scan_direction == 1 else self.scan_nums        
        ax.set_xticks(np.arange(0,xMax+1,math.floor(xMax/2)))
        if (zMax-zMin)> 5000 : ax.set_yticks(np.arange(zMin,zMax+1,3000))
        elif  (zMax-zMin)> 3000:ax.set_yticks(np.arange(zMin,zMax+1,1500))
        elif  (zMax-zMin)> 1500:ax.set_yticks(np.arange(zMin,zMax+1,800))
        else: ax.set_yticks(np.arange(zMin,zMax+1,300))
        ax.set_zticks(np.arange(0,yMax+1,math.floor(yMax/2)))
        ax.zaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")
        ax.xaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")
        ax.yaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")
        animate_steps=math.floor(wr_array.shape[0]/outFrame_nums)
        slit_len_dim=math.floor((yMax/xMax)*drawLineNum)  
        slit_len_steps = math.ceil(yMax/ slit_len_dim)
        slit_len_dim = math.ceil(yMax / slit_len_steps)
        frame_scan_steps=math.ceil(int(width) / drawLineNum)  if self.scan_direction%2 == 1 else math.ceil(int(width) / slit_len_dim) 
        scan_steps=math.ceil(xMax / drawLineNum)  if self.scan_direction%2 == 1 else math.ceil(yMax / drawLineNum) 
        drawLineNum=wr_array[0,::math.ceil(xMax /drawLineNum),0].shape[0] if self.scan_direction == 1 else wr_array[0,::math.ceil(yMax /drawLineNum),0].shape[0] 
        frame_slit_len_steps = math.ceil(yMax/ slit_len_dim) if self.scan_direction==1 else math.ceil(yMax/ drawLineNum)
        Yarray=np.zeros((slit_len_dim,drawLineNum)) 
        Xarray=np.zeros((slit_len_dim,drawLineNum))
        Zarray=np.zeros((slit_len_dim,drawLineNum))
        num = 0
        if os.path.isdir("sequence")==False:
            os.makedirs("sequence")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True :
                if num%animate_steps == 0 :
                    if self.scan_direction == 1 :
                        refColorIMG=frame[::slit_len_steps,::frame_scan_steps,:]
                        colors = refColorIMG[::-1,:,::-1]/255 #画像の天地合わせと、BGRからRGBへの変換
                    else:
                        refColorIMG=np.rot90(frame)[::frame_scan_steps,::frame_slit_len_steps,:] #90度回転
                        colors = refColorIMG[::-1,::-1,::-1]/255 #画像の天地合わせと、BGRからRGBへの変換
                    for n in range(slit_len_dim):
                        Yarray[n]=np.full(drawLineNum,n*yMax/slit_len_dim) if self.scan_direction == 1 else wr_array[num,::scan_steps,0]
                    for n in range(slit_len_dim):
                        Xarray[n]=wr_array[num,::scan_steps,0] if self.scan_direction == 1 else np.full(drawLineNum,n*xMax/slit_len_dim) 
                    # for n in range(slit_len_dim):
                    #     Yarray[n]=np.full(drawLineNum,n*yMax/slit_len_dim) 
                    # for n in range(slit_len_dim):
                    #     Xarray[n]=wr_array[num,::scan_steps,1] 
                    for n in range(slit_len_dim):
                        Zarray[n]=wr_array[num,::scan_steps,1]
                    # if self.scan_direction%2 == 1 :
                    #     ax.plot_surface(Xarray,Zarray,Yarray,rstride=1,cstride=1,facecolors=colors)
                    # else :
                    #     ax.plot_surface(Yarray,Zarray,Xarray,rstride=1,cstride=1,facecolors=colors)
                    if colors.shape[0] != Xarray.shape[0]:
                        newcolors=np.zeros((Xarray.shape[0],Xarray.shape[1],3))
                        newcolors[:colors.shape[0],:,:]=colors
                        colors=newcolors

                    ax.plot_surface(Xarray,Zarray,Yarray,rstride=1,cstride=1,facecolors=colors,shade=False)
                    ax.set_xlim(xMin, xMax) # x軸固定
                    ax.set_ylim(zMin, zMax) # z軸固定
                    ax.set_xticks(np.arange(0,xMax+1,math.floor(xMax/2)))
                    # if (zMax-zMin)> 5000 : ax.set_yticks(np.arange(zMin,zMax+1,3000))
                    # elif  (zMax-zMin)> 3000:ax.set_yticks(np.arange(zMin,zMax+1,1500))
                    # elif  (zMax-zMin)> 1500:ax.set_yticks(np.arange(zMin,zMax+1,800))
                    # else: ax.set_yticks(np.arange(zMin,zMax+1,300))
                    ax.set_yticks(np.arange(zMin,zMax+1,math.floor((zMax-zMin)/3)))
                    ax.set_zticks(np.arange(0,yMax+1,math.floor(yMax/2)))
                    ax.zaxis.set_tick_params(labelsize=9,color="#777777",labelcolor="#777777")
                    ax.xaxis.set_tick_params(labelsize=9,color="#777777",labelcolor="#777777")
                    ax.yaxis.set_tick_params(labelsize=9,color="#777777",labelcolor="#777777")
                    # ax.xaxis.line.set_color('#cccccc')
                    # ax.yaxis.line.set_color('#cccccc')
                    # ax.zaxis.line.set_color('#cccccc')
                    # ax.xaxis.set_tick_params(color='none',labelcolor='none')
                    # ax.yaxis.set_tick_params(color='none',labelcolor='none')
                    # ax.zaxis.set_tick_params(color='none',labelcolor='none')
                    ax.xaxis.line.set_color('none')
                    ax.yaxis.line.set_color('none')
                    ax.zaxis.line.set_color('none')
                    ax.grid(True, gridcolor='#777777')
                    # grid_alpha = 0.2  # 透明度の値（0に近いほど透明）
                    # グリッドを表示
                    # ax.grid(True)
    
                    # 現在の視点の角度を取得する
                    # current_elev = ax.elev
                    current_azim = ax.azim
                    # ax.view_init(elev=25, azim=current_azim+(30/outFrame_nums))
                    
                    # figvideo.write(plt)
                    plt.savefig('sequence/'+str(self.log)+'_3dPlot_pixColor-'+str(num)+'.png',dpi=dpi,transparent=True,bbox_inches='tight')
                    plt.cla()
                num += 1
                ci=num
                bar= '■'*int(ci*self.progressbarsize/wr_array.shape[0])+ "."*int((wr_array.shape[0]-ci)*self.progressbarsize/wr_array.shape[0])
                print(f"\r\033[K[\033[31m{bar}\033[39m] {ci/wr_array.shape[0]*100:.02f}%",end="")
            else:
                break
        cap.release()
        img=cv2.imread('sequence/'+str(self.log)+'_3dPlot_pixColor-0.png')
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
        if self.embedHistory_intoName : 
            figvideo = cv2.VideoWriter(self.ORG_NAME+"_"+self.out_name_attr +'_img_3d-pixelMap.mp4',fourcc,out_fps,(int(img.shape[1]),int(img.shape[0])),1) 
        else :
            figvideo = cv2.VideoWriter(self.ORG_NAME+"process"+str(self.log)+'_img_3d-pixelMap.mp4',fourcc,out_fps,(int(img.shape[1]),int(img.shape[0])),1) 
        for i in range(0,wr_array.shape[0],animate_steps):
            figvideo.write(cv2.imread('sequence/'+ str(self.log) +'_3dPlot_pixColor-'+str(i)+'.png'))
            # os.remove(self.ORG_NAME+"_"+self.out_name_attr +'_3dPlot_pixColor-'+str(i)+'.png')
        figvideo.release()
        runOverTime = time.time()
        lnterval = round(runOverTime-runFirstTime,2)
        print("All Done",lnterval,"sec")

    # dataに新たな軌跡を加えて返す関数
    def addInterpolation(self,frame_nums,i_direction,z_direction,axis_position,s_reversal=0,z_reversal=0,cycle_degree=90,extra_degree=0,zslide=0,speed_round = True,rrange=[0,1],zscale=1):
        self.interpolation(frame_nums,i_direction,z_direction,axis_position,s_reversal,z_reversal,cycle_degree,extra_degree,zslide,speed_round,rrange,zscale) 
        self.maneuver_log("+IP"+str(frame_nums)+"(ID"+str(i_direction)+"-ZD"+str(z_direction)+"-AP"+str(axis_position)+"-SREV"+str(s_reversal)+"-ZREV"+str(z_reversal)+")")

    def interpolation(self,frame_nums,i_direction,z_direction,axis_position,s_reversal=0,z_reversal=0,cycle_degree=90,extra_degree=0,zslide=0,speed_round = True,rrange=[0,1],zscale=1):
        wr_array= [] 
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
    def rootingA_interporation(self,FRAME_NUMS,loop_num=2,axis_first_p=0,speed_round=True):
        ap=axis_first_p
        r=0
        for i in range(loop_num):
            self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=0,z_direction=0,axis_position=ap%2,reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round)
            r+=1
            self.interpolation(int(FRAME_NUMS/(loop_num*2)),i_direction=1,z_direction=1,axis_position=ap%2,reversal=r%2,zslide=(self.scan_nums-1)*i,speed_round = speed_round)
            ap+=1
        self.maneuver_log("IP"+str(FRAME_NUMS)+"(rootingA)")
          
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
        if len(self.data) != 0 : 
            extra_array = np.zeros((self.data.shape[0]+frame_nums,self.scan_nums,2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
            extra_array[0:self.data.shape[0]] = self.data
            outroFrameXP = self.data[-1,:,0]
            outroFrameZ = self.data[-1,:,1]
        else : 
            extra_array = np.zeros((frame_nums,self.scan_nums,2),dtype=np.float64)#audioへの適合もあるため、この時点ではビットレートを高くして計測
            outroFrameXP = np.full(self.scan_nums,start_line*(self.scan_nums-1)) 
            outroFrameZ = np.arange(0,self.scan_nums* zscale,1 * zscale) if zd else np.arange(self.scan_nums * zscale -1, -1, -1 * zscale)
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
    def addKeepSpeedTrans(self,frame_nums,under_xyp,over_xyp=1,rendertype=0):
        # rendertype１ の場合は、配列ごとに移動の差分ベクトルを取得する。0は平均のベクトルを出す。
        print(sys._getframe().f_code.co_name)
        outroFrameZ = self.data[-1,:,1]
        #xp,ypの空間領域(SpaceDomain)の差分計算
        vector_xyp = self.data[-1,:,0]-self.data[-2,:,0] if rendertype != 0 else np.mean( self.data[-1,:,0]-self.data[-2,:,0] )
        prevector_xyp = self.data[-2,:,0]-self.data[-3,:,1] if rendertype != 0 else np.mean( self.data[-2,:,0]-self.data[-3,:,1] )
        acceleration_xyp = vector_xyp/prevector_xyp 
        #zpの差分計算
        vector_zp = self.data[-1,:,1]-self.data[-2,:,1] if rendertype != 0 else np.mean( self.data[-1,:,1]-self.data[-2,:,1] )
        prevector_zp = self.data[-2,:,1]-self.data[-3,:,2] if rendertype != 0 else np.mean( self.data[-2,:,1]-self.data[-3,:,2] )
        acceleration_zp = vector_zp/prevector_zp 
        # print(acceleration)
        normalFrame=np.arange(0,self.scan_nums)
        n=0
        while np.amax(self.data[-1,:,0]) < under_xyp :
            newframe=np.zeros((1,self.scan_nums,2),dtype=np.float64)
            vector_xyp=vector_xyp*acceleration_xyp
            vector_zp=vector_zp*acceleration_zp
            # print(n,vector_xyp)
            newframe[:,:,0] = self.data[-1,:,0] + vector_xyp
            newframe[:,:,1] = self.data[-1,:,1] + vector_zp
            if np.amax(newframe[:,:,0]) > under_xyp:
                break
            if np.amin(newframe[:,:,0]) < over_xyp:
                break
            self.data=np.vstack((self.data,newframe))
            n+=1
            if n == frame_nums:
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
    # self.data、に対して、after_arrayで受け取った配列の間を滑らかに補う
    def insertKeepSpeedTrans(self,frame_nums,under_xyp=None,over_xyp=1,after_array=[],rendertype=0):
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
                after_array[:,:,2]+=deftime
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
        np.save(self.ORG_NAME+'_raw.npy',self.data)
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
        wr_array=[]
        for i in range(0,frame_nums):
            crad = math.radians(extra_degree)+math.radians(cycle_degree)* i / (frame_nums-1) if speed_round == False else math.radians(extra_degree)+math.radians(cycle_degree)*(1-(math.cos(math.radians(i/(frame_nums-1)*180))+1.0)/2)
            csin=math.sin(crad)
            ccos=math.cos(crad)
            write_array=[]
            if self.scan_direction%2 == 0:
                for y in range(0,int(self.height)):
                    pernum = (y-(self.height/2))/self.height
                    yp = self.height/2 + ccos*(self.height-1)*pernum if spaceflow else y
                    zp = zslide -csin*maxz*pernum
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
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1].split("Trans")[0]+str(cycle_degree)+"deg-zscale"+str(t_auto_scaling)+"-"+str(frame_nums)+"f"+spf_attr)

    # 回転の中心軸を漸次的に変化させる
    #extradegree=90,180,270などの設定の時に、初期値（i=0）が数値が飛ぶ。i=0のときに、うまくshift_num=0で始まってi=1のときにshift_num=1にするようにしたいが、いろいろ調整が難しそう。現状90.1あるいは89などの数値をextradegreeにはめてあげると解決策になる
    def addCustomCycleTrans(self,frame_nums,cycle_degree,start_center=1/2,end_center=1/2,t_auto_scaling=False,extra_degree=0,speed_round = True,zslide=0,auto_zslide=True,t_auto_scaling_num=0.9,zscale=1,spaceflow=True):
        print(sys._getframe().f_code.co_name)
        permit_auto_zslide=False
        if zslide == 0:
            if auto_zslide and len(self.data)>0:
                permit_auto_zslide=True
        wr_array=[]
        defcenter=end_center-start_center
        csinarray=[]
        ccosarray=[]

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
                    if permit_auto_zslide : zslide=self.data[-1:y:1]
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
            write_array = np.array(write_array)
            wr_array.append(write_array)
            self.cycle_axis.append(xcenter)
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
        timevalues:読み込む時間軸方向のピクセル数    
        timepoints=書き出す総フレームに対する時間軸のキーフレーム  
    wave_type=0:サイン波
    wave_type=1:三角波      
    '''
    def addBlowupTrans(self,frame_nums,deg=360,speed_round = True,connect_round=1,timevalues=[],timepoints=[],timecenter=[],extra_degree=0,wave_type=0):
        #2021.09.02　New。
        #2022.09.19 プロセスをcycletransと同等に
        """
        blowupの動きをキーフレームにより制御する
        timevalues:読み込む時間軸方向のピクセル数
        timepoints=書き出す総フレームに対する時間軸のキーフレーム
        """
        if len(timevalues)==0:
            timevalues=[self.count,self.scan_nums,1,0]#左端から右端までの時間差（Frame）
            timepoints=[0,0.7,0.95,1]
            if len(timecenter)== 0 : timecenter=[0.5,0.5,0.5,0.5]
        else : 
            if timevalues[0] > self.count : timevalues[0] = self.count
            #もし、timepoints が設定されていなければ、自動的に入力する
            if len(timevalues)!=len(timevalues):
                for i in range(0,len(timevalues)):timepoints.append(1/(len(timevalues)-1)*i)
            if len(timecenter) == 0:
                timecenter=np.full(len(timevalues), 0.5)

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
            ts = ts if connect_round == 0 else ((math.sin(math.radians(ts*180-90)))/2+0.5)
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
                    write_array.append([xp,zp])
            else:
                yp = (self.height-1)-(zcos*(self.height-1)/2+(self.height-1)/2)
                for y in range(0,int(self.height)):#Xは固定でzポイントが飛び飛びから詰まっていく。
                    zp=pre_front_point+y*ajstlen
                    write_array.append([yp,zp])
            wr_array.append(write_array)
        if len(self.data) != 0: self.data = np.vstack((self.data,np.array(wr_array)))
        else: self.data=np.array(wr_array)
        self.maneuver_log((sys._getframe().f_code.co_name).split("add")[1].split("Trans")[0]+str(frame_nums)+"-deg"+str(deg))

# リストを最下層の要素数が均等になるように分割
def split_list_based_on_elements(original_list, k):
    total_elements = sum([len(sublist) for sublist in original_list])
    average_elements = total_elements / k
    # Splitting original_list
    split_lists = []
    current_list = []
    current_count = 0
    residue = 0  # 余りの要素数

    for sublist in original_list:
        if current_count + len(sublist) <= average_elements + residue:
            current_list.append(sublist)
            current_count += len(sublist)
        else:
            split_lists.append(current_list)
            residue += current_count - average_elements
            current_list = [sublist]
            current_count = len(sublist)
            if len(split_lists) == k - 1:
                split_lists.append(original_list[original_list.index(sublist):])
                return split_lists

    if current_list:
        split_lists.append(current_list)

    return split_lists

def split_array(arr, N):
    avg_len = len(arr) // N
    remain = len(arr) % N
    splits = []

    i = 0
    for _ in range(N):
        split_len = avg_len + (1 if remain > 0 else 0)
        splits.append(arr[i:i+split_len].tolist())
        i += split_len
        remain -= 1

    return splits

def split_uniq_based_on_original(uniq_list, split_original_list):
    split_uniq = []
    index = 0
    for split_list in split_original_list:
        split_len = len(split_list)
        if isinstance(uniq_list, np.ndarray):
            split_uniq.append(uniq_list[index:index+split_len].tolist())
        else:
            split_uniq.append(uniq_list[index:index+split_len])
        index += split_len
    return split_uniq

# パス内の映像ファイルを抽出して配列を返す
def addmovfile(prepath):
    file_list=glob.glob(prepath+"/*.MOV")
    file_list.extend(glob.glob(prepath+"/*.mp4"))
    file_list.extend(glob.glob(prepath+"/*.mov"))
    file_list.extend(glob.glob(prepath+"/*.MP4"))
    print(file_list)
    return file_list

def closest_value(start_value, end_value, num_values,zRange):
    log_start = np.log10(start_value)
    log_end = np.log10(end_value)
    rounded_values = []
    for i in range(num_values):
        log_value = np.interp(i, [0, num_values - 1], [log_start, log_end])
        value = int(round(10 ** log_value, -int(np.floor(log_value))))
        rounded_values.append(value)
    return min(rounded_values, key=lambda x: abs(x - zRange))
#blowupTransのTimePoint,timevalueの値からの調整
def search(num,frame_nums,timepoints):
        vvv=0
        for h in range(0,len(timepoints)-1):
            top=timepoints[h]*frame_nums
            bottom=timepoints[h+1]*frame_nums
            if num >= top and num < bottom:
                vvv=h
        return vvv

def addCsvHeader(d):
    print(sys._getframe().f_code.co_name)
    # 列と行のヘッダーを定義
    column_header = [ i for i in range(d.shape[1]+1)]
    column_footer = [ i for i in range(d.shape[1]+1)]
    column_footer[1:5]=[np.min(d),np.max(d),np.max(d)-np.min(d),d.shape[0]]
    row_header = [i for i in range(d.shape[0])]
    # 列と行のヘッダーを追加した新しい配列を作成
    d=np.hstack((np.array(row_header).reshape(-1, 1), d))
    d = np.vstack((np.array(column_header), d))
    d = np.vstack((d,np.array(column_footer)))
    return d

#再生断面軌道の編集のログを出力。2023.10.19 added
def append_to_logfile(text_to_append):
        with open("maneuverlog.txt", "a") as file:
            file.write(text_to_append + "\n")

def bezier_interpolation(p0, p1, p2, t):
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2


def save_video_frames(video_path, img_format='jpg',frame_array=None):
      # 現在のディレクトリを取得
    current_directory = os.getcwd()

    # 新しいフォルダ名を生成
    if frame_array is None:
    
        # ビデオパスからディレクトリとファイル名を抽出
        directory, filename = os.path.split(video_path)
        filename_without_ext = os.path.splitext(filename)[0]

        # 新しいフォルダ名を生成
        output_folder = os.path.join(directory, f"{filename_without_ext}_frames_{img_format}")
    else:
        output_folder = os.path.join(current_directory, "seq")

    # 出力フォルダを作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
     # ビデオを読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None
    # 総フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 総フレーム数を取得して桁数を計算
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    digits = len(str(total_frames))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 特定のフレームのみを保存するか、すべてのフレームを保存するかを判断
        if frame_array is None or frame_count in frame_array:
            # ファイル名の生成
            if frame_array is None:
                # ファイル名にゼロパディングを適用
                frame_filename = f'frame_{str(frame_count).zfill(digits)}.{img_format}'
            else:
                frame_filename = f'frames_{frame_count}.{img_format}'
            # 保存処理 
            output_path = os.path.join(output_folder, frame_filename)
            if img_format == 'npy':
                np.save(output_path, frame)
            else:
                cv2.imwrite(output_path, frame)

        frame_count += 1

        # 進捗状況をパーセンテージで表示
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.2f}%", end='\r')

    cap.release()
    print(f"\nTotal {frame_count} frames saved.")
    # 新しく作成したディレクトリのパスを返す
    return output_folder

def convert_npy_to_jpg(npy_file_path, output_folder):
    # Numpy ファイルを読み込む
    data = np.load(npy_file_path)
    
    # 出力ファイル名を作成（拡張子を .jpg に変更）
    output_file_name = npy_file_path.split('/')[-1].replace('.npy', '.jpg')
    output_path = f"{output_folder}/{output_file_name}"
    
    # BGR 形式に変換して JPG として保存
    cv2.imwrite(output_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
    
    return output_path

def extract_number(filename):
    # ファイル名から連番部分を抽出
    match = re.search(r'_(\d+)\.npy$', filename)
    return int(match.group(1)) if match else -1

def convert_npys_to_video(input_folder, output_folder, fps=30):
    # 指定フォルダ内のすべての .npy ファイルを取得し、連番でソート
    npy_files = sorted(glob.glob(os.path.join(input_folder, '*.npy')), key=extract_number)

    # 指定フォルダ内の最初の .npy ファイルを取得
    first_npy_file = next(glob.iglob(os.path.join(input_folder, '*.npy')), None)
    if not first_npy_file:
        raise ValueError("No .npy files found in the specified folder.")

    # 最初のファイルから解像度を取得
    first_frame = np.load(first_npy_file)
    resolution = (first_frame.shape[1], first_frame.shape[0])

    # 出力ファイルパスを設定
    # output_path = os.path.join(input_folder, output_file)
    # 出力ファイル名を作成（拡張子を .jpg に変更）
    output_path = f"{output_folder}/tmp.mp4"

    # 映像ファイルのライターを初期化
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#コーデック指定
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    # 指定フォルダ内のすべての .npy ファイルを取得
    total_files = len(npy_files)
    # print(npy_files)

    for index, file in enumerate(npy_files):
        # Numpy ファイルを読み込む
        frame = np.load(file)

        # 映像にフレームを追加
        video_writer.write(frame)

        # 進捗を表示
        progress = (index + 1) / total_files * 100
        print(f"Processing {index + 1}/{total_files} ({progress:.2f}%)")


    # ライターを閉じる
    video_writer.release()

    return output_path


def custom_blur(data, s_frame, e_frame, bl_time, dim_num=1):
    time_array = data[:,:,dim_num]
    blur_array = np.zeros([e_frame - s_frame, data.shape[1]], dtype=np.float64)

    # ブラー時間が短い場合の調整
    if e_frame - s_frame < bl_time :
        bl_time = (e_frame - s_frame) 

    for y in range(e_frame - s_frame):
        # フレームの位置に応じてブラー適応範囲を動的に決定
        if y < bl_time / 2:
            # s_frameに近い場合は、範囲を縮小
            apply_bl_time = y
        elif (e_frame - s_frame) - y - 1 < bl_time / 2:
            # e_frameに近い場合は、範囲を縮小
            apply_bl_time = (e_frame - s_frame) - y - 1
        else:
            apply_bl_time = bl_time // 2

        start_index =int(max(s_frame + y - apply_bl_time, 0))
        end_index = int(min(s_frame + y + apply_bl_time + 1, time_array.shape[0]))
        blur_range = time_array[start_index:end_index, :]
        # print(y,apply_bl_time,start_index,end_index)
        blur_array[y, :] = np.mean(blur_range, axis=0)

    # 更新されたブラー配列を元のデータに代入
    data[s_frame:e_frame,:,dim_num] = blur_array
    return data
if __name__ == '__main__':
    print("hello,ver=1.00")
    # 定義されている関数のリストを取得
    functions = [func for func in locals().values() if inspect.isfunction(func)]

    # 関数の名前を表示
    for func in functions:
        print(func.__name__)