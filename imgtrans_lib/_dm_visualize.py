"""drawManeuver の可視化 (VisualizeMixin)

self.data や派生統計を 2D/3D で図示し、PNG/動画として出力する系統:
- maneuver_log                          : maneuver_log テキスト出力
- maneuver_CSV_out / info_setting       : CSV/メタ情報出力
- scd_out                               : SCD (シーン解析) 動画出力
- maneuver_2dplot                       : 主要な 2D プロット
- maneuver_3dplot / maneuver_3dplot_midtide : 3D アニメ/中潮 3D 出力
- maneuver_imgplot                      : 各種マップを 16bit PNG / 動画出力
- animationout / animationout_custome   : 任意の 3D アニメ出力
- movement_intensity_analyze            : 動き強度の解析+可視化

NOTE: 旧版 maneuver_3dplot_old / コメントアウト旧版 maneuver_imgplot は削除済み。
"""
import os
import gc
import math
import sys
import time
import json
import subprocess
import shutil
import inspect
import psutil
import cv2
import numpy as np
import av
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D

from ._utils import (
    frames_to_min_sec,
    append_to_logfile,
    addCsvHeader,
    create_video_from_images,
    calculate_parallel_perpendicular,
)
from ._hdr import _tonemap_hdr_rgb01_to_srgb, _probe_video_transfer


class VisualizeMixin:
    def maneuver_log(self,module_name_attr):
        self.log+=1
        self.out_name_attr+="_"+module_name_attr
        if len(self.out_name_attr) > 100:#文字数が255を超えるとエラーとなる問題
            self.out_name_attr = self.out_name_attr[:97] + "..."+str(self.log)
        append_to_logfile(str(self.log)+":"+module_name_attr)
        if self.auto_visualize_out : 
            if self.default_debugmode : self.maneuver_2dplot(debugmode=True)
            else : self.maneuver_2dplot()

    def maneuver_CSV_out(self,thread_num=None,time_map=True,space_map=True,time_rate_map=True,now_depth_map=False,space_rate_map=False,movement_rate_map=False):
        print(sys._getframe().f_code.co_name)
        if thread_num != None : 
            if thread_num != 0 :
                self.info_setting(thread_num)
        else:
            if self.infolog == 0 :self.info_setting() 
            elif self.infolog != self.log :self.info_setting(self.sc_resetPositionMap.shape[0])
        print(sys._getframe().f_code.co_name)
        j=self.sc_resetPositionMap.shape[0]
        #AEでの描画よう
        if time_map : np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_ResetP_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_resetPositionMap.astype(np.int32)),delimiter=',')
        if time_rate_map : np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_Rate_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_rateMap),delimiter=',')
        if space_map : np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_inPanMap_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_inPanMap.astype(np.int32)),delimiter=',')
        if now_depth_map : np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_nowDepth_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_now_depth),delimiter=',') 
        if space_rate_map : np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'_sc_SDRateMap_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_SDRateMap),delimiter=',')       
        if movement_rate_map : np.savetxt(self.ORG_NAME+"_"+self.out_name_attr +'sc_movementRateMap_AE-'+str(j)+'thread.csv',addCsvHeader(self.sc_movementRateMap),delimiter=',')       
        
        # 表計算ソフトで作図用
        # np.savetxt(self.ORG_NAME+"_"+self.out_name_attr+'_ResetP-transpose.csv',self.sc_resetPositionMap.transpose(),delimiter=',')
        # np.savetxt(self.ORG_NAME+"_"+self.out_name_attr+'_Rate-transpose.csv',self.sc_rateMap.transpose(),delimiter=',')
        # np.savetxt(self.ORG_NAME+"_"+self.out_name_attr+'_inPanMap-transpose.csv',self.sc_inPanMap.transpose(),delimiter=',')
        
    def info_setting(self,thread_num=20,raw=False):
        if raw:
            print("info_setting,分割しない")
            self.sc_resetPositionMap = np.vstack((self.data[:,:,1].transpose(),self.data[:,-1,1]))#transpose()で次元入れ替え、"[::a]"で間引き
            self.sc_rateMap = np.zeros(self.sc_resetPositionMap.shape,np.float32)#再生レートの配列
            self.sc_inPanMap = np.vstack((self.data[:,:,0].transpose(),self.data[:,-1,0]))#入力panの配列
            self.sc_SDRateMap = np.zeros(self.sc_resetPositionMap.shape,np.float32)#Spatial Direction Rate の配列
            self.sc_movementRateMap = np.zeros(self.sc_resetPositionMap.shape,np.float32)#Total Movement Rate の配列
            self.sc_parallel_component_Map = np.zeros(self.sc_resetPositionMap.shape,np.float32)#Total Movement Rate の配列
            self.sc_perpendicular_component_Map = np.zeros(self.sc_resetPositionMap.shape,np.float32)#Total Movement Rate の配列

        else:
            sep = np.ceil(self.data.shape[1]/(thread_num-1)).astype(int)
            print("info_setting,分割:",sep)
            self.sc_resetPositionMap = np.vstack((self.data[:,::sep,1].transpose(),self.data[:,-1,1]))#transpose()で次元入れ替え、"[::a]"で間引き
            self.sc_rateMap = np.zeros(self.sc_resetPositionMap.shape,np.float32)#再生レートの配列
            self.sc_inPanMap = np.vstack((self.data[:,::sep,0].transpose(),self.data[:,-1,0]))#入力panの配列
            self.sc_SDRateMap = np.zeros(self.sc_resetPositionMap.shape,np.float32)#Spatial Direction Rate の配列
            self.sc_movementRateMap = np.zeros(self.sc_resetPositionMap.shape,np.float32)#Total Movement Rate の配列
            self.sc_parallel_component_Map = np.zeros(self.sc_resetPositionMap.shape,np.float32)#Total Movement Rate の配列
            self.sc_perpendicular_component_Map = np.zeros(self.sc_resetPositionMap.shape,np.float32)#Total Movement Rate の配列

        for i in range(self.sc_resetPositionMap.shape[0]):
            for k in range(self.sc_resetPositionMap.shape[1]):
                if((k+1)<self.sc_resetPositionMap.shape[1]):
                    #前後の差分から変化率を計測して再生レートを計算
                    self.sc_rateMap[i,k]=(self.sc_resetPositionMap[i,k+1]-self.sc_resetPositionMap[i,k])/(self.recfps/self.outfps)
                else:
                    self.sc_rateMap[i,k]=self.sc_rateMap[i,k-1]

        for i in range(self.sc_inPanMap.shape[0]):
            for k in range(self.sc_inPanMap.shape[1]):
                if((k+1)<self.sc_inPanMap.shape[1]):
                    #前後の差分から変化率を計測して再生レートを計算 空間方向は虚の時間の流となるので、絶対値とする
                    # self.sc_SDRateMap[i,k]=abs((self.sc_inPanMap[i,k+1] - self.sc_inPanMap[i,k])/(self.recfps/self.outfps))
                    #絶対値でなくてよいのでは？2024/10/21
                    self.sc_SDRateMap[i,k]=(self.sc_inPanMap[i,k+1] - self.sc_inPanMap[i,k])/(self.recfps/self.outfps)
                else:
                    self.sc_SDRateMap[i,k]=self.sc_SDRateMap[i,k-1]
        for i in range(self.sc_inPanMap.shape[0]):
            for k in range(self.sc_inPanMap.shape[1]):
                if((k+1)<self.sc_inPanMap.shape[1]):
                    #前後の差分から変化率を計測して再生レートを計算
                    self.sc_movementRateMap[i,k]=math.sqrt((self.sc_resetPositionMap[i,k+1]-self.sc_resetPositionMap[i,k])**2+(self.sc_inPanMap[i,k+1] - self.sc_inPanMap[i,k])**2)/(self.recfps/self.outfps)
                else:
                    self.sc_movementRateMap[i,k]=self.sc_movementRateMap[i,k-1]
        
        #実験用。リバーブなどに適応する。また、fpsの異なる複数の映像ソースを適宜選択しながらレンダリングする際の、選択判断の材料とする。
        self.sc_now_depth =np.zeros(self.data.shape[0],np.float32)#フレーム内に収まる時間の幅
        for i in range(self.sc_now_depth.shape[0]):
            self.sc_now_depth[i]=abs(np.amax(self.data[i,:,1])-np.amin(self.data[i,:,1]))
        

        #2024.10追加　ベクトルの成分を、基準とする再生断面に対して明確に平行成分と垂直成分に分ける
        previousAB=[1,1]
        for i in range(self.sc_resetPositionMap.shape[0]):
            for k in range(self.sc_resetPositionMap.shape[1]):
                if((k+1)<self.sc_resetPositionMap.shape[1]):
                    # 座標を取得
                    x1 = self.sc_inPanMap[i, k]
                    x2 = self.sc_inPanMap[i+1, k] if(i+1)<self.sc_resetPositionMap.shape[0] else self.sc_inPanMap[i-1, k] 
                    y1 = self.sc_resetPositionMap[i, k]
                    y2 = self.sc_resetPositionMap[i+1, k] if(i+1)<self.sc_resetPositionMap.shape[0] else self.sc_resetPositionMap[i-1, k] 
                    y4 = self.sc_resetPositionMap[i, k+1]
                    x4 = self.sc_inPanMap[i, k+1]
                    # 点A, B, qを定義
                    A = np.array([x1, y1])
                    B = np.array([x2, y2])
                    q = np.array([x4, y4])


                    # 並行成分と垂直成分を計算
                    # if self.log==3:
                    #     print(self.sc_resetPositionMap.shape[0],i,k,A,B,q,calculate_parallel_perpendicular(A,B,q))
                    self.sc_parallel_component_Map[i,k],self.sc_perpendicular_component_Map[i,k]=calculate_parallel_perpendicular(A,B,q,previousAB)
                    previousAB = B-A
                else:
                    self.sc_parallel_component_Map[i,k]=self.sc_parallel_component_Map[i,k-1]
                    self.sc_perpendicular_component_Map[i,k]=self.sc_perpendicular_component_Map[i,k-1]
                
                
        #array to adjust : 
        array_to_adjust = self.recfps/(np.clip(self.sc_now_depth,1, None)/self.data.shape[1])#np.clipで1以上の数値とする。除算のエラー対策
        # 各要素に対して、より大きい近似値を選ぶ
        # np.digitizeを使用して、各要素が指定されたビンのどこに属するかを見つける
        if len(self.some_recfps_array)>0:
            indices = np.digitize(array_to_adjust,  np.array(self.some_recfps_array), right=True)
            # インデックスが配列の長さを超えないように調整
            indices[indices == len(self.some_recfps_array)] = len(self.some_recfps_array) - 1
            # 近似値配列から対応する要素を取得
            self.depth_to_sel_recfps = np.array(self.some_recfps_array)[indices]
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
            print("audio_form_outは、一時的に非対応、librosaライブラリを読み込む必要あり")
            '''
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
            plt.clf()
            '''

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
    
    # axnum は0-7 レンダリングするグラフの数
    def maneuver_2dplot(self, thread_num=None,thread_through=False, debugmode=False, normal_line_draw=False, w_inc=None, h_inc=None,
                    plinewidth=1.0, individual_output=False, timeflow_scaling=True, palpha=1.0,
                    x_positions=[], y_positions=[], s_frame=None, e_frame=None, axnum = 3,label=True,ax_title=True,colormode='black',add_x_frames=0,attr=None,custome_label=False,rate_lange_lim=None,ax1ylim=None,ax2ylim=None,ax3ylim=None,ax4ylim=None,ax5ylim=None,ax6ylim=None,ax7ylim=None,video_out=False,video_alpha=False,video_tc_overlay=True,space_axis_invert=False,video_height_px=None):
        print(sys._getframe().f_code.co_name)
        if w_inc is None:
            w_inc = self.plot_w_inc
        if h_inc is None:
            h_inc = self.plot_h_inc
        if attr is None:
            attr = ""
        def remove_edge_labels(ax):
            # print("remove")
            """Remove edge labels from the axes."""
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()

            # x軸の目盛りラベルを取得してエッジ部分を削除
            labels = [item.get_text() for item in ax.get_xticklabels()]
            # print(labels)
            if labels:
                labels[0] = ''
                labels[1] = ''
                labels[-1] = ''
                labels[-2] = ''
                ax.xaxis.set_major_locator(FixedLocator(xticks))
                ax.xaxis.set_major_formatter(FixedFormatter(labels))

            # y軸の目盛りラベルを取得してエッジ部分を削除
            labels = [item.get_text() for item in ax.get_yticklabels()]
            if labels:
                labels[0] = ''
                labels[1] = ''
                labels[-1] = ''
                labels[-2] = ''
                ax.yaxis.set_major_locator(FixedLocator(yticks))
                ax.yaxis.set_major_formatter(FixedFormatter(labels))

        def set_variable_locator(ax, axis, num_ticks):
            """Set a variable number of ticks on the given axis."""
            if axis == 'x':
                ax.xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
            elif axis == 'y':
                ax.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
            

        if thread_through == False:    
            if thread_num is not None or thread_num == 0:
                if thread_num == 0:
                    self.info_setting(raw=True)
                else:
                    self.info_setting(thread_num)
            else:
                if self.infolog == 0:
                    self.info_setting()
                elif self.infolog != self.log:
                    self.info_setting(self.sc_resetPositionMap.shape[0])

        j = self.sc_resetPositionMap.shape[0]
        cmap = LinearSegmentedColormap.from_list('original', [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)], N=j)
        color_map = plt.get_cmap(cmap)

        full_range = self.data.shape[0]
        s_frame = 0 if s_frame is None else s_frame
        e_frame = full_range if e_frame is None else e_frame
        x_range = np.arange(s_frame, e_frame)

        plt.style.use("bmh")
        
        if colormode == 'white':
            text_color = (1, 1, 1)
            edge_color = (0.3,0.3,0.3, 1)
            grid_color = (0.3,0.3,0.3, 1)
        else:
            text_color = (0, 0, 0)
            edge_color = (0, 0, 0, 0.5)
            grid_color = "#aaaaaa"

        mpl.rcParams['axes.facecolor'] = (1, 1, 1, 0)
        mpl.rcParams['axes.edgecolor'] = edge_color
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['axes.labelcolor'] = text_color
        mpl.rcParams['xtick.color'] = text_color
        mpl.rcParams['ytick.color'] = text_color
        mpl.rcParams['grid.color'] = grid_color
        mpl.rcParams['text.color'] = text_color
        mpl.rcParams['legend.fontsize'] = 'small'
        mpl.rcParams['legend.facecolor'] = 'inherit'
        mpl.rcParams['legend.edgecolor'] = 'inherit'
        mpl.rcParams['legend.labelcolor'] = text_color

        label_green_atr = 'LeftSide' if self.scan_direction == 1 else 'TopSide'
        label_red_atr = 'RightSide' if self.scan_direction == 1 else 'BottomSide'
        # label_space_atr = 'X-Space Flow(px) ' if self.scan_direction == 1 else 'Y-Space Flow(px)'
        label_space_atr = 'Space(px)'
        
        label_X_atr = 'boxel(X)' if self.scan_direction == 1 else 'boxel(y)'

        # frame to min : sec
        formatter_x = ticker.FuncFormatter(lambda x, pos: frames_to_min_sec(x + add_x_frames, self.outfps))
        formatter_y = ticker.FuncFormatter(lambda y, pos: frames_to_min_sec(y, self.recfps))

        rate_info = f"1 = {self.recfps:.0f}box/sec({self.recfps:.0f}frame)"
        fps_info = f"source: {frames_to_min_sec(self.count,self.recfps)}({self.recfps:.2f}fps), output:{frames_to_min_sec(self.data.shape[0],self.outfps)}({self.outfps:.2f}fps)"

        if individual_output:
            # グラフにFPS情報を追加する文字列
            rate_info =f"1 = {self.recfps:.0f}box/sec({self.recfps:.0f}frame)" 
            fps_info = f"source: {frames_to_min_sec(self.count,self.recfps)}({self.recfps:.2f}fps), output:{frames_to_min_sec(self.data.shape[0],self.outfps)}({self.outfps:.2f}fps)"  # FPS情報を文字列として整形
            fig1, ax1 = plt.subplots()   # 新しい図とaxesを作成
            for n in range(j):
                if label:
                    if n == 0 :
                        ax1.plot(x_range,self.sc_inPanMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_green_atr)
                    elif n == j-1 : 
                        ax1.plot(x_range,self.sc_inPanMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_red_atr)
                ax1.plot(x_range,self.sc_inPanMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha)
            if ax_title : ax1.set_title("Space")
            ax1.set_ylabel(label_X_atr)
            ax1.xaxis.set_major_formatter(formatter_x)
            ax1.set_xlabel(fps_info, fontsize='small')
            if label:ax1.legend(loc='upper right',fontsize='small')  # レジェンドを追加
            if space_axis_invert:
                ax1.invert_yaxis()
            fig1.set_size_inches(w_inc, h_inc/3)
            fig1.tight_layout()
            fig1.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread_Space.png', dpi=300, transparent=True)
            fig2, ax2 = plt.subplots()
            for n in range(j):
                if label:
                    if n == 0 :
                        ax2.plot(x_range,self.sc_resetPositionMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_green_atr)
                    elif n == j-1 : 
                        ax2.plot(x_range,self.sc_resetPositionMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_red_atr)
                ax2.plot(x_range,self.sc_resetPositionMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                # ax2.plot(self.sc_resetPositionMap[n,s_frame:e_frame], color="red",linewidth=plinewidth)
            if ax_title : ax2.set_title('Time')
            ax2.set_ylabel('Time(h:m:s)')
            ax2.yaxis.set_major_formatter(formatter_y) 
            ax2.xaxis.set_major_formatter(formatter_x)
            ax2.set_xlabel(fps_info, fontsize='small')
            if label:ax2.legend(loc='upper right',fontsize='small')  # レジェンドを追加
            # Y軸の最大値を設定
            if timeflow_scaling != True : ax2.set_ylim(top=self.count)
            fig2.set_size_inches(w_inc, h_inc/3)
            fig2.tight_layout()
            fig2.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread_Time.png', dpi=300, transparent=True)
            if axnum > 2: 
                fig3, ax3 = plt.subplots()
                for n in range(j):
                    if label:
                        if n == 0 :
                            ax3.plot(x_range,self.sc_rateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_green_atr)
                        elif n == j-1 : 
                            ax3.plot(x_range,self.sc_rateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_red_atr)
                    ax3.plot(self.sc_rateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                if ax_title : ax3.set_title('Rate(Time)')
                ax3.set_ylabel(rate_info,fontsize='small')
                ax3.xaxis.set_major_formatter(formatter_x)
                ax3.set_xlabel(fps_info,fontsize='small')
                if label:ax3.legend(loc='upper right',fontsize='small')  # レジェンドを追加
                if normal_line_draw :
                    ax3.axhline(y=1.0, color='black', linestyle='-',linewidth=0.3)
                    # y軸に1.0を強調表示
                    current_ticks = ax3.get_yticks()
                    new_ticks = np.append(current_ticks, 1.0)
                    ax3.set_yticks(new_ticks)
                fig3.set_size_inches(w_inc, h_inc/3)
                fig3.tight_layout()
                fig3.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread_PlayRate.png', dpi=300, transparent=True)
            if axnum > 3:
                fig4, ax4 = plt.subplots()
                for n in range(j):
                    if label:
                        if n == 0 :
                            ax4.plot(x_range,self.sc_SDRateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_green_atr)
                        elif n == j-1 : 
                            ax4.plot(x_range,self.sc_SDRateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_red_atr)
                    ax4.plot(self.sc_SDRateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                if ax_title : ax4.set_title('Rate(Space)')
                ax4.set_ylabel(rate_info,fontsize='small')
                ax4.xaxis.set_major_formatter(formatter_x)
                ax4.set_xlabel(fps_info,fontsize='small')
                if label:ax4.legend(loc='upper right',fontsize='small')  # レジェンドを追加
                fig4.set_size_inches(w_inc, h_inc/3)
                fig4.tight_layout()
                #メモリの単位をframeからmin,secへ
                fig4.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread_SDRate.png', dpi=300, transparent=True)
            if axnum > 4:    
                fig5, ax5 = plt.subplots()
                for n in range(j):
                    if label:
                        if n == 0 :
                            ax5.plot(x_range,self.sc_movementRateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_green_atr)
                        elif n == j-1 : 
                            ax5.plot(x_range,self.sc_movementRateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha,label=label_red_atr)
                    ax5.plot(self.sc_movementRateMap[n,s_frame:e_frame], color=color_map(n),linewidth=plinewidth,alpha=palpha)
                if ax_title : ax5.set_title('Rate(space & time)')
                ax5.set_ylabel(rate_info,fontsize='small')
                ax5.xaxis.set_major_formatter(formatter_x)
                ax5.set_xlabel(fps_info,fontsize='small')
                if label:ax5.legend(loc='upper right',fontsize='small')  # レジェンドを追加
                if label:ax5.legend()
                fig5.set_size_inches(w_inc, h_inc/3)
                fig5.tight_layout()
                fig5.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_'+str(j)+'thread_movementRate.png', dpi=300, transparent=True)

        else:
            # axnum = 5 if 4873
            # G else 3
            fig = plt.figure()
            ax1 = fig.add_subplot(axnum, 1, 1)
            for n in range(j):
                if label:
                    if n == 0:
                        ax1.plot(x_range, self.sc_inPanMap[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha, label=label_green_atr)
                    elif n == j - 1:
                        ax1.plot(x_range, self.sc_inPanMap[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha, label=label_red_atr)
                ax1.plot(x_range, self.sc_inPanMap[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha)
            if ax_title : ax1.set_ylabel(label_space_atr)
            if label: ax1.legend(loc='upper right')
            ax2 = fig.add_subplot(axnum, 1, 2)
            for n in range(j):
                ax2.plot(x_range, self.sc_resetPositionMap[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha)
            if not timeflow_scaling:
                ax2.set_ylim(top=self.count)
            if label: plt.text(0.95, 0.95, fps_info, transform=ax2.transAxes, horizontalalignment='right', verticalalignment='top', color=text_color)
            if len(self.depth_to_sel_recfps[s_frame:e_frame]) > 0:
                ax2_twin = ax2.twinx()
                ax2_twin.plot(x_range, self.depth_to_sel_recfps[s_frame:e_frame], color='blue', linestyle='-', linewidth=0.5)
                change_indices = np.where(np.diff(self.depth_to_sel_recfps[s_frame:e_frame]) != 0)[0] + 1
                for x_position in change_indices:
                    ax2.axvline(x=x_position + s_frame, color='blue', linestyle='--', linewidth=0.5)
                    time_text = frames_to_min_sec(x_position + s_frame, self.outfps)
                    ax2.text(x_position + s_frame, ax2.get_ylim()[1], f"{x_position + s_frame} ({time_text})", rotation=90, verticalalignment='top', color='blue')
                ax2_twin.set_ylabel('Rendering FPS')
                ax2_twin.spines['right'].set_position(('outward', 60))
                ax2_twin.yaxis.label.set_color('blue')
                ax2_twin.tick_params(axis='y', colors='blue')
                ax2_twin.spines['right'].set_position(('axes', 1.05))
                ax2_twin.spines['right'].set_visible(True)
                ax2_twin.yaxis.set_label_position('right')
                ax2_twin.yaxis.set_ticks_position('right')
            if ax_title : ax2.set_ylabel('Time(h:m:s)')
            ax1.tick_params(labelbottom=False)  # x軸の目盛りラベルを非表示
            ax2.yaxis.set_major_formatter(formatter_y) 
            ax2.xaxis.set_major_formatter(formatter_x)
            if ax1ylim != None:
                ax1.set_ylim(ax1ylim[0], ax1ylim[1])
            if ax2ylim != None:
                ax2.set_ylim(ax2ylim[0], ax2ylim[1])
            # 空間(ax1)の y 軸を反転: 0 を上、max を下にする
            if space_axis_invert:
                ax1.invert_yaxis()

            if axnum > 2:
                ax3 = fig.add_subplot(axnum, 1, 3)
                for n in range(j):
                    ax3.plot(x_range, self.sc_rateMap[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha)
                if ax_title : ax3.set_ylabel('Rate(Time)')
                if rate_lange_lim != None: ax3.set_ylim(-rate_lange_lim, rate_lange_lim)  # Set constant y-axis range 
                if normal_line_draw:
                    ax3.axhline(y=1.0, color='black', linestyle='-', linewidth=palpha)
                    current_ticks = ax3.get_yticks()
                    new_ticks = np.append(current_ticks, 1.0)
                    ax3.set_yticks(new_ticks)
                for x_position in x_positions:
                    ax2.axvline(x=x_position, color='black', linestyle='--', linewidth=0.5)
                    time_text = frames_to_min_sec(x_position, self.outfps)
                    ax2.text(x_position, ax2.get_ylim()[1], f"{x_position} ({time_text})", rotation=90, verticalalignment='top', color=text_color)
                for y_position in y_positions:
                    ax2.axhline(y=y_position, color='black', linestyle='--', linewidth=0.5)

                ax2.tick_params(labelbottom=False)  # x軸の目盛りラベルを非表示
                ax3.xaxis.set_major_formatter(formatter_x)
                if ax3ylim != None:
                    ax3.set_ylim(ax3ylim[0], ax3ylim[1]) 

                
                if custome_label:
                    # ax1.grid(False)  # グリッドを非表示
                    # ax1.set_xticks([])  # x軸の目盛りを非表示
                    # ax1.set_yticks([])  # y軸の目盛りを非表示
                    # ax2.grid(False)  # グリッドを非表示
                    # ax2.set_xticks([])  # x軸の目盛りを非表示
                    # ax2.set_yticks([])  # y軸の目盛りを非表示
                    # ax3.grid(False)  # グリッドを非表示
                    # ax3.set_xticks([])  # x軸の目盛りを非表示
                    # ax3.set_yticks([])  # y軸の目盛りを非表示
                    # remove_edge_labels(ax1)
                    # set_fixed_locator(ax1, 'x', 5)  # x軸の目盛りを5本に固定
                    # set_fixed_locator(ax1, 'y', 5)  # y軸の目盛りを5本に固定
                    # set_fixed_locator(ax2, 'x', 5)  # x軸の目盛りを5本に固定
                    # set_fixed_locator(ax2, 'y', 7)  # y軸の目盛りを5本に固定
                    # set_fixed_locator(ax3, 'x', 5)  # x軸の目盛りを5本に固定
                    # set_fixed_locator(ax3, 'y', 5)  # y軸の目盛りを5本に固定
                    # remove_edge_labels(ax1)
                    # set_variable_locator(ax2, 'x', 10)  # x軸の目盛りを可変数に設定
                    set_variable_locator(ax2, 'y', 6)  # y軸の目盛りを可変数に設定
                    set_variable_locator(ax3, 'y', 6)  # y軸の目盛りを可変数に設定
                    ax1.tick_params(labelbottom=False)  # x軸の目盛りラベルを非表示
                    ax2.tick_params(labelbottom=False)  # x軸の目盛りラベルを非表示
                    # remove_edge_labels(ax1)
                    remove_edge_labels(ax2)
                    remove_edge_labels(ax3)
                    

            if axnum > 3:
                ax4 = fig.add_subplot(axnum, 1, 4)
                for n in range(j):
                    ax4.plot(x_range, self.sc_SDRateMap[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha)
                if ax_title : ax4.set_ylabel('Rate(Space)')
                if rate_lange_lim != None: ax4.set_ylim(0, rate_lange_lim)  # Set constant y-axis range 
                ax3.tick_params(labelbottom=False)  # x軸の目盛りラベルを非表示
                ax4.xaxis.set_major_formatter(formatter_x)
                if ax4ylim != None:
                    ax4.set_ylim(ax4ylim[0], ax4ylim[1]) 

            if axnum > 4:
                ax5 = fig.add_subplot(axnum, 1, 5)
                for n in range(j):
                    ax5.plot(x_range, self.sc_movementRateMap[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha)
                if ax_title : ax5.set_ylabel('Rate(Space-Time)')
                # h_inc *= 1.5
                if rate_lange_lim != None: ax5.set_ylim(0, rate_lange_lim)  # Set constant y-axis range 
                # ax4.xaxis.set_major_formatter(formatter_x)
                ax4.tick_params(labelbottom=False)  # x軸の目盛りラベルを非表示
                ax5.xaxis.set_major_formatter(formatter_x)
                if ax5ylim != None:
                    ax5.set_ylim(ax5ylim[0], ax5ylim[1]) 

            if axnum > 5:
                ax6 = fig.add_subplot(axnum, 1, 6)
                for n in range(j):
                    ax6.plot(x_range, self.sc_parallel_component_Map[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha)
                if ax_title : ax6.set_ylabel('Vector(parallel)')
                ax5.tick_params(labelbottom=False)  # x軸の目盛りラベルを非表示
                ax6.xaxis.set_major_formatter(formatter_x)
                if ax6ylim != None:
                    ax6.set_ylim(ax6ylim[0], ax6ylim[1]) 

            if axnum > 6:
                ax7 = fig.add_subplot(axnum, 1, 7)
                for n in range(j):
                    ax7.plot(x_range, self.sc_perpendicular_component_Map[n, s_frame:e_frame], color=color_map(n), linewidth=plinewidth, alpha=palpha)
                if ax_title : ax7.set_ylabel('Vector(perpendicular)')
                ax6.tick_params(labelbottom=False)  # x軸の目盛りラベルを非表示
                ax7.xaxis.set_major_formatter(formatter_x)
                if ax7ylim != None:
                    ax7.set_ylim(ax7ylim[0], ax7ylim[1]) 

            fig.set_size_inches(w_inc, h_inc)
            plt.tight_layout()
            if x_positions or y_positions:
                plt.savefig(self.ORG_NAME + "_" + self.out_name_attr + '_' + str(j) + 'thread-renderSep'+ attr + "-"+str(w_inc)+"x"+str(h_inc)+"("+str(axnum)+').png', dpi=300, transparent=True)
            else:
                if s_frame != 0 or e_frame != full_range:
                    plt.savefig(self.ORG_NAME + "_" + self.out_name_attr + '_' + str(j) + 'th-part' + str(s_frame) + '-' + str(e_frame) + attr+"-"+str(w_inc)+"x"+str(h_inc) +"("+str(axnum)+').png', dpi=300, transparent=True)
                else:
                    plt.savefig(self.ORG_NAME + "_" + self.out_name_attr + '_' + str(j) + 'thread'+ attr+"-"+str(w_inc)+"x"+str(h_inc) +"("+str(axnum)+').png', dpi=300, transparent=True)

            # --- シークバー付き動画出力 (video_out=True のときのみ) ---
            if video_out:
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                # video_height_px を指定されたら fig.dpi を target_h / h_inc に設定し、
                # canvas レンダ時のピクセル解像度を目標に合わせる。
                # (PNG savefig は dpi=300 を明示指定しているので影響なし)
                if video_height_px is not None and h_inc > 0:
                    target_dpi = float(video_height_px) / float(h_inc)
                    fig.set_dpi(target_dpi)
                    print(f"  [video_out] target_height={video_height_px}px "
                          f"→ fig.dpi={target_dpi:.2f} (width≈{int(w_inc*target_dpi)}px)")

                # 各サブプロットの axes をリスト化（ギャップ描画スキップ用）
                _all_ax = [ax1, ax2]
                if axnum > 2: _all_ax.append(ax3)
                if axnum > 3: _all_ax.append(ax4)
                if axnum > 4: _all_ax.append(ax5)
                if axnum > 5: _all_ax.append(ax6)
                if axnum > 6: _all_ax.append(ax7)
                _bax = _all_ax[-1]  # x座標変換の基準

                # シークバー / タイムコード色（colormode 連動）
                if colormode == 'white':
                    sb_val = 255; fig_bg_c = 'black'
                else:
                    sb_val = 0;   fig_bg_c = 'white'

                # --- figure レンダリング ---
                canvas = FigureCanvasAgg(fig)
                if video_alpha:
                    fig.patch.set_alpha(0.0)
                    for _a in fig.get_axes():
                        _a.set_facecolor((1, 1, 1, 0))
                    canvas.draw()
                    img_bg_rgba = np.asarray(canvas.buffer_rgba()).copy()
                    fig_w_px, fig_h_px = canvas.get_width_height()
                    img_bg = cv2.cvtColor(img_bg_rgba, cv2.COLOR_RGBA2BGRA)
                else:
                    fig.patch.set_facecolor(fig_bg_c)
                    for _a in fig.get_axes():
                        _a.set_facecolor(fig_bg_c)
                    canvas.draw()
                    img_plot = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
                    fig_w_px, fig_h_px = canvas.get_width_height()
                    img_bg = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)

                # 各サブプロット枠の y 範囲（ピクセル, 画像座標系: 上=0）
                ax_y_ranges = []
                for _a in _all_ax:
                    pos = _a.get_position()
                    yt = fig_h_px - int(pos.y1 * fig_h_px)
                    yb = fig_h_px - int(pos.y0 * fig_h_px)
                    ax_y_ranges.append((yt, yb))

                # シークバー x 座標を事前計算
                y_ref = _bax.get_ylim()[0]
                px_positions = []
                for f in range(s_frame, e_frame):
                    xd = _bax.transData.transform((f, y_ref))[0]
                    px_positions.append(int(np.clip(xd, 0, fig_w_px - 1)))

                # ドット線パターン: dot_on=線幅（正方形ドット）, dot_off=間隔
                sb_w = 2  # シークバー幅 2px
                dot_on  = sb_w
                dot_off = sb_w * 2

                # タイムコード設定
                tc_font = cv2.FONT_HERSHEY_SIMPLEX
                tc_scale = max(0.4, fig_h_px / 1500.0)
                tc_thickness = max(1, int(tc_scale + 0.5))
                tc_margin = int(10 * tc_scale)

                # シークバー描画ヘルパー（各サブプロット枠内のみ、枠間ギャップはスキップ）
                def _draw_seekbar(frame, px, color):
                    x0 = max(0, px - sb_w // 2)
                    x1 = min(fig_w_px, px - sb_w // 2 + sb_w)
                    for (yt, yb) in ax_y_ranges:
                        y = yt
                        while y < yb:
                            ye = min(y + dot_on, yb)
                            frame[y:ye, x0:x1, :] = color
                            y += dot_on + dot_off

                # --- 書き出し ---
                attr_s = attr if attr else ""
                if video_alpha:
                    # ProRes 4444 (RGBA) via PyAV
                    sv_name = (self.ORG_NAME + "_" + self.out_name_attr + '_'
                               + str(j) + 'thread_seekbar' + attr_s + ".mov")
                    container = av.open(sv_name, mode='w')
                    stream = container.add_stream('prores_ks', rate=int(self.outfps))
                    stream.width = fig_w_px
                    stream.height = fig_h_px
                    stream.pix_fmt = 'yuva444p10le'
                    stream.options = {'profile': '4444'}

                    sb_bgra = (sb_val, sb_val, sb_val, 255)
                    tc_bgra = (sb_val, sb_val, sb_val, 255)

                    for i in range(e_frame - s_frame):
                        frame = img_bg.copy()  # BGRA
                        _draw_seekbar(frame, px_positions[i], sb_bgra)
                        # タイムコード (video_tc_overlay=False でスキップ)
                        if video_tc_overlay:
                            current_frame = s_frame + i
                            tc_text = frames_to_min_sec(current_frame + add_x_frames, self.outfps)
                            (tw, th), _ = cv2.getTextSize(tc_text, tc_font, tc_scale, tc_thickness)
                            tx = fig_w_px - tw - tc_margin
                            ty = fig_h_px - tc_margin
                            cv2.putText(frame, tc_text, (tx, ty), tc_font, tc_scale,
                                        tc_bgra, tc_thickness, cv2.LINE_AA)
                        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                        av_frame = av.VideoFrame.from_ndarray(frame_rgba, format='rgba')
                        for packet in stream.encode(av_frame):
                            container.mux(packet)

                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()

                else:
                    # MP4 (BGR, 不透明) via cv2
                    sv_name = (self.ORG_NAME + "_" + self.out_name_attr + '_'
                               + str(j) + 'thread_seekbar' + attr_s + ".mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vout = cv2.VideoWriter(sv_name, fourcc, float(self.outfps), (fig_w_px, fig_h_px))

                    sb_bgr = (sb_val, sb_val, sb_val)
                    tc_bgr = (sb_val, sb_val, sb_val)

                    for i in range(e_frame - s_frame):
                        frame = img_bg.copy()  # BGR
                        _draw_seekbar(frame, px_positions[i], sb_bgr)
                        # タイムコード (video_tc_overlay=False でスキップ)
                        if video_tc_overlay:
                            current_frame = s_frame + i
                            tc_text = frames_to_min_sec(current_frame + add_x_frames, self.outfps)
                            (tw, th), _ = cv2.getTextSize(tc_text, tc_font, tc_scale, tc_thickness)
                            tx = fig_w_px - tw - tc_margin
                            ty = fig_h_px - tc_margin
                            cv2.putText(frame, tc_text, (tx, ty), tc_font, tc_scale,
                                        tc_bgr, tc_thickness, cv2.LINE_AA)
                        vout.write(frame)

                    vout.release()

                print(f"Seekbar video saved: {sv_name}")
                self.maneuver_log("seekbar_video")

        if debugmode:
            plt.show()
        plt.clf()
        plt.close('all')

    def maneuver_3dplot(self, thread_num=None, thread_through=False, zRangeFix=False, out_framenums=50, out_fps=25, colormode='white',line_width=1,aspect_ratio=(1, 1, 1),elev=25, azim=-40, dpi=200,xticks=False,zticks=False,yticks_normal=True,only_seq_img=False,only_seq_img_num=None,lineplot=True,vectorplot=False,gridplot=True,vector_def_frame=120,velocity=10,vector_color_amp=1.0,s_frame=0,zRangeMin=None,zRangeMax=None):
        print(sys._getframe().f_code.co_name)
        if thread_through == False:
                if thread_num != None:
                    self.info_setting(thread_num)
                else:
                    if self.infolog == 0:
                        self.info_setting()
                    elif self.infolog != self.log:
                        self.info_setting(self.sc_resetPositionMap.shape[0])

        formatter_y = ticker.FuncFormatter(lambda y, pos: frames_to_min_sec(y, self.recfps))
        j = self.sc_resetPositionMap.shape[0]
        cmap = LinearSegmentedColormap.from_list('original', [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)], N=j)
        color_map = plt.get_cmap(cmap)


        def remove_edge_labels(ax):
            # print("remove")
            """Remove edge labels from the axes."""
            yticks = ax.get_yticks()
            print(yticks)
            # # y軸の目盛りラベルを取得してエッジ部分を削除
            labels = [item.get_text() for item in ax.get_yticklabels()]
            if labels:
                # labels[0] = ''
                # labels[1] = ''
                labels[-1] = ''
                labels[-2] = ''
                ax.yaxis.set_major_locator(FixedLocator(yticks))
                ax.yaxis.set_major_formatter(FixedFormatter(labels))

        # カラーマップと正規化
        # カラーマップと色の設定
        def get_color(v, vmin=-velocity, vmax=velocity):
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            norm_v = norm(v)
            # 黄色から青への線形補間
            yellow = np.array([1, 1, 0,1])
            blue = np.array([0, 0, 1,1])
            color = (1 - norm_v) * blue + norm_v * yellow
            return color

        allzMin = np.amin(self.data[:, :, 1])
        allzMax = np.amax(self.data[:, :, 1])
        # allzRange = allzMax - allzMin
        # X軸(空間方向): 実データの範囲を使う（scan_numsが出力幅と乖離する場合に対応）
        xMin = np.amin(self.data[:, :, 0])
        xMax = np.amax(self.data[:, :, 0])
        # 余白を少し追加
        x_margin = (xMax - xMin) * 0.02
        xMin = max(0, xMin - x_margin)
        xMax = xMax + x_margin
        yMax = self.slit_length if self.scan_direction % 2 == 1 else self.scan_nums
        drawLineNum = j
        if self.data.shape[0] < out_framenums:
            out_framenums = self.data.shape[0]
        nsteps = math.floor(self.data.shape[0] / out_framenums)
        if os.path.isdir('3dPlot_seq') == False:
            os.makedirs('3dPlot_seq')  # ディレクトリ作成

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.view_init(elev=25, azim=-40)  # 3D視点設定

        if colormode == 'black':
            bg_color = (0, 0, 0)
            edge_color = (1, 1, 1,0.11)
            # grid_color = (1, 1, 1,0.3)
            grid_color = (0.3,0.3,0.3, 1)
            # label_color = (1, 1, 1,0.6)
            tick_color = (1, 1, 1,1)
            pane_color = (0,0,0,0.0)  # なし
            linewidth = line_width  # 線の太さを指定
            vector_linewidth=0.6
            grid_linewidth = 0.5  # グリッド線の太さを指定
            edge_linewidth = 0.5
        elif colormode == 'gray':
            bg_color = (0, 0, 0)
            edge_color = (1, 1, 1,0.11)
            # grid_color = (1, 1, 1,0.3)
            grid_color = (0.3,0.3,0.3, 1)
            # label_color = (1, 1, 1,0.6)
            tick_color = (1, 1, 1,1)
            pane_color = (0.1,0.1,0.1)  # なし
            linewidth = line_width # 線の太さを指定
            vector_linewidth=0.6
            grid_linewidth = 0.5  # グリッド線の太さを指定
            edge_linewidth = 0.5
        else: 
            #white
            bg_color = (1, 1, 1,0)
            edge_color = (1,1,1,0)
            grid_color = "#dddddd"
            label_color = (0, 0, 0)
            tick_color = (0, 0, 0)
            pane_color = (1, 1, 1, 0.6)  # 半透明の白
            linewidth = line_width  # 線の太さを指定
            grid_linewidth = 0.8  # グリッド線の太さを指定
            edge_linewidth = 0.5
            
            plt.style.use("bmh")
            mpl.rcParams['axes.labelsize']= 12
            ax.xaxis.line.set_color('#888888')
            ax.yaxis.line.set_color('#888888')
            ax.zaxis.line.set_color('#888888')


        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
        ax.xaxis.line.set_color(pane_color)
        ax.yaxis.line.set_color(pane_color)
        ax.zaxis.line.set_color(pane_color)


        ax.xaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)
        ax.yaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)
        ax.zaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)

        ax.view_init(elev=elev, azim=azim)

        for i in range(s_frame, self.data.shape[0], nsteps):
            zMin = np.amin(self.data[i, :, 1])
            zMax = np.amax(self.data[i, :, 1])
            
            if zRangeFix:
                ylimmin=allzMin if zRangeMin == None else zRangeMin
                ylimmax=allzMax if zRangeMax == None else zRangeMax
                ax.set_ylim(ylimmin, ylimmax) # z軸固定
            else:
                # y軸の表示範囲を設定
                y_range_min = min(zMin, zMax - 5760)
                y_range_max = max(zMax, zMin + 5760)
                y_range=y_range_max-y_range_min
                ax.set_ylim(y_range_min, y_range_max)
                ax.set_box_aspect((aspect_ratio[0],np.clip(int(aspect_ratio[1]*(y_range / 11520)),aspect_ratio[1],aspect_ratio[0]*10),aspect_ratio[2]))

            ax.set_xlim(xMin, xMax)  # x軸固定
            ax.set_zlim(0, yMax) 
            
            if xticks == False : ax.xaxis.set_ticks([])
            else :
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if zticks == False : ax.zaxis.set_ticks([])
            else : 
                # ax.set_zticks(np.arange(0, yMax + 1))
                ax.set_zticks(np.arange(0, yMax + 1, math.floor(yMax / 2)))


            # y軸の目盛り数を7に制限
            # ax.set_yticks(np.linspace(y_range_min, y_range_max, 7))
            # y軸の目盛りを可変に設定して、7本に制限
            if yticks_normal :
                # ax.set_yticks(np.arange(zMin,zMax+1),math.floor(zMax / 2))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_formatter(formatter_y)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
                ax.yaxis.set_major_formatter(formatter_y)
                remove_edge_labels(ax)
            # ax.xaxis.set_ticks([])
            # ax.zaxis.set_ticks([])
            # ax.set_xticks(np.arange(0,xMax+1,math.floor(xMax/2)))
            # ax.set_zticks(np.arange(0, yMax + 1, math.floor(yMax / 2)))

            # ax.xaxis.line.set_color(tick_color)
            # ax.yaxis.line.set_color(tick_color)
            # ax.zaxis.line.set_color(tick_color)
            if only_seq_img == False :
                if lineplot :
                    for k in range(drawLineNum):
                        rp = k * math.ceil(self.data.shape[1] / (drawLineNum - 1)) if k != (drawLineNum - 1) else (self.data.shape[1] - 1)
                        if self.scan_direction % 2 == 1:
                            ax.plot([self.data[i, rp, 0], self.data[i, rp, 0]], [self.data[i, rp, 1], self.data[i, rp, 1]], [self.slit_length, 0], color=color_map(k),linewidth=linewidth)
                        else:
                            ax.plot([self.slit_length, 0], [self.data[i, rp, 1], self.data[i, rp, 1]], [self.data[i, rp, 0], self.data[i, rp, 0]], color=color_map(k),linewidth=linewidth)
                if vectorplot :
                    for k in range(drawLineNum):
                        # rp = k * math.ceil(self.data.shape[1] / (drawLineNum - 1)) if k != (drawLineNum - 1) else (self.data.shape[1] - 1)
                        # x = self.data[i-10, rp, 0]
                        # y = self.data[i-10, rp, 1]
                        # z = 1000
                        # u = self.data[i, rp, 0]
                        # v = self.data[i, rp, 1]
                        # w = 1000
                        # ax.quiver(x, y, z, u, v, w, color=color_map(k), linewidth=linewidth)
                        rp = k * math.ceil(self.data.shape[1] / (drawLineNum - 1)) if k != (drawLineNum - 1) else (self.data.shape[1] - 1)
                        pastframe=i-vector_def_frame
                        if pastframe < 0 : pastframe=0
                        x = self.data[pastframe, rp, 0]
                        # y = self.data[pastframe, rp, 1]
                        # z = 0
                        # u = self.data[i, rp, 0] - x
                        # u = 0
                        # v = self.data[i, rp, 1] - y
                        # w = 0
                        
                        # x = self.data[i, rp, 0]
                        y = self.data[i, rp, 1] - self.sc_rateMap[k,i]*velocity
                        z = 0
                        u = self.data[i, rp, 0] - x
                        # u = 0
                        # v = self.data[i, rp, 1] - y
                        v= self.sc_rateMap[k,i] * velocity
                        w = 0
                        vector_length = np.sqrt(u**2 + v**2 + w**2)
    
                        # 矢印の形の比率をベクトルの長さに基づいて調整
                        # arrow_length = base_arrow_length * vector_length
                        # head_length = base_head_length * vector_length
                        # head_axis_length = base_head_axis_length * vector_length
                        # head_width = base_head_width * vector_length
                        # print(v)
                        # https://www.anarchive-beta.com/entry/2023/08/01/235128
                        for l in range(0,self.slit_length,(self.slit_length-1)//3):
                            # ax.quiver(x, y, z+l, u, v, w, color=color_map(k), linewidth=linewidth)
                            # ax.quiver(x, y, z+l, u, v, w, color=get_color(v), linewidth=linewidth,arrow_length_ratio=0.5,length=arrow_length,headlength=head_length,headaxislength=head_axis_length,headwidth=head_width)
                            ax.quiver(x, y, z+l, u, v, w, color=get_color(v * vector_color_amp), linewidth=vector_linewidth,arrow_length_ratio=0.6)


            ax.xaxis.line.set_color(pane_color)
            ax.yaxis.line.set_color(pane_color)
            ax.zaxis.line.set_color(pane_color)
            ax.zaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)
            ax.xaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)
            ax.yaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)

            if only_seq_img :
                y_ticks = ax.get_yticks()  # Y軸の目盛り位置を取得
                y_ticks = y_ticks[1:-1]
                if only_seq_img_num  != None:
                    if only_seq_img_num < len(y_ticks):
                        y_tick = y_ticks[only_seq_img_num]
                        y_min_image = cv2.imread('/Volumes/RF-4T-2023/midtide-document-EditFile/NAMUYA-Seq-mini/NAMUYA-mini_'+str(int(y_tick)).zfill(5)+'.png', cv2.IMREAD_UNCHANGED)
                        if y_min_image is not None:
                            y_min_image = cv2.flip(y_min_image, 0)
                            # BGRからRGBに変換
                            y_min_image = cv2.cvtColor(y_min_image, cv2.COLOR_BGR2RGB)
                            y_min_texture = np.ones((y_min_image.shape[0], y_min_image.shape[1], 4))  # RGBAの空の配列を作成
                            y_min_texture[..., :3] = y_min_image[..., :3] / 255.0  # RGB値を正規化して設定
                            y_min_texture[..., 3] = 1 # 透過度を設定
                            x = np.linspace(xMin, xMax, y_min_texture.shape[1])
                            z = np.linspace(0, self.slit_length, y_min_texture.shape[0])
                            x, z = np.meshgrid(x, z)
                            ax.plot_surface(x, y_tick, z, rstride=1, cstride=1, facecolors=y_min_texture,shade=False,antialiased=False)
                else :
                    for y_tick in y_ticks:
                        y_min_image = cv2.imread('/Volumes/RF-4T-2023/midtide-document-EditFile/NAMUYA-Seq-mini/NAMUYA-mini_'+str(int(y_tick)).zfill(5)+'.png', cv2.IMREAD_UNCHANGED)
                        if y_min_image is not None:
                            y_min_image = cv2.flip(y_min_image, 0)
                            # BGRからRGBに変換
                            y_min_image = cv2.cvtColor(y_min_image, cv2.COLOR_BGR2RGB)
                            y_min_texture = np.ones((y_min_image.shape[0], y_min_image.shape[1], 4))  # RGBAの空の配列を作成
                            y_min_texture[..., :3] = y_min_image[..., :3] / 255.0  # RGB値を正規化して設定
                            y_min_texture[..., 3] = 1 # 透過度を設定
                            x = np.linspace(xMin, xMax, y_min_texture.shape[1])
                            z = np.linspace(0, self.slit_length, y_min_texture.shape[0])
                            x, z = np.meshgrid(x, z)
                            ax.plot_surface(x, y_tick, z, rstride=1, cstride=1, facecolors=y_min_texture,shade=False,antialiased=False)
            
            if gridplot ==False:
                ax.yaxis.set_ticks([])
                    

            # # 手前の面 (y=min) に画像を貼り付け
            # y_min_image = cv2.imread('/Volumes/RF-4T-2023/midtide-document-EditFile/NAMUYA-Seq-mini/NAMUYA-mini_'+str(int(y_range_min)).zfill(5)+'.png', cv2.IMREAD_UNCHANGED)
            # y_min_texture = cv2.flip(y_min_image, 0)
            # x = np.linspace(xMin, xMax, y_min_texture.shape[1])
            # z = np.linspace(0, self.slit_length, y_min_texture.shape[0])
            # x, z = np.meshgrid(x, z)
            # ax.plot_surface(x, y_range_min, z, rstride=1, cstride=1, facecolors=y_min_texture / 255.0, alpha=0.3, antialiased=False)

            # # 奥の面 (y=max) に画像を貼り付け
            # y_max_image = cv2.imread('/Volumes/RF-4T-2023/midtide-document-EditFile/NAMUYA-Seq-mini/NAMUYA-mini_'+str(int(y_range_max)).zfill(5)+'.png', cv2.IMREAD_UNCHANGED)
            # y_max_texture = cv2.flip(y_max_image, 0)
            # x = np.linspace(xMin, xMax, y_max_texture.shape[1])
            # z = np.linspace(0, self.slit_length, y_max_texture.shape[0])
            # x, z = np.meshgrid(x, z)
            # ax.plot_surface(x, y_range_max, z, rstride=1, cstride=1, facecolors=y_max_texture / 255.0, alpha=0.3, antialiased=False)

            # fig.set_size_inches(img_width, img_height)
            plt.savefig('3dPlot_seq/' + self.ORG_NAME + "_process" + str(self.log) + '_3dPlot_' + str(j) + '-' + str(i) + '.png', dpi=dpi, transparent=True, bbox_inches='tight')
            plt.cla()
        plt.close()

        # 最初に保存されたフレームのPNGでサイズを取得
        _first_saved_i = s_frame if s_frame < self.data.shape[0] else 0
        img = cv2.imread('3dPlot_seq/' + self.ORG_NAME + "_process" + str(self.log) + '_3dPlot_' + str(j) + '-' + str(_first_saved_i) + '.png')
        if img is None:
            print("Warning: 3dPlot PNG not found, skipping video output")
            return

        attr_name='_3dPlot'
        if vectorplot: attr_name+='+v'
        if lineplot: attr_name+='+l'
        if gridplot: attr_name+='+g'

        # 1フレームのみの場合はPNG静止画として出力
        if out_framenums <= 1 or self.data.shape[0] <= 1:
            png_name = self.ORG_NAME + "_" + self.out_name_attr + "_process" + str(self.log) + attr_name + '.png'
            if self.embedHistory_intoName:
                if len(png_name) > 230:
                    self.embedHistory_intoName = False
            if self.embedHistory_intoName == False:
                png_name = self.ORG_NAME + "_process" + str(self.log) + attr_name + '.png'
            # 透過PNGから背景色付きPNGに変換して保存
            readimg = cv2.imread('3dPlot_seq/' + self.ORG_NAME + "_process" + str(self.log) + '_3dPlot_' + str(j) + '-' + str(_first_saved_i) + '.png', cv2.IMREAD_UNCHANGED)
            if readimg is not None and readimg.shape[2] == 4:
                b, g, r, a = cv2.split(readimg)
                bg = np.zeros_like(a)
                bg[a == 0] = 255
                result = cv2.merge([b, g, r])
                if colormode == "black":
                    result[bg == 255] = [0, 0, 0]
                else:
                    result[bg == 255] = [255, 255, 255]
                cv2.imwrite(png_name, result)
            else:
                import shutil
                src = '3dPlot_seq/' + self.ORG_NAME + "_process" + str(self.log) + '_3dPlot_' + str(j) + '-' + str(_first_saved_i) + '.png'
                shutil.copy2(src, png_name)
            print(f"3dPlot PNG saved: {png_name}")
        else:
            # 複数フレーム: 動画として出力
            attr_name += '.mp4'
            vname = self.ORG_NAME + "_" + self.out_name_attr + "_process" + str(self.log) +  attr_name
            if self.embedHistory_intoName:
                max_length = 230
                if len(vname) > max_length:
                    self.embedHistory_intoName = False
            if self.embedHistory_intoName == False:
                vname = self.ORG_NAME + "_process" + str(self.log)  +  attr_name
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video = cv2.VideoWriter(vname, fourcc, out_fps, (int(img.shape[1]), int(img.shape[0])), 1)
            for i in range(0, self.data.shape[0], nsteps):
                readimg = cv2.imread('3dPlot_seq/' + self.ORG_NAME + "_process" + str(self.log) + '_3dPlot_' + str(j) + '-' + str(i) + '.png',cv2.IMREAD_UNCHANGED)
                if img.shape != readimg.shape:
                    readimg = cv2.resize(readimg, (img.shape[1], img.shape[0]))
                b, g, r, a = cv2.split(readimg)
                black_background = np.zeros_like(a)
                black_background[a == 0] = 255
                readimg = cv2.merge([b, g, r])
                if colormode == "black":
                    readimg[black_background == 255] = [0, 0, 0]
                else :
                    readimg[black_background == 255] = [255,255,255]
                video.write(readimg)
            video.release()  
             
        """ 
        if thread_through == False:
            if thread_num != None : 
                self.info_setting(thread_num)
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
        if self.data.shape[0]< out_framenums:
            out_framenums= self.data.shape[0]
        nsteps = math.floor(self.data.shape[0]/out_framenums)
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
        self._close_video_sink(_sink_kind, _sink_obj)
        """
    
    def maneuver_3dplot_midtide(self, thread_num=None, thread_through=False, zRangeFix=False, out_framenums=50, out_fps=25, colormode='white',aspect_ratio=(1, 1, 1), elev=25, azim=-40, dpi=200):
        print(sys._getframe().f_code.co_name)
        if thread_through == False:
                if thread_num != None:
                    self.info_setting(thread_num)
                else:
                    if self.infolog == 0:
                        self.info_setting()
                    elif self.infolog != self.log:
                        self.info_setting(self.sc_resetPositionMap.shape[0])

        formatter_y = ticker.FuncFormatter(lambda y, pos: frames_to_min_sec(y, self.recfps))
        j = self.sc_resetPositionMap.shape[0]
        cmap = LinearSegmentedColormap.from_list('original', [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)], N=j)
        color_map = plt.get_cmap(cmap)
        allzMin = np.amin(self.data[:, :, 1])
        allzMax = np.amax(self.data[:, :, 1])
        # allzRange = allzMax - allzMin
        # X軸(空間方向): 実データの範囲を使う（scan_numsが出力幅と乖離する場合に対応）
        xMin = np.amin(self.data[:, :, 0])
        xMax = np.amax(self.data[:, :, 0])
        # 余白を少し追加
        x_margin = (xMax - xMin) * 0.02
        xMin = max(0, xMin - x_margin)
        xMax = xMax + x_margin
        yMax = self.slit_length if self.scan_direction % 2 == 1 else self.scan_nums
        drawLineNum = j
        
        if self.data.shape[0] < out_framenums:
            out_framenums = self.data.shape[0]
        nsteps = math.floor(self.data.shape[0] / out_framenums)
        if os.path.isdir('3dPlot_seq') == False:
            os.makedirs('3dPlot_seq')  # ディレクトリ作成
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.view_init(elev=25, azim=-40)  # 3D視点設定
        if colormode == 'black':
            bg_color = (0, 0, 0)
            edge_color = (1, 1, 1,0.11)
            grid_color = (1, 1, 1,0.3)
            label_color = (1, 1, 1,0.6)
            tick_color = (1, 1, 1,0.6)
            pane_color = (0,0,0,0.0)  # なし
            linewidth = 0.6  # 線の太さを指定
            grid_linewidth = 0.5  # グリッド線の太さを指定
            edge_linewidth = 0.5
        else:
            bg_color = (1, 1, 1)
            edge_color = (0, 0, 0, 0.5)
            grid_color = "#dddddd"
            label_color = (0, 0, 0)
            tick_color = (0, 0, 0)
            pane_color = (1, 1, 1, 0.6)  # 半透明の白
            linewidth = 1  # 線の太さを指定
            grid_linewidth = 0.8  # グリッド線の太さを指定
            edge_linewidth = 0.5


        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.w_xaxis.set_pane_color(pane_color)
        ax.w_yaxis.set_pane_color(pane_color)
        ax.w_zaxis.set_pane_color(pane_color)
        ax.xaxis.line.set_color(pane_color)
        ax.yaxis.line.set_color(pane_color)
        ax.zaxis.line.set_color(pane_color)


        ax.xaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)
        ax.yaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)
        ax.zaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)

        ax.view_init(elev=elev, azim=azim)

        for i in range(0, self.data.shape[0], nsteps):
            zMin = np.amin(self.data[i, :, 1])
            zMax = np.amax(self.data[i, :, 1])
            
            if zRangeFix:
                ax.set_ylim(allzMin, allzMax)  # z軸固定
            else:
                # y軸の表示範囲を設定
                y_range_min = min(zMin, zMax - 5760)
                y_range_max = max(zMax, zMin + 5760)
                y_range=y_range_max-y_range_min
                ax.set_ylim(y_range_min, y_range_max)
                ax.set_box_aspect((aspect_ratio[0],np.clip(int(aspect_ratio[1]*(y_range / 11520)),aspect_ratio[1],aspect_ratio[0]*10),aspect_ratio[2]))

            ax.set_xlim(xMin, xMax)  # x軸固定
            ax.xaxis.set_ticks([])
            ax.zaxis.set_ticks([])

            # y軸の目盛りを可変に設定して、7本に制限
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
            ax.yaxis.set_major_formatter(formatter_y)

            # 画像データを3Dプロットに貼り付ける
            img_path ='/Volumes/RF-4T-2023/midtide-document-EditFile/midtide3_mix_seq/midtide3_mix_seq_00035.png'  # 画像ファイルのパスを指定
            img = plt.imread(img_path)

            for k in range(drawLineNum):
                print(k)
                rp = k * math.ceil(self.data.shape[1] / (drawLineNum - 1)) if k != (drawLineNum - 1) else (self.data.shape[1] - 1)
                # ax.plot([self.data[i, rp, 0], self.data[i, rp, 0]], [self.data[i, rp, 1], self.data[i, rp, 1]], [self.slit_length, 0], color=color_map(k),linewidth=linewidth)
                
                # 画像の短冊状の切り出し
                x_start = int(k * img.shape[1] / drawLineNum)
                x_end = int((k + 1) * img.shape[1] / drawLineNum)
                img_slice = img[:, x_start:x_end]

                # 短冊状の画像を貼り付けるための座標を設定
                if rp > 47: 
                    rp = 47
                print(rp)
                x_img = np.array([[self.data[i, rp, 0], self.data[i, rp, 0]],[self.data[i, rp+1, 0], self.data[i, rp+1, 0]]])
                y_img = np.array([[self.data[i, rp, 1], self.data[i, rp+1, 1]],[self.data[i, rp, 1], self.data[i, rp+1, 1]]])
                X_img, Y_img = np.meshgrid(x_img, y_img)
                Z_img  = np.array([[self.slit_length, self.slit_length], [0, 0]])


                ax.plot_surface(X_img, Y_img, Z_img, rstride=1, cstride=1, facecolors=img_slice, shade=False)

                # x_img = np.linspace(xMin, xMax, img.shape[1])
                # y_img = np.linspace(y_range_min, y_range_max, img.shape[0])
                # X_img, Y_img = np.meshgrid(x_img, y_img)
                # Z_img = np.zeros_like(X_img)  # Z座標を0に固定

                # ax.plot_surface(X_img, Y_img, Z_img, rstride=1, cstride=1, facecolors=plt.imread(img_path), shade=False)



            ax.xaxis.line.set_color(pane_color)
            ax.yaxis.line.set_color(pane_color)
            ax.zaxis.line.set_color(pane_color)
            ax.zaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)
            ax.xaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)
            ax.yaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)

                    
            # fig.set_size_inches(img_width, img_height)
            plt.savefig('3dPlot_seq/' + self.ORG_NAME + "_process" + str(self.log) + '_3dPlot_' + str(j) + '-' + str(i) + '.png', dpi=dpi, transparent=True, bbox_inches='tight')
            plt.cla()
        plt.close()

        img = cv2.imread('3dPlot_seq/' + self.ORG_NAME + "_process" + str(self.log) + '_3dPlot_' + str(j) + '-' + str(nsteps) + '.png')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # コーデック指定
        vname = self.ORG_NAME + "_" + self.out_name_attr + '_3dPlot.mp4'
        if self.embedHistory_intoName:
            max_length = 230
            if len(vname) > max_length:
                self.embedHistory_intoName = False
        if self.embedHistory_intoName == False:
            vname = self.ORG_NAME + "_process" + str(self.log) + '_3dPlot.mp4'
        video = cv2.VideoWriter(vname, fourcc, out_fps, (int(img.shape[1]), int(img.shape[0])), 1)
        for i in range(0, self.data.shape[0], nsteps):
            readimg = cv2.imread('3dPlot_seq/' + self.ORG_NAME + "_process" + str(self.log) + '_3dPlot_' + str(j) + '-' + str(i) + '.png',cv2.IMREAD_UNCHANGED)
            if img.shape != readimg.shape:
                readimg = cv2.resize(readimg, (img.shape[1], img.shape[0]))
            # 白い背景部分を黒に変換
            # アルファチャンネルを分離
            b, g, r, a = cv2.split(readimg)
            # アルファチャンネルがゼロ（透明）の部分を黒に設定
            black_background = np.zeros_like(a)
            black_background[a == 0] = 255

            # カラー画像にアルファチャンネルを適用
            readimg = cv2.merge([b, g, r])
            if colormode == "black":
                readimg[black_background == 255] = [0, 0, 0]
            else : 
                readimg[black_background == 255] = [255,255,255]

            video.write(readimg)
        video.release()    
             
        #added 2024,0428 
    def maneuver_imgplot(self, plot_mode="all", colormode='black', nticks_x=4, nticks_y=4, save_png=True, time_axis='auto'):
        """
        plot_mode : str
            "space", "time", "rate", or combinations ("all" = all three)
        time_axis : str
            'auto'  -> scan_direction==1 のとき Y軸を時間軸に設定
            'x'     -> 常にX軸を時間軸とする
            'y'     -> 常にY軸を時間軸とする
        """
        # -------- カラー設定 --------
        if colormode == 'white':
            text_color = (1, 1, 1)
            edge_color = (0.3, 0.3, 0.3, 1)
            grid_color = (0.3, 0.3, 0.3, 1)
        else:
            text_color = (0, 0, 0)
            edge_color = (0, 0, 0, 0.5)
            grid_color = "#aaaaaa"

        mpl.rcParams['axes.facecolor'] = (1, 1, 1, 0)
        mpl.rcParams['axes.edgecolor'] = edge_color
        mpl.rcParams['axes.labelcolor'] = text_color
        mpl.rcParams['xtick.color'] = text_color
        mpl.rcParams['ytick.color'] = text_color
        mpl.rcParams['grid.color'] = grid_color
        mpl.rcParams['text.color'] = text_color
        mpl.rcParams['figure.facecolor'] = (1, 1, 1, 0)

        # -------- データ --------
        spatial = self.data[:, :, 0].T
        temporal = self.data[:, :, 1].T
        frame_diff = np.diff(temporal, axis=1)
        if frame_diff.shape[1] == 0:
            # 1フレームのみの場合: レートは0
            frame_diff = np.zeros_like(temporal)
        else:
            frame_diff = np.pad(frame_diff, ((0, 0), (0, 1)), mode='edge')
        rate = (self.outfps / self.recfps) * frame_diff

        fx = ticker.FuncFormatter(lambda x, pos: frames_to_min_sec(int(round(x)), self.outfps))

        # -------- モード設定 --------
        modes = {
            "space": ["space"], "time": ["time"], "rate": ["rate"],
            "space+time": ["space", "time"], "time+rate": ["time", "rate"],
            "space+rate": ["space", "rate"], "all": ["space", "time", "rate"]
        }
        plot_mode = plot_mode.lower()
        targets = modes.get(plot_mode)
        if not targets:
            raise ValueError(f"Invalid plot_mode: {plot_mode}")

        # -------- 軸方向自動設定 --------
        if time_axis == 'auto':
            time_axis = 'y' if getattr(self, "scan_direction", 1) == 1 else 'x'

        # -------- 縦軸ラベル --------
        if getattr(self, "scan_direction", 1) == 1:
            top_label, bottom_label = "start", "end"
            left_label, right_label = "left", "right"
        else:
            top_label, bottom_label = "top", "bottom"
            left_label, right_label = "start", "end"

        # -------- プロット --------
        fig, ax = plt.subplots(1, len(targets), figsize=(8 * len(targets), 6))
        if len(targets) == 1:
            ax = [ax]

        for i, t in enumerate(targets):
            # === 各データ選択 ===
            if t == "space":
                data = spatial
                title = "Space Coordinate (x)" if self.scan_direction == 1 else "Space Coordinate (y)"
                cmap = 'gray'
                vmin, vmax = np.nanmin(data), np.nanmax(data)
            elif t == "time":
                data = temporal
                title = "Time Coordinate"
                cmap = 'gray'
                vmin, vmax = np.min(data), np.max(data)
            else:
                data = rate
                title = "Playback Rate"
                cmap = 'gray'
                max_dev = max(np.abs(data.max() - 1.0), np.abs(data.min() - 1.0))
                if max_dev < 1.0:
                    max_dev = 1.0
                vmin, vmax = 1.0 - max_dev, 1.0 + max_dev

            # === 軸範囲 ===
            y_max = data.shape[0] - 1
            x_max = data.shape[1] - 1

            # === scan_direction==1 → 上下反転 ===
            # if getattr(self, "scan_direction", 1) == 1:
            #     extent = [0, x_max, 0, y_max]
            # else:
            #     extent = [0, x_max, y_max, 0]
            data_to_plot = data

            # === 軸設定（time_axisに応じて） ===
            if time_axis == 'y':
                # 時間軸をY方向に
                data_to_plot = data_to_plot.T
                y_max = data_to_plot.shape[0] - 1
                x_max = data_to_plot.shape[1] - 1

                # Y軸反転（上が時間0、下が時間進行）
                extent = [0, x_max, y_max, 0]

                xlabel = "Width(px)" if self.scan_direction == 1 else "Height(px)"
                ylabel = "← Time ←"

                # tickの設定（上から下に時間が進むように）
                x_ticks = np.linspace(0, x_max, nticks_x)
                y_ticks = np.linspace(0, y_max, nticks_y)
                y_tick_labels = [frames_to_min_sec(int(round(v)), self.outfps) for v in y_ticks]

            else:
                # 時間軸をX方向に
                extent = [0, x_max, y_max, 0]
                xlabel = "→ Time →"
                ylabel = "Width(px)" if self.scan_direction == 1 else "Height(px)"
                x_ticks = np.linspace(0, x_max, nticks_x)
                y_ticks = np.linspace(0, y_max, nticks_y)
                y_tick_labels = [f"{int(v)}" for v in y_ticks]


            im = ax[i].imshow(
                data_to_plot,
                cmap=cmap,
                aspect='auto',
                extent=extent,
                vmin=vmin,
                vmax=vmax
            )

            # === カラーバー ===
            cbar = fig.colorbar(im, ax=ax[i])
            cbar.ax.tick_params(labelsize=10)
            if t == "space":
                cbar.set_label("Input Video Space Coordinate", fontsize=10)
                tick_values = np.linspace(vmin, vmax, 5)
                tick_labels = [f"{int(v)}" for v in tick_values]
                cbar.set_ticks(tick_values)
                cbar.set_ticklabels(tick_labels)

            elif t == "time":
                tick_values = np.linspace(vmin, vmax, 5)
                tick_labels = [frames_to_min_sec(v, self.recfps) for v in tick_values]
                cbar.set_ticks(tick_values)
                cbar.set_ticklabels(tick_labels)
                cbar.set_label("Input Video Time Coordinate", fontsize=10)

            elif t == "rate":
                cbar.set_label("Playback Rate", fontsize=10)
                tick_values = [vmin, -1.0, 0.0, 1.0, vmax]
                tick_labels = [
                    f"{vmin:.2f}",
                    "-1.00\n(Reverse)",
                    "0.00\n(stop)",
                    "1.00\n(Normal)",
                    f"{vmax:.2f}"
                ]
                cbar.set_ticks(tick_values)
                cbar.set_ticklabels(tick_labels)

            ax[i].set_title(title, fontsize=14, fontweight='bold', pad=16)
            ax[i].set_xlabel(xlabel, fontsize=12)
            ax[i].set_ylabel(ylabel, fontsize=12)

            # === 軸のtick ===
            ax[i].set_xticks(x_ticks)
            ax[i].set_yticks(y_ticks)
            if time_axis == 'x':
                ax[i].xaxis.set_major_formatter(fx)
            else:
                ax[i].yaxis.set_major_formatter(fx)

            ax[i].tick_params(axis='both', labelsize=10)
            ax[i].grid(True, linestyle='--', alpha=0.35)

            # ---- 軸ラベル（図の左側に配置）----
            ax[i].text(-0.08, 0.02, bottom_label, transform=ax[i].transAxes,
                    ha='right', va='bottom', fontsize=10, color=text_color, rotation=90)

            ax[i].text(-0.08, 0.98, top_label, transform=ax[i].transAxes,
                    ha='right', va='top', fontsize=10, color=text_color, rotation=90)
            
            # 下側横ラベル
            ax[i].text(0.02, -0.08, left_label, transform=ax[i].transAxes,
                    ha='left', va='top', fontsize=10, color=text_color)
            
            ax[i].text(0.98, -0.08, right_label, transform=ax[i].transAxes,
                    ha='right', va='top', fontsize=10, color=text_color)

            # ====== モノクロ画像出力（16bit） ======
            if save_png:
                # --- データの向き調整 ---
                if getattr(self, "scan_direction", 1) == 1:
                    # scan_direction=1 → 時間軸をYに取る → 上が時間0 → 下が時間経過
                    # グラフと一致させるため、上下反転かつ転置して保存
                    # data_to_save = np.flipud(data.T)
                    data_to_save = data.T
                else:
                    # 通常 → X軸方向に時間経過
                    data_to_save = data

                # --- 正規化＆保存処理 ---
                if t == "space":
                    norm = (np.clip((data_to_save - vmin) / (vmax - vmin), 0, 1) * 65535).astype(np.uint16)
                    t = f"{t}_{int(self.data.shape[1])}"

                elif t == "time":
                    norm = (np.clip((data_to_save - vmin) / (vmax - vmin), 0, 1) * 65535).astype(np.uint16)
                    t = f"{t}_{int(vmin)}-{int(vmax)}"

                else:  # rate
                    max_dev = max(np.abs(data_to_save.max() - 1.0), np.abs(data_to_save.min() - 1.0))
                    if max_dev < 1.0:
                        max_dev = 1.0
                    vmin, vmax = 1.0 - max_dev, 1.0 + max_dev
                    norm = (data_to_save - vmin) / (vmax - vmin)
                    norm = np.clip(norm, 0, 1)
                    norm = (norm * 65535).astype(np.uint16)
                    t = f"{t}_{max_dev:.2f}"

                # --- ファイル出力 ---
                png_name = f"{self.ORG_NAME}_{self.out_name_attr}_outimage_{t}.png"
                cv2.imwrite(png_name, norm)

        # === 保存 ===
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        file = f"{self.ORG_NAME}_{self.out_name_attr}_maneuver_visualization_{plot_mode}.png"
        plt.savefig(file, dpi=300, transparent=True, bbox_inches='tight')
        plt.close()



    def animationout(self,out_framenums=100,drawLineNum=250,dpi=200,out_fps=10):#out_framenums=50,out_fps=10
        runFirstTime = time.time()
        if self.data.shape[0]< out_framenums:
            out_framenums = self.data.shape[0]
        cap = cv2.VideoCapture(self.out_videopath)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)# 幅
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)# 高さ

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
        
        ax.zaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")
        ax.xaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")
        ax.yaxis.set_tick_params(labelsize=9,color="#cccccc",labelcolor="#aaaaaa")

        xMin=0
        inputFrame_xMax=self.scan_nums if self.scan_direction == 1  else self.slit_length 
        xMax=self.data.shape[1] if self.scan_direction == 1  else self.slit_length 
        
        # xMax=self.scan_nums
        zMin=np.amin(wr_array[:,:,1])
        zMax=np.amax(wr_array[:,:,1])
        ax.set_xlim(xMin, inputFrame_xMax) # x軸固定
        ax.set_ylim(zMin, zMax) # z軸固定
        yMin=0
        yMax =self.slit_length  if self.scan_direction == 1 else self.scan_nums        
        ax.set_xticks(np.arange(0,xMax+1,math.floor(xMax/2)))
        if (zMax-zMin)> 5000 : ax.set_yticks(np.arange(zMin,zMax+1,3000))
        elif  (zMax-zMin)> 3000:ax.set_yticks(np.arange(zMin,zMax+1,1500))
        elif  (zMax-zMin)> 1500:ax.set_yticks(np.arange(zMin,zMax+1,800))
        else: ax.set_yticks(np.arange(zMin,zMax+1,300))
        ax.set_zticks(np.arange(0,yMax+1,math.floor(yMax/2)))
        animate_steps=math.floor(wr_array.shape[0]/out_framenums)
        slit_len_dim=math.floor((yMax/xMax)*drawLineNum)  
        slit_len_steps = math.ceil(yMax/ slit_len_dim)
        slit_len_dim = math.ceil(yMax / slit_len_steps)
        frame_scan_hsteps=math.ceil(int(height) /slit_len_dim)  if self.scan_direction%2 == 1 else math.ceil(int(height) / drawLineNum) 
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
        # HDR PQ/HLG 出力を読み戻すと facecolors が褪せて見えるためトーンマップする
        _anim_transfer = _probe_video_transfer(self.out_videopath)
        if _anim_transfer in ("smpte2084", "arib-std-b67"):
            print(f"  [animationout] HDR源 ({_anim_transfer}) を検出 → PQ/HLG→sRGBトーンマップを適用")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True :
                if num%animate_steps == 0 :
                    for n in range(slit_len_dim):
                        Yarray[n]=np.full(drawLineNum,n*yMax/slit_len_dim) if self.scan_direction == 1 else wr_array[num,::scan_steps,0]
                    for n in range(slit_len_dim):
                        Xarray[n]=wr_array[num,::scan_steps,0] if self.scan_direction == 1 else np.full(drawLineNum,n*xMax/slit_len_dim)
                    for n in range(slit_len_dim):
                        Zarray[n]=wr_array[num,::scan_steps,1]

                    if self.scan_direction == 1 :
                        # refColorIMG=frame[::slit_len_steps,::frame_scan_steps,:]
                        refColorIMG=frame[::frame_scan_hsteps,::frame_scan_steps,:]
                        colors = refColorIMG[::-1,:,::-1]/255 #画像の天地合わせと、BGRからRGBへの変換
                    else:
                        refColorIMG=np.rot90(frame)[::frame_scan_steps,::frame_slit_len_steps,:] #90度回転
                        colors = refColorIMG[::-1,::-1,::-1]/255 #画像の天地合わせと、BGRからRGBへの変換

                    if _anim_transfer in ("smpte2084", "arib-std-b67"):
                        colors = _tonemap_hdr_rgb01_to_srgb(colors, _anim_transfer)
                        
                    

                    if colors.shape[0] != Xarray.shape[0]:
                        newcolors=np.zeros((Xarray.shape[0],Xarray.shape[1],3))
                        newcolors[:colors.shape[0],:colors.shape[1],:]=colors[:,::]
                        colors=newcolors


                    ax.plot_surface(Xarray,Zarray,Yarray,rstride=1,cstride=1,facecolors=colors,shade=False)
                    ax.set_xlim(xMin, inputFrame_xMax) # x軸固定
                    ax.set_ylim(zMin, zMax) # z軸固定
                    ax.set_xticks(np.arange(0,inputFrame_xMax+1,math.floor(inputFrame_xMax/2)))
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
                    # ax.xaxis.set_tic k_params(color='none',labelcolor='none')
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
                    # ax.view_init(elev=25, azim=current_azim+(30/out_framenums))
                    
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

    
    # レンダリング済みの映像ファイルと、軌道配列から、軌道の3D可視化アニメーションを作成する。aspect_ratio=(x,t,y)
    def animationout_custome(self,zRangeFix=False,out_framenums=100,drawLineNum=250,dpi=200,out_fps=10,aspect_ratio=(16, 50, 9),elev=25, azim=-40,colormode='white',transparent=False,gridplot=True,vectorplot=False,vector_def_frame=120,velocity=10,vector_color_amp=1.0,s_frame=0):#out_framenums=50,out_fps=10
        runFirstTime = time.time()
        print(self.data.shape,out_framenums)
        if self.data.shape[0]< out_framenums:
            out_framenums = self.data.shape[0]
        cap = cv2.VideoCapture(self.out_videopath)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)# 幅
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)# 高さ
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        wr_array = self.data.astype(np.int32)
        print(width,height,count,wr_array.shape)
        num_rate= count / wr_array.shape[0]
        
        if vectorplot :
            j = self.sc_resetPositionMap.shape[0]
            drawVectorNum = j
            # カラーマップと正規化
            # カラーマップと色の設定
            def get_color(v, vmin=-velocity, vmax=velocity):
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
                norm_v = norm(v)
                # 黄色から青への線形補間
                yellow = np.array([1, 1, 0,1])
                blue = np.array([0, 0, 1,1])
                color = (1 - norm_v) * blue + norm_v * yellow
                return color

        formatter_y = ticker.FuncFormatter(lambda y, pos: frames_to_min_sec(y, self.recfps))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.view_init(elev=25, azim=-40)
        if colormode == 'black':
            bg_color = (0, 0, 0)
            edge_color = (1, 1, 1,0.11)
            # grid_color = (1, 1, 1,0.3)
            grid_color = (0.3,0.3,0.3, 1)
            # label_color = (1, 1, 1,0.6)
            tick_color = (1, 1, 1,0.6)
            pane_color = (0,0,0,0.0)  # なし
            linewidth = 0.6  # 線の太さを指定
            grid_linewidth = 0.5  # グリッド線の太さを指定
            edge_linewidth = 0.5
            

        else:
            bg_color = (1, 1, 1)
            edge_color = (0, 0, 0, 0.5)
            grid_color = "#dddddd"
            label_color = (0, 0, 0)
            tick_color = (0, 0, 0)
            pane_color = (1, 1, 1, 0.6) if transparent == False else (1,1,1,0) # 半透明の白
            linewidth = 1  # 線の太さを指定
            grid_linewidth = 0.8  # グリッド線の太さを指定
            edge_linewidth = 0.5
        
        if vectorplot :vector_linewidth=0.6

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)

        ax.xaxis.line.set_color(pane_color)
        ax.yaxis.line.set_color(pane_color)
        ax.zaxis.line.set_color(pane_color)
        
        ax.xaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)
        ax.yaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)
        ax.zaxis._axinfo['grid'].update(color=grid_color, linewidth=grid_linewidth)
        ax.view_init(elev=elev, azim=azim)

        # xlim は maneuver_3dplot と揃えるためデータ駆動 (self.data[:,:,0] の範囲 + 2%マージン)
        _xp_min = float(np.amin(self.data[:, :, 0]))
        _xp_max = float(np.amax(self.data[:, :, 0]))
        _xp_margin = (_xp_max - _xp_min) * 0.02
        xMin = max(0.0, _xp_min - _xp_margin)
        inputFrame_xMax = _xp_max + _xp_margin
        xMax=self.data.shape[1] if self.scan_direction == 1  else self.slit_length

        # xMax=self.scan_nums
        zMin=np.amin(wr_array[:,:,1])
        zMax=np.amax(wr_array[:,:,1])
        ax.set_xlim(xMin, inputFrame_xMax) # x軸固定
        ax.set_ylim(zMin, zMax) # z軸固定
        yMin=0
        yMax =self.slit_length  if self.scan_direction == 1 else self.scan_nums        
        ax.set_xticks(np.arange(0,xMax+1,math.floor(xMax/2)))
        if (zMax-zMin)> 5000 : ax.set_yticks(np.arange(zMin,zMax+1,3000))
        elif  (zMax-zMin)> 3000:ax.set_yticks(np.arange(zMin,zMax+1,1500))
        elif  (zMax-zMin)> 1500:ax.set_yticks(np.arange(zMin,zMax+1,800))
        else: ax.set_yticks(np.arange(zMin,zMax+1,300))
        ax.set_zticks(np.arange(0,yMax+1,math.floor(yMax/2)))
        # animate_steps=math.floor(wr_array.shape[0]/out_framenums)
        animate_steps=math.floor(count/out_framenums)
        print("animate_steps=",animate_steps,count,out_framenums,count//out_framenums)
        slit_len_dim=math.floor((yMax/xMax)*drawLineNum)  
        slit_len_steps = math.ceil(yMax/ slit_len_dim)
        slit_len_dim = math.ceil(yMax / slit_len_steps)
        frame_scan_hsteps=math.ceil(int(height) /slit_len_dim)  if self.scan_direction%2 == 1 else math.ceil(int(height) / drawLineNum) 
        frame_scan_steps=math.ceil(int(width) / drawLineNum)  if self.scan_direction%2 == 1 else math.ceil(int(width) / slit_len_dim) 
        scan_steps=math.ceil(xMax / drawLineNum)  if self.scan_direction%2 == 1 else math.ceil(yMax / drawLineNum) 
        drawLineNum=wr_array[0,::math.ceil(xMax /drawLineNum),0].shape[0] if self.scan_direction == 1 else wr_array[0,::math.ceil(yMax /drawLineNum),0].shape[0] 
        frame_slit_len_steps = math.ceil(yMax/ slit_len_dim) if self.scan_direction==1 else math.ceil(yMax/ drawLineNum)
        Yarray=np.zeros((slit_len_dim,drawLineNum)) 
        Xarray=np.zeros((slit_len_dim,drawLineNum))
        Zarray=np.zeros((slit_len_dim,drawLineNum))

        num = s_frame
        if os.path.isdir("sequence")==False:
            os.makedirs("sequence")
        if not cap.isOpened():
            print("Error: Cannot open video.")
        # HDR PQ/HLG 出力を読み戻すと facecolors が褪せて見えるためトーンマップする
        _anim_transfer = _probe_video_transfer(self.out_videopath)
        if _anim_transfer in ("smpte2084", "arib-std-b67"):
            print(f"  [animationout_custome] HDR源 ({_anim_transfer}) を検出 → PQ/HLG→sRGBトーンマップを適用")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True :
                if num%animate_steps == 0 and wr_array.shape[0] >= num // num_rate:
                    for n in range(slit_len_dim):
                        Yarray[n]=np.full(drawLineNum,n*yMax/slit_len_dim) if self.scan_direction == 1 else wr_array[int(num // num_rate) ,::scan_steps,0]
                    for n in range(slit_len_dim):
                        Xarray[n]=wr_array[int(num // num_rate),::scan_steps,0] if self.scan_direction == 1 else np.full(drawLineNum,n*xMax/slit_len_dim)
                    for n in range(slit_len_dim):
                        Zarray[n]=wr_array[int(num // num_rate),::scan_steps,1]

                    if self.scan_direction == 1 :
                        # refColorIMG=frame[::slit_len_steps,::frame_scan_steps,:]
                        refColorIMG=frame[::frame_scan_hsteps,::frame_scan_steps,:]
                        colors = refColorIMG[::-1,:,::-1]/255 #画像の天地合わせと、BGRからRGBへの変換
                    else:
                        refColorIMG=np.rot90(frame)[::frame_scan_steps,::frame_slit_len_steps,:] #90度回転
                        colors = refColorIMG[::-1,::-1,::-1]/255 #画像の天地合わせと、BGRからRGBへの変換

                    if _anim_transfer in ("smpte2084", "arib-std-b67"):
                        colors = _tonemap_hdr_rgb01_to_srgb(colors, _anim_transfer)

                    if colors.shape[:2] != Xarray.shape[:2]:
                        # newcolors=np.zeros((Xarray.shape[0],Xarray.shape[1],3))
                        target_size=(Xarray.shape[1],Xarray.shape[0])
                        newcolors = cv2.resize(colors, target_size, interpolation=cv2.INTER_NEAREST)
                        colors=newcolors

                    zMin = np.amin(wr_array[int(num // num_rate), :, 1])
                    zMax = np.amax(wr_array[int(num // num_rate), :, 1])

                    if zRangeFix:
                        ax.set_ylim(zMin, zMax)  # z軸固定
                    else:
                        # y軸の表示範囲を設定
                        y_range_min = min(zMin, zMax - 5760)
                        y_range_max = max(zMax, zMin + 5760)
                        y_range=y_range_max-y_range_min
                        ax.set_ylim(y_range_min, y_range_max)
                        ax.set_box_aspect((aspect_ratio[0],np.clip(int(aspect_ratio[1]*(y_range / 11520)),aspect_ratio[1],aspect_ratio[0]*10),aspect_ratio[2]))

                    ax.set_xlim(xMin, inputFrame_xMax) # x軸固定
                    ax.xaxis.set_ticks([])
                    ax.zaxis.set_ticks([])

                    ax.plot_surface(Xarray,Zarray,Yarray,rstride=1,cstride=1,facecolors=colors,shade=False)
                       # y軸の目盛りを可変に設定して、7本に制限
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
                    ax.yaxis.set_major_formatter(formatter_y)
                    ax.xaxis.line.set_color(pane_color)
                    ax.yaxis.line.set_color(pane_color)
                    ax.zaxis.line.set_color(pane_color)
                    ax.zaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)
                    ax.xaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)
                    ax.yaxis.set_tick_params(labelsize=5, color=tick_color, labelcolor=tick_color)
                    
                    if gridplot ==False:
                        ax.yaxis.set_ticks([])

                    if vectorplot :
                        for k in range(drawVectorNum):
                            rp = k * math.ceil(self.data.shape[1] / (drawVectorNum - 1)) if k != (drawVectorNum - 1) else (self.data.shape[1] - 1)
                            pastframe=num-vector_def_frame
                            if pastframe < 0 : pastframe=0
                            x = self.data[pastframe, rp, 0]
                            y = self.data[num, rp, 1] - self.sc_rateMap[k,num]*velocity
                            z = 0
                            u = self.data[num, rp, 0] - x
                            v= self.sc_rateMap[k,num] * velocity
                            w = 0
                            for l in range(0,self.slit_length,(self.slit_length-1)//3):
                                ax.quiver(x, y, z+l, u, v, w, color=get_color(v * vector_color_amp), linewidth=vector_linewidth,arrow_length_ratio=0.6)
                                
                    # 現在の視点の角度を取得する
                    # current_elev = ax.elev
                    current_azim = ax.azim
                    plt.savefig('sequence/'+str(self.log)+'_3dPlot_pixColor-'+str(num)+'.png',dpi=dpi,transparent=transparent,bbox_inches='tight')
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
        for i in range(0,int(count),animate_steps):
        
            # figvideo.write(cv2.imread('sequence/'+ str(self.log) +'_3dPlot_pixColor-'+str(i)+'.png'))
            # os.remove(self.ORG_NAME+"_"+self.out_name_attr +'_3dPlot_pixColor-'+str(i)+'.png')
            readimg = cv2.imread('sequence/'+ str(self.log) +'_3dPlot_pixColor-'+str(i)+'.png',cv2.IMREAD_UNCHANGED)
            if img.shape != readimg.shape:
                readimg = cv2.resize(readimg, (img.shape[1], img.shape[0]))
            # 白い背景部分を黒に変換
            # アルファチャンネルを分離
            b, g, r, a = cv2.split(readimg)
            # アルファチャンネルがゼロ（透明）の部分を黒に設定
            black_background = np.zeros_like(a)
            black_background[a == 0] = 255
            # カラー画像にアルファチャンネルを適用
            readimg = cv2.merge([b, g, r])
            if colormode == "black":
                readimg[black_background == 255] = [0, 0, 0]
            else : 
                readimg[black_background == 255] = [255,255,255]
            figvideo.write(readimg)

        figvideo.release()
        runOverTime = time.time()
        lnterval = round(runOverTime-runFirstTime,2)
        print("All Done",lnterval,"sec")


    # dataに新たな軌跡を加えて返す関数
    def movement_intensity_analyze(self):
        """
        Process the video to adjust playback speed dynamically.
        """
        def calculate_frame_difference(frame1, frame2):
            """
            Calculate motion intensity between two frames.
            """
            diff = cv2.absdiff(frame1, frame2)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            mean_diff = np.mean(diff_gray)
            return mean_diff
        def plot_frame_differences(frame_diffs, fps,y_range=(0,8)):
            """
            Plot motion intensity over time and save the graph as a file.
            """
            time = np.arange(len(frame_diffs)) / fps
            plt.figure(figsize=(10, 6))
            plt.plot(time, frame_diffs, label="Motion Intensity")
            plt.axhline(y=np.mean(frame_diffs[1:]), color='r', linestyle='--', label="Baseline (P)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Motion Intensity")
            plt.title("Motion Intensity Over Time")
            plt.legend()
            plt.grid(True)
            # Set Y-axis range if specified
            if y_range:
                plt.ylim(y_range)

            # Save the plot to a file in the same directory as the input video
            plt.savefig(self.ORG_NAME+"_"+self.out_name_attr+'_graph.png')
            print(f"Graph saved")

            # plt.show()
        # Calculate frame differences
        print("Calculating motion intensity...")
        ret, prev_frame = self.cap.read()
        # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        print(f"First frame size: {prev_frame.shape}")

        frame_diffs = []
        for i in range(1, int(self.count)):
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break
            # print(f"Current frame size: {frame.shape if frame is not None else 'None'}")
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = calculate_frame_difference(prev_frame,frame)
            frame_diffs.append(diff)
            prev_frame = frame
            
            # Progress
            sys.stdout.write(f"\rProcessing frame diff={diff}_{i}/{self.count}...")
            sys.stdout.flush()
        
        print("\nMotion intensity calculation complete.")
        
        # Calculate baseline P
        P = np.mean(frame_diffs[1:])
        print(f"Baseline motion intensity (P): {P}")
        # Plot and save frame differences
        plot_frame_differences(frame_diffs, self.recfps)
        print(f"Processed video saved to")


# =====================================================================
# HDR PQ (Rec.2100) → sRGB トーンマッピング変換
# =====================================================================

