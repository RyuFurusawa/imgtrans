ServerOptions.devices; // all devices
ServerOptions.inDevices;
Server.default.options.outDevice_("外部");
Server.default.options.outDevice_("H6");
Server.default.options.outDevice_("BlackHole 64ch");
Server.default.options.outDevice_("Duet 3");
Server.default.options.outDevice_("MacBook Proのス");
Server.default.options.outDevice_("WF-1000XM3");
Server.default.options.outDevice_("Bose USB Audio");
Server.default.options.outDevice_("BlackHole 64ch");
ServerOptions.numChannels;

s.options.numOutputBusChannels=4;//デフォルトでは2、マルチチャンネルの場合はここ変更する必要がある。
s.options.numOutputBusChannels;
s.boot;
s.quit;

//AudioSetting

(
Server.default.options.outDevice_("外部");
s.options.numOutputBusChannels=2;
s.options.numOutputBusChannels;
s.options.inDevice_("MacBook Proの");
s.options.outDevice_("外部");
s.boot;
s.meter;
s.plotTree;
s.scope;
)

(
Server.default.options.outDevice_("BlackHole 64ch");
s.options.numOutputBusChannels=4;
s.options.numOutputBusChannels;
s.options.inDevice_("MacBook Proの");
s.boot;
s.meter;
s.plotTree;
s.scope;
)

//mic使わない場合
(
Server.default.options.outDevice_("Duet 3");
s.options.numOutputBusChannels=2;
s.options.numOutputBusChannels;
s.plotTree;
s.boot;
)
//インプットとアウトプットのサンプルレートの差によるエラーのばあい
"open -a 'Audio MIDI Setup'".unixCmd;

"open -a 'QuickTime Player'".unixCmd;

v=0.3;
{SinOsc.ar(440,0,[v,0,0,0])}.play;
{SinOsc.ar(880,0,[0,v,0,0])}.play;
{SinOsc.ar(1760,0,[0,0,v,0])}.play;
{SinOsc.ar(220,0,[0,0,0,v])}.play;

(
x= {
	var sig;
	sig = SoundIn.ar(0!2);
	sig = sig + CombN.ar(sig,delaytime:1.0, decaytime:3,mul:0.5);//delay
}.play;
)


//映像の3次元マップ情報を読み込む
r=[
"/Users/ryufurusawa/Movies/xperia/2022-11-21/VideoPro_20221121_154733-t1_Vertical_wavetrans_2000seq_120deg-Wave-noFlow500zdepth_2022_1121-2030/VideoPro_20221121_154733-t1_Vertical_wavetrans_2000seq_120deg-Wave-noFlow500zdepth_ResetP.csv",
"/Users/ryufurusawa/Movies/xperia/2022-11-21/VideoPro_20221121_154733-t1_Vertical_wavetrans_2000seq_120deg-Wave-noFlow500zdepth_2022_1121-2030/VideoPro_20221121_154733-t1_Vertical_wavetrans_2000seq_120deg-Wave-noFlow500zdepth_Rate.csv",
"/Users/ryufurusawa/Movies/xperia/2022-11-21/VideoPro_20221121_154733-t1_Vertical_wavetrans_2000seq_120deg-Wave-noFlow500zdepth_2022_1121-2030/VideoPro_20221121_154733-t1_Vertical_wavetrans_2000seq_120deg-Wave-noFlow500zdepth_inPanMap.csv","/Users/ryufurusawa/Movies/xperia/2022-11-21/VideoPro_20221121_154733-t1-AIFF.aiff"];

x = CSVFileReader.readInterpret(r[0]).postcs;//ResetP
y = CSVFileReader.readInterpret(r[1]).postcs;//Rate
z = CSVFileReader.readInterpret(r[2]).postcs;//inPanMap

a = Buffer.readChannel(s,r[3],channels:[1]);
b = Buffer.readChannel(s,r[3],channels:[0]);
x=[0,0,0];
max(c[1]);
r[2]
z[0][0]
z[0][2800]
b.play;
b.plot;
b.numFrames / b.sampleRate / 60 ; //min
a.free;
b.free;
b.bufnum.post;

//////////////////////////////////
//////////////////////////////////
//////////////////////////////////
(
SynthDef(\simplePlay, {
	arg bufL = 0 , bufR = 1,trig = 0.3, rate=1, pan=0.5, inpan=0.5,resetp=1000,amp=0.1,lag=(1/30),out=0,v=0.3;
	var sigL,sigR;
	sigL = PlayBuf.ar(1,bufL, BufRateScale.kr(bufL) *  Lag2UD.kr(rate,lag,lag), trigger:Impulse.kr(trig), startPos:resetp*BufSampleRate.kr(bufL), loop:1);
	sigR = PlayBuf.ar(1,bufR, BufRateScale.kr(bufR) *  Lag2UD.kr(rate,lag,lag), trigger:Impulse.kr(trig), startPos:resetp*BufSampleRate.kr(bufR), loop:1);
	//2ch out
	Out.ar(out, sigL * [1*pan,1*(1-pan)]* Lag2UD.kr(inpan,lag,lag)*amp);
	Out.ar(out, sigR * [1*(1-pan),1*pan]* Lag2UD.kr(inpan,lag,lag)*amp);

	//4ch out
	// Out.ar(out, sigL * max(0,(1-[abs(1-pan),abs(2/3-pan),abs(1/3-pan),abs(0-p)]-v) / v)* Lag2UD.kr(inpan,lag,lag)*amp);
	// Out.ar(out, sigR * max(0,(1-[abs(1-pan),abs(2/3-pan),abs(1/3-pan),abs(0-p)]-v) / v)* Lag2UD.kr(inpan,lag,lag)*amp);
}).add;

//////////////////////////////////


/*SynthDef.new(\bufgrain,{
	arg out=0,rate=1,dur=0.1,trig=0,resetp=0.1,pan=0.5,amp=1;
		var sig;
	sig = GrainBuf.ar(
		2,
		trigger:Impulse.ar(30),
		dur:dur,
		sndbuf:b,
		rate:rate,//MouseX.kr(0.5,2.1)
		pos:(Phasor.ar(Impulse.ar(trig),rate*BufRateScale.ir(b),0,BufSamples.ir(b),resetp*BufSamples.ir(b))) / BufSamples.ir(b),
		//(Phasor.ar(0,MouseY.kr(0.1,2,1)*BufRateScale.ir(b),0,BufSamples.ir(b)-1)+LFNoise1.ar(100).bipolar(0.0*SampleRate.ir)) / BufSamples.ir(b)
		interp:2,
		pan:0,
		envbufnum:-1,
		maxGrains:512);
	Out.ar(out, sig * [1*pan,1*(1-pan)]*amp);
}).add;*/

//////////////////////////////////

SynthDef.new(\bufgrainLR,{
	arg bufL = 0 , bufR = 1,out=0,rate=1,dur=0.1,trig=0,resetp=0.1,pan=0.5,inpan=0.5,ampl=1,ampr=1,amp=1,lag=(1/30),v=0.3;
		var sigL,sigR;
	sigL = GrainBuf.ar(
		2,
		trigger:Impulse.ar(30),
		dur:dur,
		sndbuf:bufL,
		rate:1,//MouseX.kr(0.5,2.1)
		pos:(Phasor.ar(Impulse.ar(trig),Lag2UD.kr(rate,lag,lag)*BufRateScale.ir(bufL),0,BufSamples.ir(bufL),resetp*BufSamples.ir(bufL))) / BufSamples.ir(bufL), //(Phasor.ar(0,MouseY.kr(0.1,2,1)*BufRateScale.ir(b),0,BufSamples.ir(b)-1)+LFNoise1.ar(100).bipolar(0.0*SampleRate.ir)) / BufSamples.ir(b)
		interp:2,
		pan:0,
		envbufnum:-1,
		maxGrains:512);
	sigR = GrainBuf.ar(
		2,
		trigger:Impulse.ar(30),
		dur:dur,
		sndbuf:bufR,
		rate:1,//MouseX.kr(0.5,2.1)
		pos:(Phasor.ar(Impulse.ar(trig),Lag2UD.kr(rate,lag,lag)*BufRateScale.ir(bufR),0,BufSamples.ir(bufR),resetp*BufSamples.ir(bufR))) / BufSamples.ir(bufR), //(Phasor.ar(0,MouseY.kr(0.1,2,1)*BufRateScale.ir(b),0,BufSamples.ir(b)-1)+LFNoise1.ar(100).bipolar(0.0*SampleRate.ir)) / BufSamples.ir(b)
		interp:2,
		pan:0,
		envbufnum:-1,
		maxGrains:512);
	//2ch out
	Out.ar(out, sigL * [1*pan,1*(1-pan)]* Lag2UD.kr(inpan,lag,lag)*amp);
	Out.ar(out, sigR * [1*(1-pan),1*pan]* Lag2UD.kr(inpan,lag,lag)*amp);
	//4ch out
	// Out.ar(out, sigL * max(0,(1-[abs(1-pan),abs(2/3-pan),abs(1/3-pan),abs(0-p)]-v) / v)* Lag2UD.kr(inpan,lag,lag)*amp);
	// Out.ar(out, sigR * max(0,(1-[abs(1-pan),abs(2/3-pan),abs(1/3-pan),abs(0-p)]-v) / v)* Lag2UD.kr(inpan,lag,lag)*amp);
}).add;

//////////////////////////////////
SynthDef("preDelay", {
    ReplaceOut.ar(
        6,
        DelayN.ar(In.ar(4, 2), 0.048, 0.048)
    )
}).add;

SynthDef("combs", { arg dtime=0.1;
    ReplaceOut.ar(
        8,
        Mix.arFill(7, { CombL.ar(In.ar(6, 2), dtime, LFNoise1.kr(Rand(0, dtime), 0.04, 0.05), 15) })
    )
}).add;

SynthDef("allpass", { arg gain = 0.2;
    var source;
    source = In.ar(8, 2);
    4.do({ source = AllpassN.ar(source, 0.050, [Rand(0, 0.05), Rand(0, 0.05)], 1) });
    ReplaceOut.ar(
        10,
        source * gain
    )
}).add;

SynthDef("theMixer", { arg gain = 1,dry=1.0;
    ReplaceOut.ar(
        0,
        Mix.ar([In.ar(4, 2)*dry, In.ar(10, 2)]) * gain
    )
}).add;
)

//////////////////////////////////
//////////////////////////////////
//////////////////////////////////

(//映像の3次元マップ情報を読み込む
 r=["/Users/ryufurusawa/Movies/VideoPro_20230214_125545_eventhorizon_2000z_2023_0612/VideoPro_20230214_125545_eventhorizon_2000z_ResetP_7thread.csv",
"/Users/ryufurusawa/Movies/VideoPro_20230214_125545_eventhorizon_2000z_2023_0612/VideoPro_20230214_125545_eventhorizon_2000z_Rate_7thread.csv",
"/Users/ryufurusawa/Movies/VideoPro_20230214_125545_eventhorizon_2000z_2023_0612/VideoPro_20230214_125545_eventhorizon_2000z_inPanMap_7thread.csv",
"/Users/ryufurusawa/Movies/VideoPro_20230214_125545_eventhorizon_2000z_2023_0612/VideoPro_20230214_125545_eventhorizon_2000z_nowDepth_7thread.csv",
"/Users/ryufurusawa/Movies/VideoPro_20230214_125545-AIFF.aiff"];
x = CSVFileReader.readInterpret(r[0]).postcs;//ResetP
y = CSVFileReader.readInterpret(r[1]).postcs;//Rate
z = CSVFileReader.readInterpret(r[2]).postcs;//inPanMap
c = CSVFileReader.readInterpret(r[3]).postcs;//nowdepth
a = Buffer.readChannel(s,r[4],channels:[1]);
b = Buffer.readChannel(s,r[4],channels:[0]);
)


e=Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",1,"pan",0,"amp",1/7,"resetp",x[0].[0]/((b.numFrames / b.sampleRate)*120)]);


(
//Grainチャンネルミックスをrecording, LR分析バージョン
"open -a 'QuickTime Player' '/Users/ryufurusawa/Movies/VideoPro_20230214_125545_eventhorizon_2000z_2023_0612/VideoPro_20230214_125545_eventhorizon_2000z.mp4' ".unixCmd;
// s.record;
fork{var cha=7,num=x[0].size,recfps=119.88507790650145,trig=0.333,gs0,gs1,gs2,gs3,gs4,gs5,gs6;
    gs0 = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",0,"amp",1/cha,"resetp",x[0].[0]/((b.numFrames / b.sampleRate)*recfps)]);
    gs1 = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",0.16666666666666666,"amp",1/cha,"resetp",x[1].[0]/((b.numFrames / b.sampleRate)*recfps)]);
    gs2 = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",0.4,"amp",1/cha,"resetp",x[2].[0]/((b.numFrames / b.sampleRate)*recfps)]);
    gs3 = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",0.75,"amp",1/cha,"resetp",x[3].[0]/((b.numFrames / b.sampleRate)*recfps)]);
    gs4 = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",1.3333333333333333,"amp",1/cha,"resetp",x[4].[0]/((b.numFrames / b.sampleRate)*recfps)]);
    gs5 = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",2.5,"amp",1/cha,"resetp",x[5].[0]/((b.numFrames / b.sampleRate)*recfps)]);
    gs6 = Synth("bufgrainLR",["bufL",a,"bufR",b,"rate",1,"trig",trig,"pan",1,"amp",1/cha,"resetp",x[6].[0]/((b.numFrames / b.sampleRate)*recfps)]);
    num.do{arg i;
        i.post;"/".post;num.postln;
        (1/30).yield;

        gs0.set("resetp",x[0].[i]/((b.numFrames / b.sampleRate)*recfps));
        gs0.set("rate",y[0].[i]);
        gs0.set("inpan",z[0].[i]/3840);

        gs1.set("resetp",x[1].[i]/((b.numFrames / b.sampleRate)*recfps));
        gs1.set("rate",y[1].[i]);
        gs1.set("inpan",z[1].[i]/3840);

        gs2.set("resetp",x[2].[i]/((b.numFrames / b.sampleRate)*recfps));
        gs2.set("rate",y[2].[i]);
        gs2.set("inpan",z[2].[i]/3840);

        gs3.set("resetp",x[3].[i]/((b.numFrames / b.sampleRate)*recfps));
        gs3.set("rate",y[3].[i]);
        gs3.set("inpan",z[3].[i]/3840);

        gs4.set("resetp",x[4].[i]/((b.numFrames / b.sampleRate)*recfps));
        gs4.set("rate",y[4].[i]);
        gs4.set("inpan",z[4].[i]/3840);

        gs5.set("resetp",x[5].[i]/((b.numFrames / b.sampleRate)*recfps));
        gs5.set("rate",y[5].[i]);
        gs5.set("inpan",z[5].[i]/3840);

        gs6.set("resetp",x[6].[i]/((b.numFrames / b.sampleRate)*recfps));
        gs6.set("rate",y[6].[i]);
        gs6.set("inpan",z[6].[i]/3840);
        };
    };
)