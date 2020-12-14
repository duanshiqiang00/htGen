%%
% C919的分割完成的数据 其中文件目录中包含原有的紊流激励响应信号和基于waveUNet的时域传递函数的结果 
% 原始文件目录中包含
%   --原始的紊流响应信号
%   --WaveUNet的紊流响应信号的结果
%
%
%%
clc
clear
PathRoot='E:\python\CNNLSTMGenerateFRFht\result\remote_turbhtGen_tur1order_T-20s_Fs-512_E-50000_LR-0.0001_LayerNum-10_filterNum-2\C919TurResSignal';
T=20;Fs=512;
t=0:1/Fs:T;
t= t(1:length(t)-1);
f=0:Fs/(T*Fs):Fs;
f=f(1:length(f)/2);

list=dir(fullfile(PathRoot));
for i=3:size(list,1)
    filename=list(i).name;
    filenamesplit = split(filename(1:length(filename)-4),{'-'});
    
    fileNameTypeIndex = filenamesplit{length(filenamesplit)};
    if(strcmp( fileNameTypeIndex,'predht')==1)
        continue
    else
        resSignalFileName = filename;
        predSysFileName = [filename(1:length(filename)-4),'-predht.txt'];
        resSignalfilePath= [PathRoot,'\',resSignalFileName];
        predSysFilePath = [PathRoot,'\',predSysFileName];
        resSignal = load(resSignalfilePath);
        resSignal = resSignal(1:T*Fs);
        predSysData = load(predSysFilePath);
        predSysData = predSysData(1:T*Fs);
        figure
        subplot(2,1,1)
        plot(t,resSignal)
        title("time series of turbulence response")
        
        subplot(2,1,2)
        plot(t,predSysData,'b-')        
        title("time series of predicted time domain ht by waveUNet")
        timeserierfigPath = [resSignalfilePath(1:length(resSignalfilePath)-4),'.jpg' ];
        signalFFtfigPath = [resSignalfilePath(1:length(resSignalfilePath)-4),'fft.jpg' ];
        saveas(gcf,timeserierfigPath)
        close
        figure
        subplot(2,1,1)
        data1= abs(fft(resSignal));
        data1=data1(1:length(data1)/2);        
        plot(f,data1)
        title("spectrum of turbulence response")
        subplot(2,1,2)        
        data3= abs(fft(predSysData));
        data3=data3(1:length(data3)/2);
        plot(f,data3,'b-')
        title("spectrum of predicted ht by WaveUNet")
        saveas(gcf,signalFFtfigPath)
        close
        
    end
       
end
