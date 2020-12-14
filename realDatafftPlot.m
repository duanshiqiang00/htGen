%%
% 海鸥300实测数据的结果显示 分别读取实测数据的紊流响应信号数据和神经网络的计算结果画图，
% 由于实测数据没有传递函数的真实值，此处只画紊流响应和计算结果
%
%
%%
clc
clear
realDataDirPath = 'E:\python\CNNLSTMGenerateFRFht\result\remote_multiOrderturbhtGen_tur_T-20s_Fs-512_E-100005_LR-0.0001_LayerNum-10_filterNum-2\realData_tur_haiou300';
T=20;Fs=512;
t=0:1/Fs:T;
t= t(1:length(t)-1);
f=0:Fs/(T*Fs):Fs;
f=f(1:length(f)/2);

for i=0:24
    num=i+1;
    realDataFileName = ['wl19090320200614F1VIB101234-',num2str(num),'.txt'];
    realDataFilePath = [realDataDirPath,'\',realDataFileName];
    realData =load(realDataFilePath);
    realData = realData(1:length(t));
    predhtFileName =  [realDataFileName(1:length(realDataFileName)-4),'-predht.txt'];
    predhtFilePath = [realDataDirPath,'\',predhtFileName];
    predhtData = load(predhtFilePath);
    predhtData=predhtData';
    timeseriesfigName=[realDataFileName(1:length(realDataFileName)-4),'-timeSeries.png'];
    spectrumfigName=[realDataFileName(1:length(realDataFileName)-4),'-fft.png'];
    timeseriesfigPath=[realDataDirPath,'\',timeseriesfigName];
    spectrumfigPath=[realDataDirPath,'\',spectrumfigName];
    figure()
    subplot(211)
    plot(t,realData)
    subplot(212)
    plot(t,predhtData)
    saveas(gcf,timeseriesfigPath)
    close
    
    figure()
    subplot(211) 
    realDatafft = abs(fft(realData));
    realDatafft=realDatafft(1:length(realDatafft)/2);
    plot(f,realDatafft)    
    predhtDatafft = abs(fft(predhtData));
    predhtDatafft = predhtDatafft(1:length(predhtDatafft)/2);
    subplot(212)
    plot(f,predhtDatafft)
    saveas(gcf,spectrumfigPath)
%     close
    


end