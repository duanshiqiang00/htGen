%%
% ��׷�������ѵ��waveUnet  ��ģ̬�������ݻ�ͼ ��������Ŀ¼��ÿ�������ļ�������������
% --ԭʼ������Ӧ�ź�
% --waveUNet�ļ����ʱ�򴫵ݺ������
% --��ʵ��ʱ�򴫵ݺ������ 
% ���ļ�Ŀ¼�е��ļ����ַ������ղ�ͬ�����ͼ���������ݲ���ͼ
%
%%
clc
clear
PathRoot='E:\python\CNNLSTMGenerateFRFht\result\remote_multiOrderturbhtGen_tur_T-20s_Fs-512_E-100005_LR-0.0001_LayerNum-10_filterNum-2\simulatedDataVal\tur1order_T-20s_Fs-512';
T=20;Fs=512;
t=0:1/Fs:T;
t= t(1:length(t)-1);
f=0:Fs/(T*Fs):Fs;
f=f(1:length(f)/2);

list=dir(fullfile(PathRoot));
for i=3:size(list,1)
    filename=list(i).name;
    filenamesplit = split(filename(1:length(filename)-4),{'_'});
    
    fileNameTypeIndex = filenamesplit{length(filenamesplit)};
%     strcmp(fileNameTypeIndex,'freq-1&50')
    if(strcmp( fileNameTypeIndex,'realht')==1)
        continue
    elseif(strcmp( fileNameTypeIndex,'predht')==1)
        continue
    else
        resSignalFileName = filename;
        realSysFileName = [filename(1:length(filename)-4),'_realht.txt'];
        predSysFileName = [filename(1:length(filename)-4),'_predht.txt'];
        resSignalfilePath= [PathRoot,'\',resSignalFileName];
        realSysFilePath = [PathRoot,'\',realSysFileName];
        predSysFilePath = [PathRoot,'\',predSysFileName];
        resSignal = load(resSignalfilePath);
        realSysData = load(realSysFilePath);
        predSysData = load(predSysFilePath);
        figure
        subplot(2,1,1)
        plot(t,resSignal)
        
        subplot(2,1,2)
        plot(t,predSysData,'b-')        
        hold on
        plot(t,realSysData,'r')
        legend("pred ht","pred ht")
        timeserierfigPath = [resSignalfilePath(1:length(resSignalfilePath)-4),'.jpg' ];
        signalFFtfigPath = [resSignalfilePath(1:length(resSignalfilePath)-4),'fft.jpg' ];
        saveas(gcf,timeserierfigPath)
        close
        figure
        subplot(2,1,1)
        data1= abs(fft(resSignal));
        data1=data1(1:length(data1)/2);
        
        plot(f,data1)
        subplot(2,1,2)        
        data3= abs(fft(predSysData));
        data3=data3(1:length(data3)/2);
        plot(f,data3,'b-')
        hold on 
        data2= abs(fft(realSysData));
        data2=data2(1:length(data2)/2);
        plot(f,data2,'r')
        legend("pred ht spectrum","real ht spectrum")
        saveas(gcf,signalFFtfigPath)
        close
        
    end
       
end
