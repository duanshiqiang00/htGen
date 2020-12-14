%%
% 用于训练和测试结果的画图 
% 数据文件三列 第一列响应信号 第二列真实的时域传递函数 第三列神经网络计算的
% 时域传递函数 
% 两张图 分别是时域的波形图（第一行是响应信号，第二行是传递函数的真实和预测结果对比）
% 以及频谱（与时域类似，分别是响应信号和时域传递函数的频谱对比）
%%
clc
clear

PathRoot='E:\python\CNNLSTMGenerateFRFht\result\remote_1&2OrderturbhtGen_tur_1&2order_T-20s_Fs-512_E-100007_LR-0.0001_LayerNum-10_filterNum-2\testData_result\remoteHost_testResSys_T-20_Fs-512_LayerNum-10_filterNum-2_Epoch-100007_LR-0.0001';
T=20;Fs=512;
t=0:1/Fs:T;
t= t(1:length(t)-1);
f=0:Fs/(T*Fs):Fs;
f=f(1:length(f)/2);

list=dir(fullfile(PathRoot));
for i=3:size(list,1)
    filename=list(i).name;
    if(filename(length(filename)-3:length(filename))=='.txt')
        filePath= [PathRoot,'\',filename];
        data = load(filePath);
        figure
        subplot(2,1,1)
        plot(t,data(:,1))
        
        subplot(2,1,2)
        plot(t,data(:,3),'b-')        
        hold on
        plot(t,data(:,2),'r')
        legend("pred ht","pred ht")
        timeserierfigPath = [filePath(1:length(filePath)-4),'.jpg' ];
        signalFFtfigPath = [filePath(1:length(filePath)-4),'fft.jpg' ];
        saveas(gcf,timeserierfigPath)
        close
        figure
        subplot(2,1,1)
        data1= abs(fft(data(:,1)));
        data1=data1(1:length(data1)/2);
        
        plot(f,data1)
        subplot(2,1,2)        
        data3= abs(fft(data(:,3)));
        data3=data3(1:length(data3)/2);
        plot(f,data3,'b-')
        hold on 
        data2= abs(fft(data(:,2)));
        data2=data2(1:length(data2)/2);
        plot(f,data2,'r')
        legend("pred ht spectrum","real ht spectrum")
        saveas(gcf,signalFFtfigPath)
        close
    end
end
