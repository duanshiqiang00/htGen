%%
% 一阶紊流响应信号的结果数据集生成 频率和阻尼两个变量各取stageNum个两层便利穷举出所有的紊流响应结果
%%
clc
clear
StageNumber=50;
fs=512;
T=20;
t=0:1/fs:T;
t=t(1:length(t)-1);
n=4;  %AR模型阶数

f10=1;f11=60;
f20=10;f21=18;
f30=20;f31=30;
f1=f10:((f11-f10)/(StageNumber-1)):f11;
f2=f20:((f21-f20)/(StageNumber-1)):f21;
f3=f30:((f31-f30)/(StageNumber-1)):f31;
w1=2*pi*f1;% 频率变化
w2=2*pi*f2;
w3=2*pi*f3;
xi10=0.2;xi11=0.001;
xi20=0.1;xi21=0.003;
xi30=0.1;xi31=0.005;
xi1=xi10:((xi11-xi10)/(StageNumber-1)):xi11;  %阻尼变换
xi2=xi20:((xi21-xi20)/(StageNumber-1)):xi21;
xi3=xi30:((xi31-xi30)/(StageNumber-1)):xi31;
for i=1:StageNumber
    for j =1:StageNumber
        
        flutterSimSys= (exp(-1*xi1(i)*w1(j)*t).*cos(w1(j)*t));%系统

    %     flutterSimSys(i,:)= (exp(-1*xi1(i)*w1(i)*t).*cos(w1(i)*t))+ (exp(-1*xi2(i)*w2(i)*t).*cos(w2(i)*t))+(exp(-1*xi3(i)*w3(i)*t).*cos(w3(i)*t));%系统
    %     flutterSimSys(i,:)= (exp(-1*xi1(i)*w1(i)*t).*cos(w1(i)*t))+ (exp(-1*xi2(i)*w2(i)*t).*cos(w2(i)*t));%系统
        u=0.1.*randn(1,length(t));
        flutterSimConvRes=conv(u,flutterSimSys);
        flutterSimRes=flutterSimConvRes(1:length(flutterSimSys));
        flutterSimSysSingle = flutterSimSys;
        flutterSimResTimeSeries = flutterSimRes;
        dataFileName = ['tur_1order_damp-',num2str(xi1(i)),'_freq-',num2str(f1(j)),'.txt'];
    %     dataFileName = ['tur_2order_damp-',num2str(xi1(i)),'&',num2str(xi2(i)),'_freq-',num2str(f1(i)),'&',num2str(f2(i)),'.txt'];
        homeDataFilePath = 'E:/python/CNNLSTMGenerateFRFht/data';
        dataSubFolder=[ 'tur1order_T-',num2str(T),'s_Fs-',num2str(fs)];
        mainSaveDataFilePath = [homeDataFilePath,'/',dataSubFolder];
        checkedMainSaveDataFilePath =  checkandBuildDir(mainSaveDataFilePath);
        dataFilePathName = [checkedMainSaveDataFilePath,'/',dataFileName] ;
        multiMatrixsave(u,flutterSimResTimeSeries,flutterSimSysSingle,dataFilePathName);
    end
end


function [Y] = noisegen(X,SNR)
    % noisegen add white Gaussian noise to a signal.
    % [Y, NOISE] = NOISEGEN(X,SNR) adds white Gaussian NOISE to X.  The SNR is in dB.
    NOISE=randn(size(X));
    NOISE=NOISE-mean(NOISE);
    signal_power = 1/length(X)*sum(X.*X);
    noise_variance = signal_power / ( 10^(SNR/20) );
    NOISE=sqrt(noise_variance)/std(NOISE)*NOISE;
    Y=X+NOISE;
end

function [Path]=checkandBuildDir(Path)
    if exist(Path,'dir')==0
        mkdir(Path);
        return
    else
        return     
    end 
end

function  multiMatrixsave(matrix1,matrix2,matrix3,fileDataPath)
    if(length(matrix1)==length(matrix2))
        path = fileDataPath;
        fid = fopen(path,'wt');
%         fprintf(fid,'%s\t%s\n','temp','temp2');
        for i=1:length(matrix1)
            fprintf(fid,'%f\t%f\t%f\n',matrix1(i),matrix2(i),matrix3(i));
        end
        fclose(fid);
    else
        disp("口..口 !!!!响应与传递函数时频域长度不一致 !!! 口..口")
        return
    end
    
end