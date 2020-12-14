clc
clear

StageNumber=500;
fs=512;
T=10;
t=0:1/fs:T;
t=t(1:length(t)-1);
n=4;  %ARģ�ͽ���

f10=1;f11=30;
f20=10;f21=18;
f30=20;f31=30;
f1=f10:((f11-f10)/(StageNumber-1)):f11;
f2=f20:((f21-f20)/(StageNumber-1)):f21;
f3=f30:((f31-f30)/(StageNumber-1)):f31;
w1=2*pi*f1;% Ƶ�ʱ仯
w2=2*pi*f2;
w3=2*pi*f3;
xi10=0.05;xi11=0.03;
xi20=0.1;xi21=0.003;
xi30=0.1;xi31=0.005;
xi1=xi10:((xi11-xi10)/(StageNumber-1)):xi11;  %����任
xi2=xi20:((xi21-xi20)/(StageNumber-1)):xi21;
xi3=xi30:((xi31-xi30)/(StageNumber-1)):xi31;
for i=1:StageNumber
    flutterSimSys(i,:)= (exp(-1*xi1(i)*w1(i)*t).*cos(w1(i)*t));%ϵͳ

%     flutterSimSys(i,:)= (exp(-1*xi1(i)*w1(i)*t).*cos(w1(i)*t))+ (exp(-1*xi2(i)*w2(i)*t).*cos(w2(i)*t))+(exp(-1*xi3(i)*w3(i)*t).*cos(w3(i)*t));%ϵͳ
%     flutterSimSys(i,:)= (exp(-1*xi1(i)*w1(i)*t).*cos(w1(i)*t))+ (exp(-1*xi2(i)*w2(i)*t).*cos(w2(i)*t));%ϵͳ
    u=0.1.*randn(1,length(t));
    flutterSimConvRes=conv(u,flutterSimSys(i,:));
    flutterSimRes(i,:)=flutterSimConvRes(1:length(flutterSimSys(i,:)));
    flutterSimSysSingle = flutterSimSys(i,:);
    flutterSimResTimeSeries = flutterSimRes(i,:);
    dataFileName = ['tur_3order_damp-',num2str(xi1(i)),'&',num2str(xi2(i)),'&',num2str(xi3(i)),'_freq-',num2str(f1(i)),'&',num2str(f2(i)),'&',num2str(f3(i)),'.txt'];
%     dataFileName = ['tur_2order_damp-',num2str(xi1(i)),'&',num2str(xi2(i)),'_freq-',num2str(f1(i)),'&',num2str(f2(i)),'.txt'];
    homeDataFilePath = 'E:/python/CNNLSTMGenerateFRFht/data';
    dataSubFolder=[ 'tur_1order_T-',num2str(T),'s_Fs-',num2str(fs)];
    mainSaveDataFilePath = [homeDataFilePath,'/',dataSubFolder];
    checkedMainSaveDataFilePath =  checkandBuildDir(mainSaveDataFilePath);
    dataFilePathName = [checkedMainSaveDataFilePath,'/',dataFileName] ;
    multiMatrixsave(u,flutterSimResTimeSeries,flutterSimSysSingle,dataFilePathName);
    
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
        disp("��..�� !!!!��Ӧ�봫�ݺ���ʱƵ�򳤶Ȳ�һ�� !!! ��..��")
        return
    end
    
end