%%
% 海鸥300的紊流响应数据按照column拆分成单独的文件 便于后续验证结果
%
%%
saveHomePath = "E:\python\CNNLSTMGenerateFRFht\data\realData1";

data = F1VIBturbluence2044Sec;
[row, col] = size(wl19090320200614F1VIB101234);
for i=1:col
    fileName="F1VIBturbluence2044Sec-"+num2str(i)+".txt" ;
    fileSavePath = saveHomePath+"\"+fileName;
    file=fopen(fileSavePath,'wt');
    for j =1:row       
       fprintf(file,'%f\n',data(j,i));      
    end    
    fclose(file);
end
