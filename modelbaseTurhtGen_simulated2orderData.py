'''
 仿真数据中的多阶紊流响应信号通过神经网络模型计算对应的时域传递函数数值解， 读取的仿真信号中包含
    第一列：紊流激励
    第二列：紊流响应
    第三列：系统传递函数的时域波形
 最终将保存文件至指定目录 包括
    紊流响应信号 原文件名.txt
    预测的系统时域传递函数 原文件名_predht.txt
    真实的系统时域传递函数 原文件名_realht.txt
'''
import os
import numpy
import torch
from sklearn.model_selection import train_test_split
import  torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.fftpack import fft
import torch.nn.functional as F
from torch.autograd import Variable
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataAbout():
    def load_real_data(realDataPath,realDataTimeSignalLength):
        realDataFileNameLists = os.listdir(realDataPath);
        realDataTimeSeriesList = []

        for realDataFileName in realDataFileNameLists:
            realDataFilePathName = realDataPath+"\\"+realDataFileName
            realDataFile = open(realDataFilePathName)
            realDataTimeSeries = numpy.loadtxt(realDataFile)

            realDataTimeSeriesMean = numpy.mean(realDataTimeSeries, axis=0)
            realDataTimeSeriesStd = numpy.std(realDataTimeSeries, axis=0)
            realDataTimeSeries = (realDataTimeSeries-realDataTimeSeriesMean)/realDataTimeSeriesStd
            realDataTimeSeriesList.append(realDataTimeSeries)
        return realDataFileNameLists,realDataTimeSeriesList

    def numpyTOFloatTensor(data):
        data = torch.from_numpy(data)
        tensorData = torch.FloatTensor.float(data)
        return tensorData

    def numpyTOLongTensor(data):
        data = torch.from_numpy(data)
        tensorData = torch.LongTensor.long(data)
        return tensorData

    # 数据封装函数
    def data_loader(data_x, data_y):

        train_data = Data.TensorDataset(data_x, data_y)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        return train_loader

    def mergeThreeList(List1, List2, List3):

        List1Size = len(List1)
        List2Size = len(List2)
        List3Size = len(List3)

        arrayList1 = numpy.array(List1)
        arrayList1 = arrayList1.reshape(List1Size, 1)
        arrayList2 = numpy.array(List2)
        arrayList2 = arrayList2.reshape(List2Size, 1)
        arrayList3 = numpy.array(List3)
        arrayList3 = arrayList3.reshape(List3Size, 1)
        mergedArrayList = numpy.concatenate((arrayList1, arrayList2, arrayList3), axis=1)
        return mergedArrayList

    def mergeTwoList(List1, List2):

        List1Size = len(List1)
        List2Size = len(List2)

        arrayList1 = numpy.array(List1)
        arrayList1 = arrayList1.reshape(List1Size, 1)
        arrayList2 = numpy.array(List2)
        arrayList2 = arrayList2.reshape(List2Size, 1)
        mergedArrayList = numpy.concatenate((arrayList1, arrayList2), axis=1)
        return mergedArrayList

    def mkSaveModelResultdir(path):
        folder = os.path.exists(path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
            return path
        else:
            return path

# class flutterFilterNet(nn.Module):
#     def __init__(self):
#         super(flutterFilterNet, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=2, kernel_size=[3], stride=1, padding=1),
#             nn.Conv1d(in_channels=2, out_channels=4, kernel_size=[3], stride=1, padding=1),
#             nn.AvgPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(in_channels=4, out_channels=6, kernel_size=[3], stride=1, padding=1),
#             nn.Conv1d(in_channels=6, out_channels=8, kernel_size=[3], stride=1, padding=1),
#             nn.AvgPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(in_channels=8, out_channels=10, kernel_size=[3], stride=1, padding=1),
#             nn.Conv1d(in_channels=10, out_channels=12, kernel_size=[3], stride=1, padding=1),
#             nn.AvgPool1d(kernel_size=2, stride=2),
#         )
#
#         self.middle = nn.Sequential(
#             nn.Conv1d(12, 12, kernel_size=[3], stride=1, padding=1),  # //双斜杠取整
#             nn.BatchNorm1d(12),
#             nn.LeakyReLU(0.1)
#         )
#
#         self.decoder = nn.Sequential(
#             nn.Conv1d(in_channels=12, out_channels=10, kernel_size=[3], stride=1, padding=1),
#             nn.Conv1d(in_channels=10, out_channels=8, kernel_size=[3], stride=1, padding=1),
#             nn.Upsample(scale_factor=2),
#             nn.Conv1d(in_channels=8, out_channels=6, kernel_size=[3], stride=1, padding=1),
#             nn.Conv1d(in_channels=6, out_channels=4, kernel_size=[3], stride=1, padding=1),
#             nn.Upsample(scale_factor=2),
#             nn.Conv1d(in_channels=4, out_channels=2, kernel_size=[3], stride=1, padding=1),
#             nn.Conv1d(in_channels=2, out_channels=1, kernel_size=[3], stride=1, padding=1),
#             nn.Upsample(scale_factor=2),
#         )
#         self.output = nn.Sequential(
#             nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1, stride=1,padding=0),
#             nn.LeakyReLU(0.1)
#         )
#
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.middle(x)
#         # print(x.size())
#         x = self.decoder(x)
#         x = self.output(x)
#
#         return x
######################################
# class Unet(nn.Module):
#     def __init__(self):
#         super(Unet,self).__init__()
#         print('unet')
#         nlayers = LayerNumber
#         nefilters=NumberofFeatureChannel # 每次迭代时特征增加数量###
#         self.num_layers = nlayers
#         self.nefilters = nefilters
#         filter_size = 11
#         merge_filter_size = 11
#         self.encoder = nn.ModuleList() # 定义一个空的modulelist命名为encoder#####
#         self.decoder = nn.ModuleList()
#         self.ebatch = nn.ModuleList()
#         self.dbatch = nn.ModuleList()
#         echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
#         echannelout = [(i + 1) * nefilters for i in range(nlayers)]
#         dchannelout = echannelout[::-1]
#         dchannelin = [dchannelout[0]*2]+[(i) * nefilters + (i - 1) * nefilters for i in range(nlayers,1,-1)]
#
#         for i in range(self.num_layers):
#             self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=filter_size//2))
#             self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
#             self.ebatch.append(nn.BatchNorm1d(echannelout[i]))  #  moduleList 的append对象是添加一个层到module中######
#             self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))
#
#         self.middle = nn.Sequential(
#             nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2),
#             nn.BatchNorm1d(echannelout[-1]),
#             nn.LeakyReLU(0.1)
#         )
#         self.out = nn.Sequential(
#             nn.Conv1d(nefilters + 1, 1, 1),
#             nn.Tanh()
#         )
#     def forward(self,x):
#
#         encoder = list()
#         input = x
#         # print(x.shape)
#
#
#         for i in range(self.num_layers):
#             x = self.encoder[i](x)
#             x = self.ebatch[i](x)
#             x = F.leaky_relu(x,0.1)
#             encoder.append(x)
#             x = x[:,:,::2]
#             # print(x.shape)
#
#         x = self.middle(x)
#
#         for i in range(self.num_layers):
#             x = F.upsample(x,scale_factor=2,mode='linear')
#             # print('deconder_dim:',x.shape,
#             #       'encoder_dim:',encoder[self.num_layers - i - 1].shape)####
#             x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对#######
#             x = self.decoder[i](x)
#             x = self.dbatch[i](x)
#             x = F.leaky_relu(x,0.1)
#         x = torch.cat([x,input],dim=1)
#
#         x = self.out(x)
#         return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        print('unet')
        nlayers = LayerNumber
        nefilters=NumberofFeatureChannel # 每次迭代时特征增加数量
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 11
        merge_filter_size = 11
        self.encoder = nn.ModuleList() # 定义一个空的modulelist命名为encoder
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0]*2]+[(i) * nefilters + (i - 1) * nefilters for i in range(nlayers,1,-1)]


        for i in range(self.num_layers):
            self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=filter_size//2))
            self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))  #  moduleList 的append对象是添加一个层到module中
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))

        self.middle = nn.Sequential(
            nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2), # //双斜杠取整
            nn.BatchNorm1d(echannelout[-1]),
            # nn.LeakyReLU(0.1)
            nn.Tanh()
        )
        self.out = nn.Sequential(
            nn.Conv1d(nefilters + 1, 1, 1),
            nn.Tanh()
        )
        self.smooth = nn.Sequential(
            nn.Conv1d(1,1,1)
        )
    def forward(self,x):
        encoder = list()
        input = x


        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            encoder.append(x)
            x = x[:,:,::2]
            # print(x.shape)
        x = self.middle(x)
        # print(x.shape)
        for i in range(self.num_layers):
            x = F.upsample(x,scale_factor=2,mode='linear')
            # print('deocder_dim：',x.shape,
            #       '\tencode_dim:',encoder[self.num_layers - i - 1].shape)
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,0*input],dim=1)

        x = self.out(x)
        x = self.smooth(x)

        return x

def mkSaveModelResultdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        return path
    else:
        return path



EPOCH = 200000

LR = 0.00005
Fs=512
T=20

LayerNumber = 10
NumberofFeatureChannel = 2



resultSavePath = 'E:\\python\\CNNLSTMGenerateFRFht\\result\\remote_multiOrderturbhtGen_tur_T-20s_Fs-512_E-100005_LR-0.0001_LayerNum-10_filterNum-2'
resultSaveModelPath = resultSavePath+'\\model_result'
modelName  = 'model_state_dict_E20_Fs-512_LayerNum-10_filterNum-2_Epoch-100005_LR-0.0001.pth'
resultSaveModelAbsoluedPath = resultSaveModelPath+'\\'+modelName
print(resultSaveModelAbsoluedPath)



'''
                            口..口
'''
'''！！！！查看模型网络与loda结构是否一致！！！！'''
'''！！！关键注意pkl编码有时候复制过来容易出错 导致网络结构和load的模型参数不一致 容易出现问题 ！！！'''
'''
                            口..口
'''
model = Unet();
model.load_state_dict({k.replace('module.',''):v for k, v in torch.load(resultSaveModelAbsoluedPath).items()})
# model.to(device)



simulated2orderDataDir = 'E:\\python\\CNNLSTMGenerateFRFht\\data\\tur1order_T-20s_Fs-512'
simulated2orderFileDir = simulated2orderDataDir.split('\\')[-1]
simulated2orderSavedFileDIr = resultSavePath+'\\simulatedDataVal\\'+simulated2orderFileDir
simulated2orderSavedFileDIr =mkSaveModelResultdir(simulated2orderSavedFileDIr)

simulated2orderDataNameList = os.listdir(simulated2orderDataDir)
for i in range(len(simulated2orderDataNameList)):
    if i%20==0:
        simulated2orderDataFileName = simulated2orderDataNameList[i]
        simulated2orderDataFilePath = simulated2orderDataDir+'\\'+simulated2orderDataFileName
        dataFile = open(simulated2orderDataFilePath,'r')
        data = numpy.loadtxt(dataFile)
        turResSignal = data[:,1]
        turSysSignal = data[:,2]

        turResSignal = numpy.reshape(turResSignal, [1, 1, len(data)])

        turResSignal = (turResSignal - numpy.mean(turResSignal)) / numpy.std(turResSignal)
        turResSignal = dataAbout.numpyTOFloatTensor(turResSignal)
        predturSysht = model(turResSignal)

        f=numpy.linspace(0,Fs,T*Fs)
        simulated2orderTurResFilePath = simulated2orderSavedFileDIr+'\\'+simulated2orderDataFileName
        numpy.savetxt(simulated2orderTurResFilePath,turResSignal.flatten())
        simulated2orderTurSysFileName = simulated2orderDataFileName[:len(simulated2orderDataFileName)-4]+'_realht.txt'
        simulated2orderTurSysFilePath = simulated2orderSavedFileDIr+'\\'+ simulated2orderTurSysFileName
        numpy.savetxt(simulated2orderTurSysFilePath,turSysSignal)

        predhtSimulated2orderDataFileName =  simulated2orderDataFileName[:len(simulated2orderDataFileName)-4]+'_predht.txt'
        predhtSimulated2orderDataFilePath =  simulated2orderSavedFileDIr+"\\"+predhtSimulated2orderDataFileName
        numpy.savetxt(predhtSimulated2orderDataFilePath,predturSysht.detach().numpy().flatten())

    # plt.figure()
    # plt.plot(data)
    # plt.figure()
    # plt.plot(predht.detach().numpy().flatten())
    # plt.show()
    #
    #
    # plt.figure()
    # plt.plot(f,abs(fft(data)))
    # plt.figure()
    # plt.plot(f,abs(fft(predht.detach().numpy().flatten())))
    # plt.show()

