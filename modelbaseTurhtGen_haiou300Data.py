'''
对实测的海鸥300的紊流响应信号进行基于WaveNet的系统时域传递函数的计算，最终结果保存为 xxx-predht.txt文件
然后用matlab计算fft 的幅值
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



EPOCH = 200000

LR = 0.00005
Fs=512
T=20

LayerNumber = 10
NumberofFeatureChannel = 2


resultSaveHomePath = "E:\\python\\CNNLSTMGenerateFRFht\\result"

resultSavePath = resultSaveHomePath\
                 +'\\remote_turbhtGen_E-'+str(EPOCH)+"_LR-"+str(LR)+'_LayerNum-'+str(LayerNumber)+'_filterNum-'+str(NumberofFeatureChannel)

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



realDataT = 22; realDataSample = 200;
realDataT = 20; realDataSample = 128;
realDataTimeLength = realDataT*realDataSample;

# realDataPath = 'E:\\python\\DNNpreporcessinMethod\\data\\Turbo\\realdata\\all_real_data_zip\\all_real_unflutter_last_step_data'
# realDataPath = 'E:\\python\\DNNpreporcessinMethod\\data\\SWP\\3order_DataTimeSeries_20secondData_SNR-0'
# realDataPath = 'E:\\python\\DNNpreporcessinMethod\\data\\C919flutterSwp\\20191031'
# realDataPath = 'E:\\python\\DNNpreporcessinMethod\\data\\C919flutterSwp\\20191113'

realDataDirPath = 'E:\\python\\CNNLSTMGenerateFRFht\\result\\remote_multiOrderturbhtGen_tur_T-20s_Fs-512_E-100005_LR-0.0001_LayerNum-10_filterNum-2\\realData_tur_haiou300'

# realDataDirPath = 'E:\\python\\CNNLSTMGenerateFRFht\\data\\realData1'
for i in range(25):
    realDataFileName = 'wl19090320200614F1VIB101234-'+str(i+1)+'.txt'
    realDataFilePath = realDataDirPath+'\\'+realDataFileName

    file=open(realDataFilePath,'r')
    data=numpy.loadtxt(file)
    # if len(data)%2!=0:
    data = data[:T*Fs]
    print(data.shape)

    signal = numpy.reshape(data, [1, 1, len(data)])

    signal = (signal - numpy.mean(signal)) / numpy.std(signal)
    signal = dataAbout.numpyTOFloatTensor(signal)
    predht = model(signal)

    f=numpy.linspace(0,Fs,T*Fs)
    predhtFileName =  realDataFileName[:len(realDataFileName)-4]+'-predht.txt'
    predhtFilePath =  realDataDirPath+"\\"+predhtFileName
    numpy.savetxt(predhtFilePath,predht.detach().numpy().flatten())
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

