'''
用于C919数据的按列分割 读取原始txt第一行通道名称作为一个list 之后数据按照行列添加到allSignallist中之后全部转到ndarray里面 之后按照数据的列保存

'''
import numpy

path = 'E:\python\CNNLSTMGenerateFRFht\data\C919turData'
fileName = 'C919-10103-YL-200406-F-01-21-FLUTTER-27085-0.657-261-TUR-EXC-512.txt'
filePath = path+'\\'+fileName
file = open(filePath,'r')
channelName = file.readline()
channelNameList = channelName.split()
allSignallist = list()
allData = file.readlines()
for dataperLine in allData:
    dataperLineList =  dataperLine.split()
    allSignallist.append(dataperLineList[1:])
allSignal=numpy.array(allSignallist)
print(allSignal.shape)
for i in range(1,len(channelNameList)):
    perChannelFileName = fileName[:len(fileName)-4]+'-'+channelNameList[i]+'.txt'
    perChannelSignal = allSignal[:,i-1].flatten()
    print(perChannelSignal)
    print(perChannelFileName)
    perChannelFilePath =path+'\\'+ perChannelFileName
    print(perChannelFilePath)
    numpy.savetxt(perChannelFilePath,perChannelSignal,fmt = '%s')

