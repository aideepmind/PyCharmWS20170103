from numpy import *
import csv
import operator
import numpy as np

# 读文件
# with open('E:\kaggle\Digit Recognizer\sample_submission.csv', 'r') as sfile:
#     lines = csv.reader(sfile)
#     for line in lines:
#         print(line)

# 将字符串转化为整型
def str2int(array):
    array = np.mat(array)# 将array（可以是多维）转化为二维矩阵
    m, n = shape(array)# 获取行列数
    newArray = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray

# 标准化（归一化）
def nomalizing(array):
    m, n = shape(array)
    for i in range(m):
        for j in range(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array

# 加载训练数据
def loadTrainData():
    l = []
    with open('E:\kaggle/Digit Recognizer/train.csv', 'r') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = str2int(l)
    m = shape(l)
    trainData, sampleData = randomSampling(l, int(m[0] * 0.2))
    trainData = array(trainData)
    trainLabel = trainData[:, 0]
    trainData = trainData[:, 1:]
    sampleData = array(sampleData)
    sampleLabel = sampleData[:, 0]
    sampleData = sampleData[:, 1:]
    return nomalizing(trainData), trainLabel, nomalizing(sampleData), sampleLabel

# 加载测试数据
def loadTestData():
    l = []
    with open('E:\kaggle/Digit Recognizer/test.csv', 'r') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = array(l)
    return nomalizing(str2int(l))

# KNN分类算法 使用欧氏距离
def knnClassify(inX, dataSet, labels, k):
    inX = np.mat(inX)
    dataSet = np.mat(dataSet)
    labels = mat(labels)
    dataSetSize = dataSet.shape[0]                      # shape[0]得出dataSet的行数，即样本个数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile(A,(m,n))将数组A作为元素构造m行n列的数组
    sqDiffMat = array(diffMat) ** 2
    sqDistances = sqDiffMat.sum(axis=1)                 # array.sum(axis=1)按行累加，axis=0为按列累加
    distances = sqDistances ** 0.5
    sortedDistIndicies= distances.argsort()             # array.argsort()，得到每个元素的排序序号
    classCount = {}                                     #sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i], 0]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        # get(key,x)从字典中获取key对应的value，没有key的话返回0
    return sortedClassCount[0][0]

# 无放回随机抽样
def randomSampling(dataMat, number):
    dataMat = array(dataMat)
    try:
        #slice = random.sample(dataMat, number)
        sample = []
        for i in range(number):
            index = random.randint(0, len(dataMat)) #包含low，但不包含high
            sample.append(dataMat[index])
            dataMat = delete(dataMat, index, 0)
        return dataMat, sample
    except e:
        print(e)

# 训练
def train():
    trainData, trainLabel, sampleData, sampleLabel = loadTrainData()
    print("trainData=", trainData)
    print("trainLabel=", trainLabel)
    print("sampleData=", sampleData)
    print("sampleLabel=", sampleLabel)

    # m, n = shape(sampleData)
    # rightNumber = 0
    # errorNumber = 0
    # for i in range(m):
    #     classifyResult = knnClassify(sampleData[i], trainData, trainLabel.transpose(), 10)
    #     if classifyResult != sampleLabel[m]:
    #         errorNumber += 1
    #     else:
    #         rightNumber += 1
    # print("rightNumber=", rightNumber)
    # print("errorNumber=", errorNumber)
    # print("rightRate=", rightNumber / m)

# 预测
def predict():
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    m, n = shape(testData)
    for i in range(100):
        classifyResult = knnClassify(testData[i], trainData, trainLabel.transpose(), 10)
        print(classifyResult)

if __name__ == '__main__':
    train()