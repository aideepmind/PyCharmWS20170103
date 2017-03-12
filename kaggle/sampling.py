from numpy import *

# 抽样工具
class Sampling:

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

    # 有放回随机抽样
    def repetitionRandomSampling(dataMat, number):
        sample = []
        for i in range(number):
            sample.append(dataMat[random.randint(0, len(dataMat))])
        return dataMat, sample

    # 系统随机抽样
    def SystematicSampling(dataMat, number):
        length = len(dataMat)
        k = length / number
        sample = []
        i = 0
        if k > 0:
            while len(sample) != number:
                sample.append(dataMat[0 + i * k])
                i += 1
            return sample
        else:
            return dataMat, Sampling.randomSampling(dataMat, number)

    if __name__ == '__main__':
        randomData = random.random((4, 2))
        print(randomData)
        dataMat, sample = randomSampling(randomData, 2)
        print("dataMat=" , dataMat)
        print("sample=" , sample)