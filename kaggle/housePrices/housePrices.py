import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

# read data for analysis
def readDataForAnalysis(path):
    return pd.read_csv(path)

# nomalizing
def nomalizing(data):
    m, n = np.shape(data)
    for i in range(m):
        for j in range(n):
            if np.array[i, j] != 0:
                np.array[i, j] = 1
    return np.array


# 1 data exploration
# 1.1 check missing data
def checkMissData(data):
    noneValue = data.isnull() | data.apply(lambda x: str(x).isspace())
    return data[noneValue], data[~noneValue]
# 1.2 outter
# trainData.plot(kind = 'scatter', x = 'MSSubClass', y = 'SalePrice')
# sns.boxplot(y = 'SalePrice', data = trainData)
# sns.stripplot(y = 'SalePrice', data = trainData, jitter = True)

# 2 data preprocessing
# Let's see how many examples we have of each species
# df["Species"].value_counts()



# 3 feature engineering
# 3.1 feature selection
# 3.2 feature encoding
# 4 model selection
# 4.1 model training
# 5 cross validation
# 6 ensemble generation