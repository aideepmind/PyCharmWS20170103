import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import xgboost
from subprocess import check_output
# print(check_output(['1s', '../input']).decode('utf-8'))
sns.set(style = "white", color_codes = True)
warnings.filterwarnings('ignore')

# input
train_data = pd.read_csv('kaggle/housePrices/dataset/train.csv')
test_data = pd.read_csv('kaggle/housePrices/dataset/test.csv')
# data exploration and data processing
def get_missing_cols(data):
    return data.columns[data.isnull().any()].tolist()
# Looking at categorical values
def cat_frequency(data, col):
    return data[col].value_counts()
# Looking at data describe
def cat_describe(data, col):
    data[col].describe()
# Imputing the missing values
def cat_imputation(data, col, val):
    data.loc[data[col].isnull(), col] = val

# 第一轮模型中尽量舍弃一些看起来像异常值的变量，方便分析
# 在后续的精度提高上，再考虑这些变量
total = test_data.isnull().sum().sort_values(ascending=False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)
# 因训练集和测试集上，PoolQC/MiscFeature/Alley/Fence/FireplaceQu/LotFrontage变量的缺失值都超过15%，因此被认为是异常变量，所以暂时删除
# 另外考虑一些与房价相关性小或者跟其他变量相关性大的变量，这些有缺失值的变量暂时也不考虑
# missing data processing
# PoolQC
# MiscFeature
# Alley
# Fence
# FireplaceQu
# LotFrontage
# GarageCond
# GarageType
# GarageYrBlt
# GarageFinish
# GarageQual
# BsmtExposure
# BsmtFinType2
# BsmtFinType1
# BsmtCond
# BsmtQual
# MasVnrArea
# MasVnrType
# MSZoning test
test_data = test_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
test_data = test_data.drop(test_data.loc[test_data['Electrical'].isnull()].index)
test_data.isnull().sum().max()