import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
import warnings
import xgboost
from subprocess import check_output
# print(check_output(['1s', '../input']).decode('utf-8'))
sns.set(style = "white", color_codes = True)
warnings.filterwarnings('ignore')

# input
train_data = pd.read_csv('kaggle/housePrices/dataset/train.csv')
test_data = pd.read_csv('kaggle/housePrices/dataset/test.csv')
all_data = pd.concat((train_data[test_data.columns], test_data))

# data exploration and data processing
def get_missing_cols(data, col):
    # all_data.isnull().sum().order(ascending = False)
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
def cat_null_sum(col):
    print(all_data[col].isnull().sum())

# 区分直方图与条形图：
# 条形图是用条形的长度表示各类别频数的多少，其宽度（表示类别）则是固定的；
# 直方图是用面积表示各组频数的多少，矩形的高度表示每一组的频数或频率，宽度则表示各组的组距，因此其高度与宽度均有意义。
# 由于分组数据具有连续性，直方图的各矩形通常是连续排列，而条形图则是分开排列。
# 条形图主要用于展示分类数据，而直方图则主要用于展示数据型数据
# bar
def bar(column):
    plt.figure()
    if type(column) == str:
        column = train_data[column]
    vc = column.value_counts().sort_values(ascending = False)
    x_pos = np.arange(len(vc))
    plt.bar(x_pos, vc)
    plt.xticks(x_pos, vc.index)
    plt.xlabel(column.name)
    plt.ylabel('frequency')

# barh
def barh(column):
    plt.figure()
    if type(column) == str:
        column = train_data[column]
    vc = column.value_counts().sort_values(ascending = False)
    y_pos = np.arange(len(vc))
    plt.barh(y_pos, vc)
    plt.yticks(y_pos, vc.index)
    plt.ylabel(column.name)
    plt.xlabel('frequency')

# hist
def hist(column):
    plt.figure()
    if type(column) == str:
        plt.hist(all_data[column])
    else:
        plt.hist(column)

# displot
def displot(column):
    plt.figure()
    if type(column) == str:
        sns.distplot(all_data[column], fit=norm)
    else:
        sns.distplot(column, fit=norm)

# boxplot
def boxplot(column):
    plt.figure()
    if type(column) == str:
        sns.boxplot(all_data[column])
    else:
        sns.boxplot(column)

# violinplot
def violinplot(column):
    plt.figure()
    if type(column) == str:
        sns.violinplot(all_data[column])
    else:
        sns.violinplot(column)

# scatter
def scatter(column1, column2):
    plt.figure()
    if type(column1) == str:
        if column1 not in all_data.columns or column2 not in all_data.columns:
            plt.scatter(train_data[column1], train_data[column2])
        else:
            plt.scatter(all_data[column1], all_data[column2])
    else:
        plt.scatter(column1, column2)

# probplot
def probplot(column):
    plt.figure()
    if type(column) == str:
        stats.probplot(all_data[column], plot=plt)
    else:
        stats.probplot(column, plot=plt)

# pairplot
def pairplot(columns):
    plt.figure()
    sns.pairplot(columns)

# heatmap
def heatmap(columns):
    plt.figure()
    corrmat = train_data.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat, vmax=.8, square=True);
    k = 10  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train_data[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(columns,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=cols.values,
                xticklabels=cols.values)

# all variables
# Id, MSSubClass, MSZoning, LotFrontage, LotArea, Street,
# Alley, LotShape, LandContour, Utilities, LotConfig,
# LandSlope, Neighborhood, Condition1, Condition2, BldgType,
# HouseStyle, OverallQual, OverallCond, YearBuilt, YearRemodAdd,
# RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType,
# MasVnrArea, ExterQual, ExterCond, Foundation, BsmtQual,
# BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinSF1,
# BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, Heating,
# HeatingQC, CentralAir, Electrical, 1stFlrSF, 2ndFlrSF,
# LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath,
# HalfBath, BedroomAbvGr, KitchenAbvGr, KitchenQual,
# TotRmsAbvGrd, Functional, Fireplaces, FireplaceQu, GarageType,
# GarageYrBlt, GarageFinish, GarageCars, GarageArea, GarageQual,
# GarageCond, PavedDrive, WoodDeckSF, OpenPorchSF,
# EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, PoolQC,
# Fence, MiscFeature, MiscVal, MoSold, YrSold, SaleType,
# SaleCondition, SalePrice

# missing data processing

# MSSubClass None
# boxplot('MSSubClass')
# displot('MSSubClass')
# only train data have outter(id <1455)，so delete it or instead of mean
# print(len(all_data[all_data['MSSubClass'] > 120]['MSSubClass']))
# all_data[all_data['MSSubClass'] > 120]['MSSubClass'] = all_data['MSSubClass'].mean() # wrong code
# all_data['MSSubClass'].mean() #57.1377183967112,so use 60
# all_data.loc[all_data['MSSubClass'] > 120, 'MSSubClass'] = 60
# 值得记住，因为MSSubClass原始的数字没有意义，所以需要转换为字符类型，但考虑到种类太多，所以又可以合并一些种类，依照
# 的SalePrice均值来分类
# print(train_data[['MSSubClass', 'SalePrice']].groupby('MSSubClass').mean().sort_values(by = 'SalePrice'))
all_data = all_data.replace({"MSSubClass": {30: "A", 180: "A", 45: "A",
                                            190: "B", 90: "B", 160: "B",
                                            50: "C", 85: "C",
                                            40: "D", 70: "D", 80: "D",
                                            20: "E", 75: "E",
                                            120: "F", 60: "F",
                                            150: "E"}})# 考虑到没出现在train中，且在test中只出现一次，故取频率最高的20


# MSZoning test，使用了交叉表，里面有三个比较模拟两可的值，需要在模型优化时再重新设置一下，看哪一个妥当
# print(all_data['MSZoning'].isnull().sum())
# print(all_data[all_data['MSZoning'].isnull() == True])
# print(all_data[['MSZoning', 'MSSubClass']][all_data['MSZoning'].isnull() == True])
# print(pd.crosstab(all_data.MSSubClass, all_data.MSZoning))
all_data.loc[all_data['Id'] == 1916, 'MSZoning'] = 'RM'# RL
all_data.loc[all_data['Id'] == 2217, 'MSZoning'] = 'RL'
all_data.loc[all_data['Id'] == 2905, 'MSZoning'] = 'RL'
all_data.loc[all_data['Id'] == 2251, 'MSZoning'] = 'RM'# RL


# LotFrontage train & test，fill value 采用LotArea平方根，感觉不妥当，还得回头修改
# print(all_data['LotFrontage'].isnull().sum())
# plt.scatter(train_data['LotFrontage'], train_data['SalePrice'])
# train_data['LotFrontage'].corr(np.sqrt(train_data['LotArea']))
# train_data.LotFrontage[train_data['LotFrontage'].isnull()] = np.sqrt(train_data.LotArea[train_data['LotFrontage'].isnull()])
# test_data.LotFrontage[test_data['LotFrontage'].isnull()] = np.sqrt(test_data.LotArea[test_data['LotFrontage'].isnull()])
# 486个空值，不过，重要性排名竟然得第一，怎麼搞的？暂时delete，事实证明drop比保留更好
all_data = all_data.drop(['LotFrontage'], axis=1)
# all_data.LotFrontage[all_data['LotFrontage'].isnull()] = np.sqrt(all_data.LotArea[all_data['LotFrontage'].isnull()])


# Alley train & test
# print(all_data['Alley'].isnull().sum())
# print(cat_frequency (test_data, 'Alley'))
# print(cat_frequency (train_data, 'Alley'))
# Alley这个特征有太多的nans,这里填充None，也可以直接删除，不使用。后面在根据特征的重要性选择特征是，也可以舍去
# cat_imputation(test_data, 'Alley', 'None')
# cat_imputation(train_data, 'Alley', 'None')
all_data = all_data.drop(['Alley'], axis=1)


# Utilities test
# 并且这个column中值得分布极为不均匀，drop
# print(all_data['Utilities'].isnull().sum())
# print(cat_frequency (test_data, 'Utilities'))
# print(cat_frequency (train_data, 'Utilities'))
# print(test_data.loc[test_data['Utilities'].isnull() == True])
# test_data = test_data.drop(['Utilities'], axis=1)
# train_data = train_data.drop(['Utilities'], axis=1)
all_data = all_data.drop(['Utilities'], axis=1)

# Exterior1st & Exterior2nd test，这里采用交叉表，但是结果并不理想，因为几个值都很接近暂时采取频率最高的
# 检查Exterior1st 和 Exterior2nd 是否存在缺失值共现的情况
# print(all_data['Exterior1st'].isnull().sum())
# print(all_data['Exterior2nd'].isnull().sum())
# cat_frequency (test_data, 'Exterior1st')
# cat_frequency (train_data, 'Exterior1st')
# print(all_data[['Exterior1st', 'Exterior2nd', 'ExterQual']][all_data['Exterior1st'].isnull() == True])
# print(pd.crosstab(all_data.Exterior1st, all_data.ExterQual))
all_data.loc[all_data['Exterior1st'].isnull(), 'Exterior1st'] = 'HdBoard'# HdBoard/MetalSd/VinylSd/Wd Sdng/Plywood
all_data.loc[all_data['Exterior2nd'].isnull(), 'Exterior2nd'] = 'HdBoard'# HdBoard/MetalSd/VinylSd/Wd Sdng/Plywood

# MasVnrType & MasVnrArea train & test
# print(all_data['MasVnrType'].isnull().sum())
# print(all_data['MasVnrArea'].isnull().sum())
# print(test_data[['MasVnrType', 'MasVnrArea']][test_data['MasVnrType'].isnull() == True])
# print(train_data[['MasVnrType', 'MasVnrArea']][train_data['MasVnrType'].isnull() == True])
# So the missing values for the "MasVnr..." Variables are in the same rows.
# cat_frequency(test_data, 'MasVnrType')
# cat_frequency(train_data, 'MasVnrType')
# cat_imputation(test_data, 'MasVnrType', 'None')
# cat_imputation(train_data, 'MasVnrType', 'None')
# cat_imputation(test_data, 'MasVnrArea', 0.0)
# cat_imputation(train_data, 'MasVnrArea', 0.0)
# MasVnrType有24个空，而MasVnrArea有23个空，有一个不一致：MasVnrArea = 198.0
# all_data.loc[all_data['MasVnrArea'] == 198.0, 'MasVnrType']
all_data.loc[all_data['Id'] == 2611, 'MasVnrType'] = 'BrkFace'    # Stone
cat_imputation(all_data, 'MasVnrType', 'None')
cat_imputation(all_data, 'MasVnrArea', 0.0)

# basement train & test
# train
basement_cols = ['Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2']
# print(train_data['BsmtQual'].isnull().sum())
# print(train_data[basement_cols][train_data['BsmtExposure'].isnull() == True])
# print(pd.crosstab(all_data.BsmtQual, all_data.BsmtExposure))
# cat_frequency(all_data, 'BsmtExposure')
# BsmtExposure和BsmtFinType2一共38个空值并有2个不一致，其他37个空值，且一致
# d1 = all_data[all_data['BsmtFinSF1'] > 1000]
# d1 = all_data[all_data['BsmtFinSF1'] < 1200]
# cat_frequency(d1, 'BsmtExposure')
all_data.loc[all_data['Id'] == 949, 'BsmtExposure'] = 'No'
# print(train_data[basement_cols][train_data['BsmtFinType2'].isnull() == True])
# d1 = all_data[all_data['BsmtFinSF2'] > 450]
# d1 = d1[d1['BsmtFinSF2'] < 500]
# cat_frequency(d1, 'BsmtFinType2')
all_data.loc[all_data['Id'] == 333, 'BsmtFinType2'] = 'Rec' # LwQ/BLQ/LwQ


# test
basement_cols = ['Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
# print(test_data['BsmtQual'].isnull().sum())
# 其中,有两行有BsmtQual为NaN,该两行的其他列都有值
# print(test_data[basement_cols][test_data['BsmtQual'].isnull() == True])
# print(pd.crosstab(all_data.BsmtQual, all_data.BsmtFinType1))
all_data.loc[all_data['Id'] == 2218, 'BsmtCond'] = 'TA' # Gd
all_data.loc[all_data['Id'] == 2219, 'BsmtCond'] = 'TA' # Gd
# 其中,有三行只有BsmtCond为NaN,该三行的其他列都有值
# print(test_data[basement_cols][test_data['BsmtCond'].isnull() == True])
# print(pd.crosstab(all_data.BsmtCond, all_data.BsmtQual))
# d1 = all_data[all_data['BsmtFinSF1'] > 1000]
# d1 = d1[d1['BsmtFinSF1'] < 1100]
# cat_frequency(d1, 'BsmtCond')
all_data.loc[all_data['Id'] == 2041, 'BsmtCond'] = 'TA'
all_data.loc[all_data['Id'] == 2186, 'BsmtCond'] = 'TA'
all_data.loc[all_data['Id'] == 2525, 'BsmtCond'] = 'TA'
# 其中,有一行只有BsmtExposure为NaN,该一行的其他列都有值
# print(test_data[basement_cols][test_data['BsmtExposure'].isnull() == True])
# print(pd.crosstab(all_data.BsmtExposure, all_data.BsmtQual))
all_data.loc[all_data['Id'] == 2349, 'BsmtExposure'] = 'No'
# print(test_data[basement_cols][test_data['BsmtFinSF1'].isnull() == True])
cat_imputation(all_data, 'BsmtFinSF1', '0')
# print(test_data[basement_cols][test_data['BsmtFinSF2'].isnull() == True])
cat_imputation(all_data, 'BsmtFinSF2', '0')
# print(test_data[basement_cols][test_data['BsmtUnfSF'].isnull() == True])
cat_imputation(all_data, 'BsmtUnfSF', '0')
# print(test_data[basement_cols][test_data['TotalBsmtSF'].isnull() == True])
cat_imputation(all_data, 'TotalBsmtSF', '0')
# print(test_data[basement_cols][test_data['BsmtFullBath'].isnull() == True])
cat_imputation(all_data, 'BsmtFullBath', '0')
# print(test_data[basement_cols][test_data['BsmtHalfBath'].isnull() == True])
cat_imputation(all_data, 'BsmtHalfBath', '0')
# 除了上述之外, 其他行的NaN都是一样的
for cols in basement_cols:
    if all_data[cols].dtype == np.object:
        cat_imputation(all_data, cols, 'None')
    elif cols != 'Id':
        cat_imputation(all_data, cols, 0.0)


# 对于BsmtQual这个特征，取值有 Ex，Gd，TA，Fa，Po. 从数据的说明中可以看出，这依次是优秀，好，次好，一般，差几个等级，这具有明显的可比较性，可以使用map编码
# 除了BsmtQual这个特征以外，其他几个特征，比如BsmtCond，HeatingQC等都可以尝试类似的编码方式。避免使用one-hot编码
# all_data = all_data.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})


# KitchenQual test，这里使用交叉表，得出的结论模棱两可，可以在TA和Gd中选择，后续优化时，需考虑
# cat_frequency(all_data, 'KitchenQual')
# print(all_data[['KitchenQual', 'KitchenAbvGr']][all_data['KitchenQual'].isnull() == True])
# print(pd.crosstab(all_data.KitchenQual, all_data.KitchenAbvGr))
cat_imputation(all_data, 'KitchenQual', 'TA') # Gd


# Functional test
# 填充一个最常见的值
# cat_null_sum('Functional')
# cat_frequency(all_data, 'Functional')
cat_imputation(all_data, 'Functional', 'Typ')


# FireplaceQu train & test
# cat_null_sum('Fireplaces')
# cat_frequency(all_data, 'FireplaceQu')
# print(pd.crosstab(all_data.Fireplaces, all_data.FireplaceQu))
cat_imputation(all_data, 'FireplaceQu', 'None')


# Garage train & test
# train
garage_cols = ['Id', 'GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
all_data[garage_cols][all_data['GarageFinish'].isnull() == True].to_csv("kaggle/housePrices/temp/GarageFinish.csv", index=False)
# cat_null_sum('GarageType') 157
# cat_null_sum('GarageQual') 159
# print(pd.crosstab(all_data.GarageQual, all_data.GarageCond))
all_data.loc[all_data['Id'] == 2127, 'GarageQual'] = 'TA'
all_data.loc[all_data['Id'] == 2127, 'GarageCond'] = 'TA' # max
# 算出YearBuilt和GarageYrBlt的关系
YearBuilt_GarageYrBlt = all_data[['YearBuilt', 'GarageYrBlt']][all_data['GarageYrBlt'].isnull() == False]
YearBuilt_GarageYrBlt_sum = np.sum(YearBuilt_GarageYrBlt)
YearBuilt_GarageYrBlt_subtract = np.round((YearBuilt_GarageYrBlt_sum[0] - YearBuilt_GarageYrBlt_sum[1]) / len(YearBuilt_GarageYrBlt))
all_data.loc[all_data['Id'] == 2127, 'GarageYrBlt'] = all_data.loc[all_data['Id'] == 2127, 'YearBuilt'] -  YearBuilt_GarageYrBlt_subtract
all_data.loc[all_data['Id'] == 2127, 'GarageFinish'] = 'Fin' # 因为是1915年，所以是Finished
# 因为Id=2577只有GarageType为Detchd，其他列没有值，所干脆把GarageType置为空
all_data.loc[all_data['Id'] == 2577, 'GarageType'] = 'None'
# cat_null_sum('GarageCond') 159
# 同GarageQual
# cat_null_sum('GarageYrBlt') 159
# 同GarageQual
# cat_null_sum('GarageFinish') 159
# 同GarageQual
# cat_null_sum('GarageCars') 1
# print(all_data[garage_cols][all_data['GarageCars'].isnull() == True])
# 同GarageQual 2577
# cat_null_sum('GarageArea') 1
# print(all_data[garage_cols][all_data['GarageArea'].isnull() == True])
# 同GarageQual 2577
for cols in garage_cols:
    if all_data[cols].dtype == np.object:
        cat_imputation(all_data, cols, 'None')
    elif cols != 'Id':
        cat_imputation(all_data, cols, 0)


# PoolQC & PoolArea train & test
# cat_null_sum('PoolQC')
# print(all_data['PoolQC'][all_data['PoolQC'].isnull() == False])
# 只有10个大于0的值，drop
all_data = all_data.drop(['PoolQC'], axis=1)
# cat_null_sum('PoolArea')
# 只有13个大于0的值，drop
all_data = all_data.drop(['PoolArea'], axis=1)


# Fence train & test
# cat_null_sum('Fence')
# cat_frequency(all_data, 'Fence')
cat_imputation(all_data, 'Fence', 'None')


# MiscFeature train & test
# cat_null_sum('MiscFeature')
# cat_frequency(all_data, 'MiscFeature')
cat_imputation(all_data, 'MiscFeature', 'None')


# SaleType test
# cat_null_sum('SaleType')
# cat_frequency(all_data, 'SaleType')
cat_imputation(all_data, 'SaleType', 'WD')


# Electrical train
# cat_null_sum('Electrical')
# cat_frequency(all_data, 'Electrical')
cat_imputation(all_data, 'Electrical', 'SBrkr')


# 查看是否还有缺失值
all_data.isnull().sum().max()


# 调整数据的格式，可能因为all_data使用了concat的缘故，导致了许多原本整型或浮点型的数据格式变成了Object类型，需要跟train做比较
train_data_cols = train_data.columns
all_data_cols = all_data.columns
for col in all_data_cols:
    if col in train_data_cols:
            tmp_col = all_data[col].astype(train_data[col].dtype)
            tmp_col = pd.DataFrame({col: tmp_col})
            del all_data[col]
            all_data = pd.concat((all_data, tmp_col), axis=1)


# 到此为止，我们基本把所有的缺失值都填补完整了，但是还有一列MSSubClass，原始数据类型是int64,我并不认为这一列具有可比性，所以把MSSubClass映射成object
# convert MSSubClass to object
# all_data = all_data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
#                                                 60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
#                                                 90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})

all_data = all_data.replace({"ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}})
all_data = all_data.replace({"BsmtFinType1": {"GLQ": 5, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}})
all_data = all_data.replace({"BsmtFinType2": {"GLQ": 5, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}})
all_data = all_data.replace({"GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}})


# 将所有categorical类型的特征进行one-hot编码。需要注意的是：训练集和测试集中，相同的列可能会有不同的类型需要统一
# for col in test_data.columns:
#     t1 = test_data[col].dtype
#     t2 = train_data[col].dtype
#     if t1 != t2:
#         print(col, t1, t2)

# convert to type of int64
# cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
# for col in cols:
#     tmp_col = test_data[col].astype(pd.np.float64)
#     tmp_col = pd.DataFrame({col: tmp_col})
#     del test_data[col]
#     test_data = pd.concat((test_data, tmp_col), axis=1)
#
# cols = ['GarageCars', 'GarageArea', 'BsmtFullBath']
# for col in cols:
#     tmp_col = train_data[col].astype(pd.np.float64)
#     tmp_col = pd.DataFrame({col: tmp_col})
#     del train_data[col]
#     train_data = pd.concat((train_data, tmp_col), axis=1)

# one-hot编码，pandas get_dummies
# for cols in train_data.columns:
#     if train_data[cols].dtype == np.object:
#         train_data = pd.concat((train_data, pd.get_dummies(train_data[cols], prefix=cols)), axis=1)
#         del train_data[cols]
#
# for cols in test_data.columns:
#     if test_data[cols].dtype == np.object:
#         test_data = pd.concat((test_data, pd.get_dummies(test_data[cols], prefix=cols)), axis=1)
#         del test_data[cols]


# log transform skewed numeric features:
from scipy.stats import skew
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #Computes the skewness of a data set.
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# dummy
all_data = pd.get_dummies(all_data)
train_x = all_data[all_data['Id'] < 1461]
test_x = all_data[all_data['Id'] > 1460]
all_data = all_data.drop(['Id'], axis=1)
train_y = np.log1p(train_data['SalePrice'])
test_id = test_data['Id'].astype(pd.np.int64)
cols = 291

# pd.DataFrame(train_x.columns).to_csv("kaggle/housePrices/temp/columns.csv", index=False)



# 进行one-hot编码后，会出现一种情况就是：某个特征的某一个取值只出现在训练集中，没有出现在测试集中，或者相反，这个时候需要特征对齐
# 特征对齐
# col_train = train_data.columns
# col_test = test_data.columns
# for index in col_train:
#     if index not in col_test:
#         del train_data[index]
#
# col_train = train_data.columns
# col_test = test_data.columns
# for index in col_test:
#     if index not in col_train:
#         del test_data[index]


# 数据标准化
# scaler = preprocessing.scale(train_x)
# StandardScaler().fit_transform(train_x)


# 对齐后数据有294个特征，而训练样本只有1460个，相对而言，样本数目偏少。可通过随机森林等算法，对特征做一次初步的选择，取前100即可
# 特征重要性选择
etr = RandomForestRegressor(n_estimators=400)
#                 random_state=1,       # 指在相同数据和相同参数下，是否每次都得一样的结果，如果是1，代表是？
#                 learning_rate=0.015,  #
#                 min_samples_split=2,  # 根据属性划分节点时，每个划分最少的样本数
#                 max_features='sqrt',  # 最大特征数量: 1.Auto/None 2.sqrt  3.0.x 4.log2 5.整数，一般特征越多，模型越准，但需要考虑性能和准确率的平衡
#                 n_estimators          # 决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了）可以达到可接受的性能和误差率
#                 min_samples_leaf=1,   # 叶结点上最少的样本数量,通常越小越能捕获异常值，一般设置50
#                 subsample=0.2,        #
#                 max_depth=3,          # 设置树的最大深度，默认为None，这样建树时，会使每一个叶节点只有一个类别，或是达到min_samples_split
#                 #n_jobs=-1            # 使用处理器（并行）的数量，1表示不并行，n表示n个并行，-1表示没有限制，并行对bagging（非boosting）重要
#                 #min_samples_leaf     #(default=None)叶子树的最大样本数
#                 #min_weight_fraction_leaf    #  (default=0) 叶子节点所需要的最小权值
#                 #criterion            # "gini" or “entropy"(default="gini")是计算属性的gini(基尼不纯度)还是entropy(信息增益)，来选择最合适的节点
#                 #splitter             # "best" or “random"(default="best")随机选择属性还是选择不纯度最大的属性，建议用默认
#                 #bootstrap            # 是否有放回的采样
#                 #warm_start           # 热启动，决定是否使用上次调用该类的结果然后增加新的
#                 #class_weight         # 各个label的权重
#                 #oob_score            # oob（out of band，带外）数据，即：在某次决策树训练中没有被bootstrap选中的数据。多单个模型的参数训练，
#                                       # 我们知道可以用cross validation（cv）来进行，但是特别消耗时间，而且对于随机森林这种情况也没有大的必要，所以就用这个数据对决策树模型进行验证，算是一个简单的交叉验证。性能消耗小，但是效果不错

etr.fit(train_x, train_y)
# print(etr.feature_importances_)
imp = etr.feature_importances_
imp = pd.DataFrame({'feature': train_x.columns, 'score': imp})
# print(imp.sort(['score'], ascending=[0]))  # 按照特征重要性, 进行降序排列, 最重要的特征在最前面
imp = imp.sort(['score'], ascending=[0])
# imp.to_csv("kaggle/housePrices/temp/feature_importances2.csv", index=False)



# 进行预测可以有几种形式：
# predict_proba(x)：给出带有概率值的结果。每个点在所有label的概率和为1.
# predict(x)：直接给出预测结果。内部还是调用的predict_proba()，根据概率的结果看哪个类型的预测值最高就是哪个类型。
# predict_log_proba(x)：和predict_proba基本上一样，只是把结果给做了log()处理。

# gbrt = GradientBoostingRegressor(
#                 random_state=1,       # 作为每次产生随机数的随机种子
                                        # 使用随机种子对于调参过程是很重要的，因为如果我们每次都用不同的随机种子，即使参数值没变每次出来的结果也会不同，这样不利于比较不同模型的结果
                                        # 任一个随即样本都有可能导致过度拟合，可以用不同的随机样本建模来减少过度拟合的可能，但这样计算上也会昂贵很多，因而我们很少这样用
#                 learning_rate=0.015,  # 学习率
#                 min_samples_split=2,  # 根据属性划分节点时，每个划分最少的样本数
#                 max_features='sqrt',  # 最大特征数量: 1.Auto/None 2.sqrt  3.0.x 4.log2 5.整数，一般特征越多，模型越准，但需要考虑性能和准确率的平衡
#                 n_estimators=400,     # 决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了）可以达到可接受的性能和误差率
#                 min_samples_leaf=1,   # 叶结点上最少的样本数量,通常越小越能捕获异常值，一般设置50
#                 subsample=0.2,        # 训练每个决定树所用到的子样本占总样本的比例，而对于子样本的选择是随机的
#                                       # 用稍小于1的值能够使模型更稳健，因为这样减少了方差
#                                       # 一把来说用~0.8就行了，更好的结果可以用调参获得
#                 max_depth=3,          # 设置树的最大深度，默认为None，这样建树时，会使每一个叶节点只有一个类别，或是达到min_samples_split
#                 loss,                 # 指的是每一次节点分裂所要最小化的损失函数(loss function)
                                        # 对于分类和回归模型可以有不同的值。一般来说不用更改，用默认值就可以了，除非你对它及它对模型的影响很清楚
#                 init ,                # 它影响了输出参数的起始化过程
#                                       # 如果我们有一个模型，它的输出结果会用来作为GBM模型的起始估计，这个时候就可以用init
#                 verbose,              # 决定建模完成后对输出的打印方式：0：不输出任何结果（默认）; 1：打印特定区域的树的输出结果; >1：打印所有结果
#                 warm_start            # 这个参数的效果很有趣，有效地使用它可以省很多事
#                                       # 使用它我们就可以用一个建好的模型来训练额外的决定树，能节省大量的时间，对于高阶应用我们应该多多探索这个选项
#                 presort               # 决定是否对数据进行预排序，可以使得树分裂地更快
#                                       # 默认情况下是自动选择的，当然你可以对其更改
#             )


# from scipy.stats import skew
from scipy.stats.stats import pearsonr
#log transform the target:
# train_y["SalePrice"] = np.log1p(train_y)

#log transform skewed numeric features:
# numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index
# skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) #Computes the skewness of a data set.
# skewed_feats = skewed_feats[skewed_feats > 0.75]
# skewed_feats = skewed_feats.index
# train_data[skewed_feats] = np.log1p(train_data[skewed_feats])

# all_data = pd.get_dummies(all_data)
# #filling NA's with the mean of the column:
# all_data = all_data.fillna(all_data.mean())
# #creating matrices for sklearn:
# X_train = all_data[:train.shape[0]]
# X_test = all_data[train.shape[0]:]
# y = train.SalePrice

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
#
def rmse_cv(model):
# “neg_mean_squared_error”是将预测的relevance的值和实际的relevance的值进行均方误差，即第二步中的MSE的公式。但是用corss_val_score返回的score是负数值，需要加负号再开根号
    rmse= np.sqrt(-cross_val_score(model, train_x[imp.head(cols)['feature']], train_y, scoring="neg_mean_squared_error", cv = 5))    # neg_mean_squared_error: 计算均方误差
    return(rmse)

# RidgeCV
# min during 12.2 - 12.4
# alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
alphas = [11, 11.2, 11.4, 11.6, 11.8, 12, 12.2, 12.4, 12.6, 12.8, 13]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")

# 0.0004
alphas = [110, 120, 130, 140, 150, 160, 170, 180, 190]
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")

# 0.0004
model_lasso = LassoCV(alphas = [0.0004]).fit(train_x, train_y)
rmse_cv(model_lasso).mean()

coef = pd.Series(model_lasso.coef_, index = train_x.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
coef = coef[coef != 0]

imp_coef = pd.concat([coef.sort_values().head(10),  coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(train_x), "true": train_y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals", kind = "scatter")

import xgboost as xgb

# train_x[imp.head(293)['feature']]
# train_x[coef.index]
dtrain = xgb.DMatrix(train_x, label = train_y)
# dtest = xgb.DMatrix(test_x[imp.head(cols)['feature']])

params = {
    "max_depth": 5,
    "eta": 0.1,
    "min_child_weight": 4,
    "gamma": 0,
    "objective": "reg:linear",
    "subsample": 0.8,
    "colsample_bytree": 0.8}
model = xgb.cv(params, dtrain,  num_boost_round=1000, early_stopping_rounds=100, nfold=5, metrics='rmse')
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
print(model['test-rmse-mean'].min())
print(model['train-rmse-mean'].min())

model_xgb = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=4,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:linear',
    nthread=4,
    scale_pos_weight=1,
    seed=27) #the params were tuned using xgb.cv
model_xgb.fit(train_x, train_y)

xgb_preds = np.expm1(model_xgb.predict(test_x))
lasso_preds = np.expm1(model_lasso.predict(test_x))

lasso_xgb_preds = 0.7 * lasso_preds + 0.3 * xgb_preds
predictions = pd.DataFrame({"xgb":lasso_xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

preds = 0.7 * lasso_preds + 0.3 * xgb_preds
solution = pd.DataFrame({"Id": test_id, "SalePrice": preds})
solution.to_csv("kaggle/housePrices/temp/lasso_xgb_test_result.csv", index=False)

lasso_preds = model_lasso.predict(test_x[imp.head(cols)['feature']])
result = pd.DataFrame({"Id": test_id, "SalePrice": lasso_preds})
result.to_csv("kaggle/housePrices/temp/lasso_test_result.csv", index=False)


# 1.Get More Data 2.Invent More Data 3.Clean Your Data 4.Resample Data 5.Reframe Your Problem 6.Rescale Your Data 7.Transform Your Data 8.Project Your Data 9.Feature Selection 10.Feature Engineering
# 1.Resampling Method 2.Evaluation Metric 3.Baseline Performance 4.Spot Check Linear Algorithms 5.Spot Check Nonlinear Algorithms 6.Steal from Literature 7.Standard Configurations
# 1.Diagnostics 2.Try Intuition 3.Steal from Literature 4.Random Search 5.Grid Search 6.Optimize 7.Alternate Implementations 8.Algorithm Extensions 9.Algorithm Customizations 10.Contact Experts
# 1.Blend Model Predictions 2.Blend Data Representations 3.Blend Data Samples 4.Correct Predictions 5.Learn to Combine
# 尝试聚类，去掉异常值
# 把train和test结合起来，创造更多的数据

#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# train = pd.read_csv('train_modified.csv')
# target = 'Disbursed'
# IDcol = 'ID'

# useTrainCV=True
def modelfit(alg, train_x, train_y,useTrainCV=False, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_x, label=train_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])  # 行(第一维)的长度

    #Fit the algorithm on the data
    alg.fit(train_x, train_y, eval_metric='rmse')

    #Predict training set:
    # dtrain_predictions = alg.predict(train_x)
    # dtrain_predprob = alg.predict_proba(train_x)[:,1]

    #Print model report:
    print("\nModel Report")
    # print("dtrain_predictions:", dtrain_predictions)
    # print("Accuracy : %.4g" % metrics.accuracy_score(train_y, dtrain_predictions))
    # print("rmse Score (Train): %f" % metrics.roc_auc_score(train_y, dtrain_predprob))
    print('Training Error: {:.3f}'.format(1 - alg.score(train_x, train_y)))

    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp = pd.concat([feat_imp.head(50),  feat_imp.tail(50)])
    # feat_imp.head(50).plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

    dtest_predictions = alg.predict(test_x)
    dtest_predictions = np.expm1(dtest_predictions)
    test_result = pd.DataFrame({"Id": test_id, "SalePrice": dtest_predictions})
    test_result.to_csv("kaggle/housePrices/temp/xgb_test_result.csv", index=False)

# Choose all predictors except target & IDcols
xgb1 = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=4,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:linear',
    # scale_pos_weight=1,
    # seed=27,
    nthread=4)
modelfit(xgb1, train_x, train_y)

# max_features
# for i in np.arange(10):
#     print('i=', i)
#     modelfit(xgb1, train_x[imp.head(120 + i * 10)['feature']], train_y)

# {'max_depth': 2, 'min_child_weight': 3},
# -0.015092031338683542)
param_test1 = {
 'max_depth': [2, 3, 4, 5, 6, 7],
 'min_child_weight': [1, 2, 3, 4, 5, 6]
}

# {'n_estimators': 410},
#  -0.015797678676584055)
# param_test1 = {
#  'n_estimators': [385, 390, 395, 400, 405, 410, 415]
# }

# {'learning_rate': 0.08},
# -0.015063409091199001)
# param_test1 = {
#  'learning_rate': [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
# }

# {'gamma': 0.0},
#  -0.015063409091199001)
# param_test1 = {
#  'gamma': [i/10.0 for i in range(0,5)]
# }

# {'colsample_bytree': 0.8, 'subsample': 0.8},
#  -0.015063409091199001)
# param_test1 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }

 # {'reg_alpha': 0.1},
 # -0.014945822756303013)
# param_test1 = {
#  'reg_alpha':[0.05, 0.06,  0.07,  0.08,  0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
# }

gsearch1 = GridSearchCV(
    estimator = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=2,
        min_child_weight=3,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:linear',
        # scale_pos_weight=1,
        # seed=27,
        nthread=4),
    param_grid=param_test1,
    scoring='neg_mean_squared_error',   # scoring: ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
    n_jobs=4,
    iid=False,
    cv=5)
gsearch1.fit(train_x, train_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

# 0.12831