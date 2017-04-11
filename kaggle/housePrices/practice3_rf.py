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
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.neighbors import NearestNeighbors
import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
# print(check_output(['1s', '../input']).decode('utf-8'))
sns.set(style = "white", color_codes = True)
warnings.filterwarnings('ignore')

# input
train_data = pd.read_csv('kaggle/housePrices/input/train.csv')
test_data = pd.read_csv('kaggle/housePrices/input/test.csv')
all_data = pd.concat((train_data[test_data.columns], test_data), ignore_index=True)

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
# 486个空值，不过，重要性排名竟然得第一，怎麼搞的？暂时delete
all_data = all_data.drop(['LotFrontage'], axis=1)

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
# print(train_data[basement_cols][train_data['BsmtFinType2'].isnull() == True])
# print(pd.crosstab(all_data.BsmtQual, all_data.BsmtExposure))
# cat_frequency(all_data, 'BsmtExposure')
# BsmtExposure和BsmtFinType2一共38个空值并有2个不一致，其他37个空值，且一致
# d1 = all_data[all_data['BsmtFinSF1'] > 1000]
# d1 = all_data[all_data['BsmtFinSF1'] < 1200]
# cat_frequency(d1, 'BsmtExposure')
all_data.loc[all_data['Id'] == 949, 'BsmtExposure'] = 'No'
# d1 = all_data[all_data['BsmtFinSF2'] > 450]
# d1 = d1[d1['BsmtFinSF2'] < 500]
# cat_frequency(d1, 'BsmtFinType2')
all_data.loc[all_data['Id'] == 333, 'BsmtFinType2'] = 'Rec' # LwQ/BLQ/LwQ
# for cols in basement_cols:
#     if 'FinFS' not in cols:
#         cat_imputation(all_data, cols, 'None')

# test
basement_cols = ['Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
# print(test_data['BsmtQual'].isnull().sum())
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
all_data.loc[all_data['Id'] == 2127, 'GarageYrBlt'] = 1978 # mean
all_data.loc[all_data['Id'] == 2127, 'GarageFinish'] = 'Fin' # 因为是1978年，所以是Finished
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


# remove outters
# 524, 1299, 1183, 692
# according to SalePrice
all_data = all_data.drop(train_data[train_data['SalePrice'] > 600000].index , axis=0)
train_data = train_data.drop(train_data[train_data['SalePrice'] > 600000].index , axis=0)
# according to GrLivArea
# plt.scatter(train_data['GrLivArea'], train_data['SalePrice'])
# train_data[train_data['GrLivArea'] > 4000]
all_data = all_data.drop(train_data[train_data['GrLivArea'] > 4000].index , axis=0)
train_data = train_data.drop(train_data[train_data['GrLivArea'] > 4000].index , axis=0)
# according to LotArea & LotFrontage
# train_data['LotArea'][train_data['LotArea'] > 60000]
# test_data['LotArea'][test_data['LotArea'] > 60000]
all_data = all_data.drop(train_data[train_data['LotArea'] > 70000].index , axis=0)
train_data = train_data.drop(train_data[train_data['LotArea'] > 70000].index , axis=0)
# plt.scatter(train_data['LotFrontage'], train_data['SalePrice'])
# test_data['LotFrontage'][test_data['LotFrontage'] > 200]
# all_data = all_data.drop(train_data[train_data['LotFrontage'] > 200].index , axis=0)
# train_data = train_data.drop(train_data[train_data['LotFrontage'] > 200].index , axis=0)

# according to OverallQual
# var = 'OverallQual'
# data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
all_data = all_data.drop(train_data[(train_data['SalePrice'] > 250000) & (train_data['OverallQual'] == 4)].index , axis=0)
train_data = train_data.drop(train_data[(train_data['SalePrice'] > 250000) & (train_data['OverallQual'] == 4)].index , axis=0)
all_data = all_data.drop(train_data[(train_data['SalePrice'] > 380000) & (train_data['OverallQual'] == 7)].index , axis=0)
train_data = train_data.drop(train_data[(train_data['SalePrice'] > 380000) & (train_data['OverallQual'] == 7)].index , axis=0)
all_data = all_data.drop(train_data[(train_data['SalePrice'] < 100000) & (train_data['OverallQual'] == 7)].index , axis=0)
train_data = train_data.drop(train_data[(train_data['SalePrice'] < 100000) & (train_data['OverallQual'] == 7)].index , axis=0)
all_data = all_data.drop(train_data[(train_data['SalePrice'] > 500000) & (train_data['OverallQual'] == 8)].index , axis=0)
train_data = train_data.drop(train_data[(train_data['SalePrice'] > 500000) & (train_data['OverallQual'] == 8)].index , axis=0)
all_data = all_data.drop(train_data[(train_data['SalePrice'] < 130000) & (train_data['OverallQual'] == 8)].index , axis=0)
train_data = train_data.drop(train_data[(train_data['SalePrice'] < 130000) & (train_data['OverallQual'] == 8)].index , axis=0)
# according to TotalBsmtSF
# all_data['TotalBsmtSF'][all_data['TotalBsmtSF'] > 2500]
# all_data = all_data.drop(train_data[train_data['TotalBsmtSF'] > 3000].index , axis=0)
# train_data = train_data.drop(train_data[train_data['TotalBsmtSF'] > 3000].index , axis=0)
# 让test数据中大于3000的取all中大于2500的均值
all_data.loc[all_data['TotalBsmtSF'] > 3000, 'TotalBsmtSF'] = np.mean(all_data['TotalBsmtSF'][all_data['TotalBsmtSF'] > 2500])
# according to GarageCars
# var = 'GarageCars'
# data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
# according to GarageArea
# plt.scatter(train_data['GarageArea'], train_data['SalePrice'])



# log transform skewed numeric features:
from scipy.stats import skew
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #Computes the skewness of a data set.
skewed_feats_1 = skewed_feats[skewed_feats > 0.7]   # 0.6/0.7
skewed_feats_2 = skewed_feats[skewed_feats <= 0.7]
# skewed_feats = skewed_feats.drop('Id')
all_data[skewed_feats_1.index] = np.log1p(all_data[skewed_feats_1.index])

# 数据缩放
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #Computes the skewness of a data set.
# skewed_feats_1 = skewed_feats[skewed_feats > 0.75]   # 0.6/0.7
# numeric_feats = numeric_feats.delete(0) # del Id
# scaler = preprocessing.StandardScaler()
# all_data[numeric_feats] = scaler.fit_transform(all_data[numeric_feats])
# all_data.to_csv("kaggle/housePrices/temp/all_data.csv", index=False)


# 数据标准化
# scaler = preprocessing.scale(train_x)
# StandardScaler().fit_transform(train_x)

# dummy
all_data = pd.get_dummies(all_data)
train_x = all_data[all_data['Id'] < 1461]
test_x = all_data[all_data['Id'] > 1460]
# all_data = all_data.drop(['Id'], axis=1)
train_x = train_x.drop(['Id'], axis=1)
test_x = test_x.drop(['Id'], axis=1)
# train_y = train_data['SalePrice']
train_y = np.log1p(train_data['SalePrice'])
test_id = test_data['Id'].astype(pd.np.int64)
cols = len(all_data.columns)

# pd.DataFrame(train_x.columns).to_csv("kaggle/housePrices/temp/columns.csv", index=False)


#                 random_state=1,       # 指在相同数据和相同参数下，是否每次都得一样的结果，如果是1，代表是？
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
#                  n_estimators=10,
#                  criterion="mse",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_split=1e-7,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=1,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False
# param_test = {
#     'n_estimators': [1000, 1500, 2000]
# }

#  {'max_depth': 20, 'min_samples_split': 5},
#  0.8823693674550974)
param_grid_rf = {
    'max_depth': [5, 20, 50],
    'min_samples_split': [5, 10, 50]
}
rf = GridSearchCV(
    estimator = RandomForestRegressor(
        n_estimators=1000),
    param_grid=param_grid_rf,
    n_jobs=4,
    iid=False,
    cv=5)
rf.fit(train_x, train_y)
rf.grid_scores_, rf.best_params_, rf.best_score_