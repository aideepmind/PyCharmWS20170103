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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
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
bool_ = all_data['Alley'].isnull()
bool_ = pd.DataFrame({'HasAlley': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)
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
# all_data = all_data.drop(['PoolQC'], axis=1)
# cat_null_sum('PoolArea')
# 只有13个大于0的值，drop
# all_data = all_data.drop(['PoolArea'], axis=1)


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
# all_data = all_data.drop(train_data[(train_data['SalePrice'] > 380000) & (train_data['OverallQual'] == 7)].index , axis=0)
# train_data = train_data.drop(train_data[(train_data['SalePrice'] > 380000) & (train_data['OverallQual'] == 7)].index , axis=0)
# all_data = all_data.drop(train_data[(train_data['SalePrice'] < 100000) & (train_data['OverallQual'] == 7)].index , axis=0)
# train_data = train_data.drop(train_data[(train_data['SalePrice'] < 100000) & (train_data['OverallQual'] == 7)].index , axis=0)
# all_data = all_data.drop(train_data[(train_data['SalePrice'] > 500000) & (train_data['OverallQual'] == 8)].index , axis=0)
# train_data = train_data.drop(train_data[(train_data['SalePrice'] > 500000) & (train_data['OverallQual'] == 8)].index , axis=0)
# all_data = all_data.drop(train_data[(train_data['SalePrice'] < 130000) & (train_data['OverallQual'] == 8)].index , axis=0)
# train_data = train_data.drop(train_data[(train_data['SalePrice'] < 130000) & (train_data['OverallQual'] == 8)].index , axis=0)
# according to TotalBsmtSF
# all_data['TotalBsmtSF'][all_data['TotalBsmtSF'] > 2500]
# all_data = all_data.drop(train_data[train_data['TotalBsmtSF'] > 3000].index , axis=0)
# train_data = train_data.drop(train_data[train_data['TotalBsmtSF'] > 3000].index , axis=0)
# 让test数据中大于3000的取all中大于2500的均值
# all_data.loc[all_data['TotalBsmtSF'] > 3000, 'TotalBsmtSF'] = np.mean(all_data['TotalBsmtSF'][all_data['TotalBsmtSF'] > 2500])
# according to GarageCars
# var = 'GarageCars'
# data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
# according to GarageArea
# plt.scatter(train_data['GarageArea'], train_data['SalePrice'])

# 到此为止，我们基本把所有的缺失值都填补完整了，但是还有一列MSSubClass，原始数据类型是int64,我并不认为这一列具有可比性，所以把MSSubClass映射成object
# convert MSSubClass to object
# all_data = all_data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
#                                                 60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
#                                                 90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})

all_data = all_data.replace({"ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}})
all_data = all_data.replace({"BsmtFinType1": {"GLQ": 5, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}})
all_data = all_data.replace({"BsmtFinType2": {"GLQ": 5, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}})
all_data = all_data.replace({"GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}})
all_data = all_data.replace({"GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}})
all_data = all_data.replace({"PavedDrive": {"Y": 3, "P": 2, "N": 1}})
all_data = all_data.replace({"Fence": {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}})


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
# Neighborhood 通过K-Means来分类
# 根据地址借助外部工具（www.addressreport.com）收集一些新的数据（Location,Cost_of_living,Income,Owners,Annual_property_tax,School,Crime,Ville）
ngbr_details = pd.read_csv('kaggle/housePrices/input/neighborhood_details.csv')
ngbr_details = ngbr_details.drop('Neighborhood_etiquette', axis=1)
# 使用均值填充，但字符类型无法填充所以使用众数来填充
ngbr_details = ngbr_details.fillna(ngbr_details.mean())
# mode返回众数值和对应次数（只能是数字类型）
# ngbr_details['Location'].fillna(mode(ngbr_details['Location']).mode[0], inplace=True)
ngbr_details['Location'].fillna("NN", inplace=True)
# Location的one-hot
ngbr_details = pd.concat((ngbr_details, pd.get_dummies(ngbr_details['Location'], prefix='Location')), axis=1)
ngbr_details = ngbr_details.drop('Location', axis = 1)
# 合并 pd.merge
all_data = pd.merge(all_data, ngbr_details, how='left', on='Neighborhood', sort=False)
# 按Neighborhood分组
grouped = train_data[['Neighborhood', 'SalePrice']].groupby('Neighborhood')
temp = pd.concat((grouped.mean().rename(columns = {'SalePrice':'meanSalePrice'}), grouped.median().rename(columns = {'SalePrice':'medSalePrice'})), axis=1)
temp = pd.concat((temp, grouped.std().rename(columns = {'SalePrice':'stdSalePrice'})), axis=1)
temp = pd.concat((temp, grouped.count().rename(columns = {'SalePrice':'countSalePrice'})), axis=1)
# K-Means聚类
km = KMeans(n_clusters=10).fit_predict(temp)
temp = pd.DataFrame({'Neighborhood': temp.index, 'NeighborhoodCl': km})
all_data = pd.merge(all_data, temp, how='left', on='Neighborhood', sort=False)
for cls in np.arange(10):
    bool_ = all_data['NeighborhoodCl'] == cls
    bool_ = pd.DataFrame({'Neighborhood' + str(cls): bool_}, index=bool_.index)
    bool_ = bool_.astype(pd.np.float64)
    all_data = pd.concat((all_data, bool_), axis=1)
all_data = all_data.drop('Neighborhood', axis=1)
all_data = all_data.drop('NeighborhoodCl', axis=1)


# Condition1  & Condition2
# print((all_data['Condition1'] != all_data['Condition2']).value_counts())
# print(all_data[all_data['Condition1'] != all_data['Condition2']][['Condition1', 'Condition2']])
# 交通条件，有Condition1和Condition2，如果Condition1=Condition2，则只有一个，否则是拥有两个不一样的条件
col_values = ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
for index in np.arange(len(col_values)):
    bool_ = (all_data['Condition1'] == col_values[index]) | (all_data['Condition2'] == col_values[index])
    bool_ = pd.DataFrame({'Condition_' + col_values[index]: bool_}, index=bool_.index)
    bool_ = bool_.astype(pd.np.float64)
    all_data = pd.concat((all_data, bool_), axis=1)
bool_ = all_data['Condition1'] != all_data['Condition2']
bool_ = pd.DataFrame({'Condition_Has_Two': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)

all_data = all_data.drop('Condition1', axis=1)
all_data = all_data.drop('Condition2', axis=1)


# YearBuilt & YearRemodAdd
# displot('YearRemodAdd')
# displot('YearRemodAdd')
# scatter('YearRemodAdd', 'YearBuilt')
# all_data[['YearBuilt', 'YearRemodAdd']][all_data['YearBuilt'] > all_data['YearRemodAdd']]
# 从数据中可以看出YearRemodAdd在1950年之前的数据没有，所以需要利用与YearBuilt的线性关系预测其1950年数据
regr = LinearRegression()
year_after_1950 = all_data[['YearBuilt', 'YearRemodAdd']][all_data['YearBuilt'] > 1950]
year_before_1950 = all_data[['YearBuilt']][all_data['YearRemodAdd'] <= 1950]
# series不能转置，故year_after_1950['YearBuilt'].T无效，需转换成DataFrame
regr.fit(year_after_1950['YearBuilt'].to_frame(), year_after_1950['YearRemodAdd'])
year_before_1950_preds = np.round(regr.predict(year_before_1950))
all_data.loc[all_data['YearRemodAdd'] <= 1950, 'YearRemodAdd'] = year_before_1950_preds
bool_ = all_data['YearBuilt'] != all_data['YearRemodAdd']
bool_ = pd.DataFrame({'isRemod': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)


#Fireplaces
bool_ = all_data['Fireplaces'] > 0
bool_ = pd.DataFrame({'HasFireplaces': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)
# all_data = all_data.drop('Fireplaces', axis=1)


#WoodDeckSF
bool_ = all_data['WoodDeckSF'] > 0
bool_ = pd.DataFrame({'HasWoodDeckSF': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)


#OpenPorchSF
bool_ = all_data['OpenPorchSF'] > 0
bool_ = pd.DataFrame({'HasOpenPorchSF': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)


#EnclosedPorch
bool_ = all_data['EnclosedPorch'] > 0
bool_ = pd.DataFrame({'HasEnclosedPorch': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)


#3SsnPorch
bool_ = all_data['3SsnPorch'] > 0
bool_ = pd.DataFrame({'Has3SsnPorch': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)


# ScreenPorch
bool_ = all_data['ScreenPorch'] > 0
bool_ = pd.DataFrame({'HasScreenPorch': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)


# PoolArea 描述是否有的问题
bool_ = all_data['PoolArea'] > 0
bool_ = pd.DataFrame({'HasPoolArea': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)


# PoolQC 描述质量问题
bool_ = (all_data['PoolQC'] == 'Ex') | (all_data['PoolQC'] == 'Gd')
bool_ = pd.DataFrame({'PoolQCExOrGd': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)
all_data = all_data.drop('PoolQC', axis=1)


# Fence
bool_ = all_data['Fence'] != 0
bool_ = pd.DataFrame({'HasFence': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)


# MiscFeature
bool_ = all_data['MiscFeature'] == 'Gar2'
bool_ = pd.DataFrame({'HasMiscFeature_Gar2': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)

bool_ = all_data['MiscFeature'] == 'Shed'
bool_ = pd.DataFrame({'HasMiscFeature_Shed': bool_}, index=bool_.index)
bool_ = bool_.astype(pd.np.float64)
all_data = pd.concat((all_data, bool_), axis=1)
all_data = all_data.drop('MiscFeature', axis=1)

# YrSold
all_data['YrSold'] = all_data['YrSold'] - 2005
all_data.YearBetweenSoldAndBuilt = all_data['YrSold'] - all_data['YearBuilt']




# log transform skewed numeric features:
# from scipy.stats import skew
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #Computes the skewness of a data set.
# skewed_feats = skewed_feats[skewed_feats > 0.75]
# skewed_feats = skewed_feats.index
# all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# 数据缩放
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# numeric_feats = numeric_feats.delete(0) # del Id
# min_max_scaler = preprocessing.MinMaxScaler()
# all_data[numeric_feats] = min_max_scaler.fit_transform(all_data[numeric_feats])
# all_data.to_csv("kaggle/housePrices/temp/all_data.csv", index=False)


# 数据标准化
# scaler = preprocessing.scale(train_x)
# StandardScaler().fit_transform(train_x)


# dummy
all_data = pd.get_dummies(all_data)
train_x = all_data[all_data['Id'] < 1461]
test_x = all_data[all_data['Id'] > 1460]
all_data = all_data.drop(['Id'], axis=1)
train_x = train_x.drop(['Id'], axis=1)
test_x = test_x.drop(['Id'], axis=1)
train_y = np.log1p(train_data['SalePrice'])
# train_y = min_max_scaler.fit_transform(train_data['SalePrice'])
test_id = test_data['Id'].astype(pd.np.int64)
cols = len(all_data.columns)

# pd.DataFrame(train_x.columns).to_csv("kaggle/housePrices/temp/columns.csv", index=False)

#  {'alpha': 0.0005}, 0.9236106220776238
param_grid_lasso = {
    'alpha': [0.0005]
}
lasso = GridSearchCV(estimator=Lasso(alpha=0.0005),  param_grid=param_grid_lasso, cv=5)
lasso.fit(train_x, train_y)
lasso.grid_scores_, lasso.best_params_, lasso.best_score_
preds_lasso = lasso.predict(train_x)


# "alpha": [1e0, 1e-1, 1e-2, 1e-3]
# {'alpha': 8.6}, 0.9186295320921184)
param_grid_krr = {
    "alpha": [8.6]}
kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid_krr)
kr.fit(train_x, train_y)
kr.grid_scores_, kr.best_params_, kr.best_score_
preds_krr = kr.predict(train_x)

# preds = 0.7 * preds_lasso + 0.3 * preds_krr
# plt.scatter(preds_lasso, preds_krr)
# plt.scatter(preds, np.expm1(train_y))
# print(np.sqrt(metrics.mean_squared_error(train_y, preds_krr)))

# 把预测值与训练数据合并
preds_lasso_df = pd.DataFrame(preds_lasso, columns=['preds_lasso_df'], index=train_x.index)
preds_krr_df = pd.DataFrame(preds_krr, columns=['preds_krr_df'], index=train_x.index)
train_x = pd.concat((train_x, preds_lasso_df), axis=1)
train_x = pd.concat((train_x, preds_krr_df), axis=1)

# 再使用xgb训练
# {'max_depth': 8, 'min_child_weight': 2},0.9322832068993874)
# param_grid_xgb = {
#  'max_depth': [2],
#  'min_child_weight': [1]
# }

# {'n_estimators': 410},
#  -0.015797678676584055)
# param_grid_xgb = {
#  'n_estimators': [385, 390, 395, 400, 405, 410, 415]
# }

# {'learning_rate': 0.08},
# -0.015063409091199001)
param_grid_xgb = {
 'learning_rate': [0.06, 0.07, 0.08, 0.09, 0.1]
}

# {'gamma': 0.0},
#  -0.015063409091199001)
# param_grid_xgb = {
#  'gamma': [i/10.0 for i in range(0,5)]
# }

# {'colsample_bytree': 0.8, 'subsample': 0.8},
#  -0.015063409091199001)
# param_grid_xgb = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }

 # {'reg_alpha': 0.1},
 # -0.014945822756303013)
# param_grid_xgb = {
#  'reg_alpha':[0.05, 0.06,  0.07,  0.08,  0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
# }

gscv_xgb = GridSearchCV(
    estimator = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.08,
        max_depth=2,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:linear',
        # scale_pos_weight=1,
        # seed=27,
        nthread=4),
    param_grid=param_grid_xgb,
    n_jobs=2,
    iid=False,
    cv=5)
gscv_xgb.fit(train_x, train_y)
gscv_xgb.grid_scores_, gscv_xgb.best_params_, gscv_xgb.best_score_
preds_xgb = np.expm1(gscv_xgb.predict(train_x))
# plt.scatter(preds_lasso, preds_krr)
# plt.scatter(gscv_xgb.predict(train_x), train_y)
print(np.sqrt(metrics.mean_squared_error(train_y, gscv_xgb.predict(train_x))))
# result_lasso_krr['SalePrice'][result_lasso_krr['Id'] == 2550] = 235594.51145508
# 预测
preds_lasso_test = lasso.predict(test_x)
preds_kr_test = kr.predict(test_x)
preds_lasso_df_test = pd.DataFrame(preds_lasso_test, columns=['preds_lasso_df'], index=test_x.index)
preds_kr_df_test = pd.DataFrame(preds_kr_test, columns=['preds_krr_df'], index=test_x.index)
test_x = pd.concat((test_x, preds_lasso_df_test), axis=1)
test_x = pd.concat((test_x, preds_kr_df_test), axis=1)
preds_test = np.expm1(gscv_xgb.predict(test_x))
result = pd.DataFrame({"Id": test_id, "SalePrice": preds_test})
plt.figure()
plt.scatter(gscv_xgb.predict(test_x), preds_lasso_df_test)
plt.xlim([10, 14])
plt.ylim([10, 14])
result.to_csv("kaggle/housePrices/temp/lasso_krr_xgb_test_result.csv", index=False)