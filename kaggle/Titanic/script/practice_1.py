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
import xgboost
from subprocess import check_output
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.neighbors import NearestNeighbors
import warnings
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
# print(check_output(['1s', '../input']).decode('utf-8'))
sns.set(style = "white", color_codes = True)
warnings.filterwarnings('ignore')

# input
train_data = pd.read_csv('kaggle/Titanic/input/train.csv')
test_data = pd.read_csv('kaggle/Titanic/input/test.csv')
all_data = pd.concat((train_data[test_data.columns], test_data), ignore_index=True)

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

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
def cat_info(data):
        print(data.info())
# Imputing the missing values
def cat_imputation(data, col, val):
    data.loc[data[col].isnull(), col] = val
def cat_null_sum(data, col):
    print(data[col].isnull().sum())

# 区分直方图与条形图：
# 条形图是用条形的长度表示各类别频数的多少，其宽度（表示类别）则是固定的；
# 直方图是用面积表示各组频数的多少，矩形的高度表示每一组的频数或频率，宽度则表示各组的组距，因此其高度与宽度均有意义。
# 由于分组数据具有连续性，直方图的各矩形通常是连续排列，而条形图则是分开排列。
# 条形图主要用于展示分类数据，而直方图则主要用于展示数据型数据
# bar
def bar(data, column):
    plt.figure()
    if type(column) == str:
        column = data[column]
    vc = column.value_counts().sort_values(ascending = False)
    x_pos = np.arange(len(vc))
    plt.bar(x_pos, vc)
    plt.xticks(x_pos, vc.index)
    plt.xlabel(column.name)
    plt.ylabel('frequency')

# barh
def barh(data, column):
    plt.figure()
    if type(column) == str:
        column = data[column]
    vc = column.value_counts().sort_values(ascending = False)
    y_pos = np.arange(len(vc))
    plt.barh(y_pos, vc)
    plt.yticks(y_pos, vc.index)
    plt.ylabel(column.name)
    plt.xlabel('frequency')

# hist
def hist(data, column):
    plt.figure()
    if type(column) == str:
        plt.hist(data[data[column].isnull() == False][column])
    else:
        plt.hist(column)

# displot
def displot(data, column):
    plt.figure()
    if type(column) == str:
        sns.distplot(data[data[column].isnull() == False][column], fit=norm)
    else:
        sns.distplot(column, fit=norm)

# boxplot
def boxplot(data, column):
    plt.figure()
    if type(column) == str:
        sns.boxplot(data[data[column].isnull() == False][column])
    else:
        sns.boxplot(column)

# violinplot
def violinplot(data, column):
    plt.figure()
    if type(column) == str:
        sns.violinplot(data[data[column].isnull() == False][column])
    else:
        sns.violinplot(column)

# scatter
def scatter(data, column1, column2):
    plt.figure()
    if type(column1) == str:
        plt.scatter(data[column1], data[column2])
    else:
        plt.scatter(column1, column2)

# factorplot
def factorplot(data, column1, column2):
    plt.figure()
    sns.factorplot(column1, column2, data=data, size=4, aspect=3)

# probplot
def probplot(data, column):
    plt.figure()
    if type(column) == str:
        stats.probplot(data[data[column].isnull() == False][column], plot=plt)
    else:
        stats.probplot(column, plot=plt)

# pairplot
def pairplot(data, hue):
    plt.figure()
    sns.pairplot(data, hue=hue, size=3)

# heatmap
def heatmap(data, target, k=10):
    corrmat = data.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat, vmax=.8, square=True)
    cols = corrmat.nlargest(k, target)[target].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=cols.values,
                xticklabels=cols.values)

def FacetGrid_scatter(data, hue, column1, column2):
    sns.FacetGrid(data, hue=hue, size=5).map(plt.scatter, column1, column2).add_legend()

def FacetGrid_kdeplot(data, hue, column):
    sns.FacetGrid(data, hue=hue, size=5).map(sns.kdeplot, column).add_legend()

def FacetGrid_displot(data, hue, column):
    sns.FacetGrid(data, hue=hue, size=5).map(sns.distplot, column).add_legend()

# missing data
# Age 263 train 177 & test 86
# 查看Age为空的其他特征的情况
# cat_frequency(train_data[train_data['Age'].isnull() == True], 'Survived') # 0=125 & 1=52
# cat_frequency(train_data[train_data['Age'].isnull() == True], 'Pclass') # 3=136 & 1=30 & 2=11
# cat_frequency(train_data[train_data['Age'].isnull() == True], 'Sex') # male=124 & female=53
all_data['Age'].fillna(all_data['Age'].mean(), inplace=True)

# Fare test 1
all_data['Fare'].fillna(all_data['Fare'].mean(), inplace=True)

# Cabin train 687 & test 327


# Embarked train 2
# 暂时impute为众数，因为可以考虑家庭关系
all_data['Embarked'].fillna('S', inplace=True)

