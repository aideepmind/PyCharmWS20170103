import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
import xgboost
sns.set(style="white", color_codes=True)
warnings.filterwarnings('ignore')
# %matplotlib inline


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
'COMPREHENSIVE DATA EXPLORATION WITH PYTHON'.lower()

# 1 data exploration

# Looking at categorical values
def cat_frequency(data, col):
    return data[col].value_counts()
# Looking at data describe
def cat_describe(data, col):
    data[col].describe()
# Imputing the missing values
def cat_imputation(data, col, val):
    data.loc[data[col].isnull(), col] = val

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

# categorical variables

# 3 feature engineering
# 3.1 feature selection
# 3.2 feature encoding
# 4 model selection
# 4.1 model training
# 5 cross validation
# 6 ensemble generation

# read data
train_data = pd.read_csv('kaggle/housePrices/dataset/train.csv')
test_data = pd.read_csv('kaggle/housePrices/dataset/test.csv')

# SalePrice
# data exploration
# descriptive statistics summary
train_data['SalePrice'].describe()
# histogram
sns.distplot(train_data['SalePrice'])
#skewness and kurtosis
print("Skewness: %f" % train_data['SalePrice'].skew())
# Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1
print("Kurtosis: %f" % train_data['SalePrice'].kurt())
# Relationship with numerical variables
# scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# Relationship with categorical features
# box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
# box plot YearBuilt/saleprice
# var = 'YearBuilt'
# data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
# f, ax = plt.subplots(figsize=(16, 8))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(rotation=90)
# Correlation matrix (heatmap style)
# correlation matrix
# corrmat = train_data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# 'SalePrice' correlation matrix (zoomed heatmap style)
# saleprice correlation matrix
# k = 10 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(train_data[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
# Scatter plots between 'SalePrice' and correlated variables (move like Jagger style)
# scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(train_data[cols], size = 2.5)
# plt.show()
# missing data
total = test_data.isnull().sum().sort_values(ascending=False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# dealing with missing data
# train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
# train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
# train_data.isnull().sum().max() #just checking that there's no missing data missing...
# standardizing data
# saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);
# low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
# high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)
# bivariate analysis saleprice/grlivarea
# var = 'GrLivArea'
# data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# deleting points
# train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]
# train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
# train_data = train_data.drop(train_data[train_data['Id'] == 524].index)
# bivariate analysis saleprice/grlivarea
# var = 'TotalBsmtSF'
# data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# histogram and normal probability plot
# sns.distplot(train_data['SalePrice'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(train_data['SalePrice'], plot=plt)
# applying log transformation
# train_data['SalePrice'] = np.log(train_data['SalePrice'])
# transformed histogram and normal probability plot
# sns.distplot(train_data['SalePrice'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(train_data['SalePrice'], plot=plt)
# histogram and normal probability plot
# sns.distplot(train_data['GrLivArea'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(train_data['GrLivArea'], plot=plt)
# data transformation
# train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
# transformed histogram and normal probability plot
# sns.distplot(train_data['GrLivArea'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(train_data['GrLivArea'], plot=plt)
# histogram and normal probability plot
# sns.distplot(train_data['TotalBsmtSF'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(train_data['TotalBsmtSF'], plot=plt)
# create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0
# train_data['HasBsmt'] = pd.Series(len(train_data['TotalBsmtSF']), index=train_data.index)
# train_data['HasBsmt'] = 0
# train_data.loc[train_data['TotalBsmtSF']>0,'HasBsmt'] = 1
# transform data
# train_data.loc[train_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_data['TotalBsmtSF'])
# histogram and normal probability plot
# sns.distplot(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
# scatter plot
# plt.scatter(train_data['GrLivArea'], train_data['SalePrice'])
# scatter plot
# plt.scatter(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], train_data[train_data['TotalBsmtSF']>0]['SalePrice']);