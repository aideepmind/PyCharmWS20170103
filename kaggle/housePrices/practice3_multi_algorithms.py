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

# 大体思路：
# 多种特征工程
# 多种回归算法（所有的回归算法）
# 三层：一层是多种模型的结果；二层是利用一层结果再训练，得到结果；三层是利用二层结果再训练


# input
train_data = pd.read_csv('kaggle/housePrices/input/train.csv')
test_data = pd.read_csv('kaggle/housePrices/input/test.csv')
all_data = pd.concat((train_data[test_data.columns], test_data), ignore_index=True)

lasso_data_train = pd.read_csv('kaggle/housePrices/temp/lasso_train_result.csv')
ridge_data_train = pd.read_csv('kaggle/housePrices/temp/ridge_train_result.csv')
ard_data_train = pd.read_csv('kaggle/housePrices/temp/ard_train_result.csv')
br_data_train = pd.read_csv('kaggle/housePrices/temp/br_train_result.csv')
en_data_train = pd.read_csv('kaggle/housePrices/temp/en_train_result.csv')
hr_data_train = pd.read_csv('kaggle/housePrices/temp/hr_train_result.csv')
ll_data_train = pd.read_csv('kaggle/housePrices/temp/ll_train_result.csv')
lr_data_train = pd.read_csv('kaggle/housePrices/temp/lr_train_result.csv')
lars_data_train = pd.read_csv('kaggle/housePrices/temp/lars_train_result.csv')
krr_data_train = pd.read_csv('kaggle/housePrices/temp/krr_train_result.csv')
gbm_data_train = pd.read_csv('kaggle/housePrices/temp/gbm_train_result.csv')
rf_data_train = pd.read_csv('kaggle/housePrices/temp/rf_train_result.csv')
xbg_data_train = pd.read_csv('kaggle/housePrices/temp/xgb_train_result.csv')

lasso_data_test = pd.read_csv('kaggle/housePrices/temp/lasso_test_result.csv')
ridge_data_test = pd.read_csv('kaggle/housePrices/temp/ridge_test_result.csv')
ard_data_test = pd.read_csv('kaggle/housePrices/temp/ard_test_result.csv')
br_data_test = pd.read_csv('kaggle/housePrices/temp/br_test_result.csv')
en_data_test = pd.read_csv('kaggle/housePrices/temp/en_test_result.csv')
hr_data_test = pd.read_csv('kaggle/housePrices/temp/hr_test_result.csv')
ll_data_test = pd.read_csv('kaggle/housePrices/temp/ll_test_result.csv')
lr_data_test = pd.read_csv('kaggle/housePrices/temp/lr_test_result.csv')
lars_data_test = pd.read_csv('kaggle/housePrices/temp/lars_test_result.csv')
krr_data_test = pd.read_csv('kaggle/housePrices/temp/krr_test_result.csv')
gbm_data_test = pd.read_csv('kaggle/housePrices/temp/gbm_test_result.csv')
rf_data_test = pd.read_csv('kaggle/housePrices/temp/rf_test_result.csv')
xbg_data_test = pd.read_csv('kaggle/housePrices/temp/xgb_test_result.csv')

second_train_x = pd.DataFrame({
    'lasso': lasso_data_train['SalePrice'],
    'ridge': ridge_data_train['SalePrice'],
    'ard': ard_data_train['SalePrice'],
    'br': br_data_train['SalePrice'],
    'en': en_data_train['SalePrice'],
    'hr': hr_data_train['SalePrice'],
    'll': ll_data_train['SalePrice'],
    'lr': lr_data_train['SalePrice'],
    'lars': lars_data_train['SalePrice'],
    'krr': krr_data_train['SalePrice'],
    'gbm': gbm_data_train['SalePrice'],
    'rf': rf_data_train['SalePrice'],
    'xbg': xbg_data_train['SalePrice']
})

second_test_x = pd.DataFrame({
    'lasso': lasso_data_test['SalePrice'],
    'ridge': ridge_data_test['SalePrice'],
    'ard': ard_data_test['SalePrice'],
    'br': br_data_test['SalePrice'],
    'en': en_data_test['SalePrice'],
    'hr': hr_data_test['SalePrice'],
    'll': ll_data_test['SalePrice'],
    'lr': lr_data_test['SalePrice'],
    'lars': lars_data_test['SalePrice'],
    'krr': krr_data_test['SalePrice'],
    'gbm': gbm_data_test['SalePrice'],
    'rf': rf_data_test['SalePrice'],
    'xbg': xbg_data_test['SalePrice']
})

# 关系图
# corrmat = second_train_x.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)

second_train_x = np.log1p(second_train_x)
second_test_x = np.log1p(second_test_x)
train_y = np.log1p(train_data['SalePrice'][train_data[train_data['Id'].apply(lambda id: id in lasso_data_train['Id'].values) == True].index])

# 使用xgb训练
# max_depth=3, learning_rate=0.1, n_estimators=100,
#                  silent=True, objective="reg:linear",
#                  nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
#                  subsample=1, colsample_bytree=1, colsample_bylevel=1,
#                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#                  base_score=0.5, seed=0, missing=None
param_grid = {
    'learning_rate': [0.1]}
xgb = GridSearchCV(
    estimator=XGBRegressor(
        n_estimators=2000,
        learning_rate=0.08,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.6,
        colsample_bytree=0.6,
        objective='reg:linear',
        # scale_pos_weight=1,
        # seed=27,
        nthread=4),
    param_grid=param_grid,
    n_jobs=2,
    iid=False,
    cv=5)
xgb.fit(second_train_x, train_y)
xgb.grid_scores_, xgb.best_params_, xgb.best_score_
print(np.sqrt(metrics.mean_squared_error(train_y, xgb.predict(second_train_x))))


preds_xgb = np.expm1(xgb.predict(second_test_x))
result = pd.DataFrame({"Id": test_data['Id'], "SalePrice": preds_xgb})
result.to_csv("kaggle/housePrices/temp/multi_test_result.csv", index=False)