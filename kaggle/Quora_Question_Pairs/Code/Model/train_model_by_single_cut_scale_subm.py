"""
__file__

    train_model.py

__description__

    This file trains various models.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import sys
import csv
import os
import pickle
import numpy as np
import pandas as pd
import datetime
import time
import xgboost as xgb
from scipy.sparse import hstack, vstack
## sklearn
from sklearn.base import BaseEstimator
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
## hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
## keras
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import PReLU
# from keras.utils import np_utils, generic_utils
## cutomized module
from model_library_config_small_range import feat_folders, feat_names, param_spaces, int_feat

sys.path.append("../")
from param_config import config
from ml_metrics import quadratic_weighted_kappa, elementwise
from utils import *

global trial_counter
global log_handler

## libfm
libfm_exe = "../../libfm-1.40.windows/libfm.exe"

## rgf
call_exe = "../../rgf1.2/test/call_exe.pl"
rgf_exe = "../../rgf1.2/bin/rgf.exe"

output_path = "../../Output"

### global params
## you can use bagging to stabilize the predictions
bootstrap_ratio = 1
bootstrap_replacement = False
bagging_size = 1

ebc_hard_threshold = False
verbose_level = 1


#### warpper for hyperopt for logging the training reslut
# adopted from
#
def hyperopt_wrapper(param, feat_folder, feat_name):
    global trial_counter
    global log_handler
    trial_counter += 1
    start_time = time.time()
    print('start time: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # convert integer feat
    for f in int_feat:
        if f in param:
            param[f] = int(param[f])

    print("------------------------------------------------------------")
    print("Trial %d" % trial_counter)

    print("        Model")
    print("              %s" % feat_name)
    print("        Param")
    for k, v in sorted(param.items()):
        print("              %s: %s" % (k, v))
    print("        Result")
    print("                    Run      Fold      Bag      log_loss      Shape")

    ## evaluate performance
    log_loss_cv_mean, log_loss_cv_std = hyperopt_obj(param, feat_folder, feat_name, trial_counter)

    ## log
    var_to_log = [
        "%d" % trial_counter,
        "%.6f" % log_loss_cv_mean,
        "%.6f" % log_loss_cv_std,
        "%.2fm" % ((time.time() - start_time) / 60)
    ]
    for k, v in sorted(param.items()):
        var_to_log.append("%s" % v)
    writer.writerow(var_to_log)
    log_handler.flush()
    print('end time: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('spend time: %.1fm' % ((time.time() - start_time) / 60))

    return {'loss': log_loss_cv_mean, 'attachments': {'std': log_loss_cv_std}, 'status': STATUS_OK}

runs = 2
n_splits = 1
# generate kfold
def gen_kfold_by_stratifiedKFold(train_data):
    print('gen_kfold_by_stratifiedKFold...')
    X_train, labels_train = train_data[:, 1:train_data.shape[1] - 1], train_data[:, train_data.shape[1] - 1]
    year = datetime.datetime.now().year
    skf = StratifiedKFold(n_splits=5, random_state=year)
    split_data = np.zeros((n_splits), dtype=object)
    fold = 0
    for train_index, test_index in skf.split(X_train, labels_train):
        if fold == n_splits:
            break
        split_data[fold] = X_train[train_index], labels_train[train_index], X_train[test_index], labels_train[test_index]
        print('labels_train[train_index]:', np.mean(labels_train[train_index]))
        print('labels_train[test_index]:', np.mean(labels_train[test_index]))
        fold += 1
    return split_data

# 使 Train Data 和 Test Data 中正反比例一致
def set_scale_same(train_data):
    print('set_scale_same...')
    ids_train = pd.Series(train_data[:, 0].toarray().T[0])
    labels_train = pd.Series(train_data[:, train_data.shape[1] - 1].toarray().T[0])
    need_reduce = labels_train[labels_train == 1].shape[0] - np.round(0.165 * labels_train[labels_train == 0].shape[0] / (1 - 0.165))
    ids_pos = ids_train[labels_train == 1]
    split_data = np.zeros((runs, n_splits), dtype=object)
    year = datetime.datetime.now().year
    for run in range(0, runs):
        random_seed = year + 1000 * (run + 1)
        pos_ids = ids_pos.sample(int(need_reduce), random_state=random_seed)
        flags = ids_train.apply(lambda x: False if x in pos_ids else True)  # True or False
        flags = flags.index[flags]  # number
        train_data2 = train_data[flags]
        split_data[run] = gen_kfold_by_stratifiedKFold(train_data2.toarray())
    return split_data

def restore_data(X_train, labels_train, X_valid, labels_valid, ids_train, ids_valid):
    print('restore_data...')
    X_train = vstack([X_train, X_valid])
    labels_train = hstack([labels_train, labels_valid]).T
    ids_train = hstack([ids_train, ids_valid]).T
    train_data = hstack([X_train, labels_train])
    train_data = hstack([ids_train, train_data]).tocsr()
    train_data = train_data[np.argsort(train_data[:, 0].toarray().reshape(train_data.shape[0]), axis=0)] # 按第一列排序
    X_train, labels_train = train_data[:, 1:train_data.shape[1] - 1].toarray(), train_data[:, train_data.shape[1] - 1].toarray()
    return set_scale_same(train_data), X_train, labels_train

# 还原训练数据和验证数据，使其归为整体
def restore_data_by_path(path):
    print('restore_data_by_path...')
    # feat: combine feat file
    feat_train_path = "%s/valid.feat" % path
    feat_valid_path = "%s/train.feat" % path
    # info
    info_train_path = "%s/valid.info" % path
    info_valid_path = "%s/train.info" % path
    X_train, labels_train = load_svmlight_file(feat_train_path)
    X_valid, labels_valid = load_svmlight_file(feat_valid_path)
    info_train = pd.read_csv(info_train_path)
    info_valid = pd.read_csv(info_valid_path)
    ids_train = info_train['id'].values
    ids_valid = info_valid['id'].values
    return restore_data(X_train, labels_train, X_valid, labels_valid, ids_train, ids_valid)

global loaded
loaded = None
global split_data
split_data = None


def load_data(run, fold):
    print('load_data')
    global loaded
    loaded = True
    #### all the path
    path = "%s/Run%d/Fold%d" % (feat_folder, run, fold)
    save_path = "%s/Run%d/Fold%d" % (output_path, run, fold)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return restore_data_by_path(path)


#### train CV and final model with a specified parameter setting
def hyperopt_obj(param, feat_folder, feat_name, trial_counter):
    global loaded, split_data
    if loaded is None:
        split_data, X_train_all, labels_train_all = load_data(1, 1)
    log_loss_cv = np.zeros((split_data.shape[0], split_data.shape[1]), dtype=float)
    year = datetime.datetime.now().year
    # for run in range(1, split_data.shape[0] + 1):  # range(start, end)前包括后不包括
    #     for fold in range(1, split_data.shape[1] + 1):
    #         rng = np.random.RandomState(datetime.datetime.now().year + 1000 * run + 10 * fold)
    #         #### all the path
    #         path = "%s/Run%d/Fold%d" % (feat_folder, run, fold)
    #         save_path = "%s/Run%d/Fold%d" % (output_path, run, fold)
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         # feat: combine feat file
    #         feat_train_path = "%s/valid.feat" % path
    #         feat_valid_path = "%s/train.feat" % path
    #         raw_pred_valid_path = "%s/valid.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)  #
    #         rank_pred_valid_path = "%s/valid.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)  #
    #         X_train, labels_train, X_valid, labels_valid = split_data[run -1, fold -1]
    #         numTrain = X_train.shape[0]
    #         numValid = X_valid.shape[0]
    #         Y_valid = labels_valid
    #         # ## make evalerror func 评价函数
    #         # evalerror_regrank_valid = lambda preds,dtrain: evalerror_regrank_cdf(preds, dtrain, cdf_valid)
    #         # evalerror_softmax_valid = lambda preds,dtrain: evalerror_softmax_cdf(preds, dtrain, cdf_valid)
    #         # evalerror_softkappa_valid = lambda preds,dtrain: evalerror_softkappa_cdf(preds, dtrain, cdf_valid)
    #         # evalerror_ebc_valid = lambda preds,dtrain: evalerror_ebc_cdf(preds, dtrain, cdf_valid, ebc_hard_threshold)
    #         # evalerror_cocr_valid = lambda preds,dtrain: evalerror_cocr_cdf(preds, dtrain, cdf_valid)
    #
    #         ##############
    #         ## Training ##
    #         ##############
    #         ## you can use bagging to stabilize the predictions 还可以使用 bagging 来使模型更加稳定
    #         preds_bagging = np.zeros((numValid, bagging_size), dtype=float)
    #         for n in range(bagging_size):
    #             if bootstrap_replacement:
    #                 sampleSize = int(numTrain * bootstrap_ratio)  # bootstrap_ratio: 使用训练样本的比例
    #                 index_base = rng.randint(numTrain, size=sampleSize)
    #                 index_meta = [i for i in range(numTrain) if i not in index_base]
    #             else:
    #                 randnum = rng.uniform(size=numTrain)  # 产生 0-1 之间的唯一的均匀分布的随机数
    #                 index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
    #                 index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]
    #
    #             # 如果是xgb则先把数据转换成xgb需要的格式
    #             if "booster" in param:
    #                 dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)  # , weight=weight_valid
    #                 dtrain_base = xgb.DMatrix(X_train[index_base],
    #                                           label=labels_train[index_base])  # , weight=weight_train[index_base]
    #
    #                 watchlist = []
    #                 if verbose_level >= 2:
    #                     watchlist = [(dtrain_base, 'train'), (dvalid_base, 'valid')]
    #
    #             ## various models
    #             if param["task"] in ["regression", "ranking"]:
    #                 ## regression & pairwise ranking with xgboost
    #                 bst = xgb.train(param, dtrain_base, param['num_round'],
    #                                 watchlist)  # , feval=evalerror_regrank_valid
    #                 pred = bst.predict(dvalid_base)
    #
    #             if param["task"] in ["classification"]:
    #                 ## regression & pairwise ranking with xgboost
    #                 bst = xgb.train(param, dtrain_base, param['num_round'],
    #                                 watchlist)  # , feval=evalerror_regrank_valid
    #                 pred = bst.predict(dvalid_base)
    #
    #             elif param["task"] in ["softmax"]:
    #                 ## softmax regression with xgboost
    #                 bst = xgb.train(param, dtrain_base, param['num_round'],
    #                                 watchlist)  # , feval=evalerror_softmax_valid
    #                 pred = bst.predict(dvalid_base)
    #                 w = np.asarray(range(1, numValid))
    #                 pred = pred * w[np.newaxis,
    #                               :]  # np.newaxis: 插入一个维度，等价于w[np.newaxis]，这里pred是n*1矩阵，而w[np.newaxis,:]是1*n矩阵，注意w原是数组
    #                 pred = np.sum(pred, axis=1)
    #
    #             elif param["task"] in ["softkappa"]:
    #                 ## softkappa with xgboost 自定义损失函数
    #                 # obj = lambda preds, dtrain: softkappaObj(preds, dtrain, hess_scale=param['hess_scale'])
    #                 bst = xgb.train(param, dtrain_base, param['num_round'],
    #                                 watchlist)  # , obj=obj, feval=evalerror_softkappa_valid
    #                 pred = softmax(bst.predict(dvalid_base))
    #                 w = np.asarray(range(1, numValid))
    #                 pred = pred * w[np.newaxis, :]
    #                 pred = np.sum(pred, axis=1)
    #
    #             elif param["task"] in ["ebc"]:
    #                 ## ebc with xgboost 自定义损失函数
    #                 # obj = lambda preds, dtrain: ebcObj(preds, dtrain)
    #                 bst = xgb.train(param, dtrain_base, param['num_round'],
    #                                 watchlist)  # , obj=obj, feval=evalerror_ebc_valid
    #                 pred = sigmoid(bst.predict(dvalid_base))
    #                 pred = applyEBCRule(pred, hard_threshold=ebc_hard_threshold)
    #
    #             elif param["task"] in ["cocr"]:
    #                 ## cocr with xgboost 自定义损失函数
    #                 # obj = lambda preds, dtrain: cocrObj(preds, dtrain)
    #                 bst = xgb.train(param, dtrain_base, param['num_round'],
    #                                 watchlist)  # , obj=obj, feval=evalerror_cocr_valid
    #                 pred = bst.predict(dvalid_base)
    #                 pred = applyCOCRRule(pred)
    #
    #             elif param['task'] == "reg_skl_rf":
    #                 ## regression with sklearn random forest regressor
    #                 rf = RandomForestRegressor(n_estimators=param['n_estimators'],
    #                                            max_features=param['max_features'],
    #                                            n_jobs=param['n_jobs'],
    #                                            random_state=param['random_state'])
    #                 rf.fit(X_train[index_base], labels_train[index_base])  # , sample_weight=weight_train[index_base]
    #                 pred = rf.predict(X_valid)
    #
    #             elif param['task'] == "reg_skl_etr":
    #                 ## regression with sklearn extra trees regressor
    #                 etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
    #                                           max_features=param['max_features'],
    #                                           n_jobs=param['n_jobs'],
    #                                           random_state=param['random_state'])
    #                 etr.fit(X_train[index_base], labels_train[index_base])  # , sample_weight=weight_train[index_base]
    #                 pred = etr.predict(X_valid)
    #
    #             elif param['task'] == "reg_skl_gbm":
    #                 ## regression with sklearn gradient boosting regressor
    #                 gbm = GradientBoostingRegressor(n_estimators=param['n_estimators'],
    #                                                 max_features=param['max_features'],
    #                                                 learning_rate=param['learning_rate'],
    #                                                 max_depth=param['max_depth'],
    #                                                 subsample=param['subsample'],
    #                                                 random_state=param['random_state'])
    #                 gbm.fit(X_train.toarray()[index_base],
    #                         labels_train[index_base])  # , sample_weight=weight_train[index_base]
    #                 pred = gbm.predict(X_valid.toarray())
    #
    #             elif param['task'] == "clf_skl_lr":
    #                 ## classification with sklearn logistic regression
    #                 lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
    #                                         C=param['C'], fit_intercept=True, intercept_scaling=1.0,
    #                                         class_weight='auto', random_state=param['random_state'])
    #                 lr.fit(X_train[index_base], labels_train[index_base])
    #                 pred = lr.predict_proba(X_valid)
    #                 w = np.asarray(range(1, numValid))
    #                 pred = pred * w[np.newaxis, :]
    #                 pred = np.sum(pred, axis=1)
    #
    #             elif param['task'] == "reg_skl_svr":
    #                 ## regression with sklearn support vector regression
    #                 X_train, X_valid = X_train.toarray(), X_valid.toarray()
    #                 scaler = StandardScaler()
    #                 X_train[index_base] = scaler.fit_transform(X_train[index_base])
    #                 X_valid = scaler.transform(X_valid)
    #                 svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
    #                           degree=param['degree'], kernel=param['kernel'])
    #                 svr.fit(X_train[index_base], labels_train[index_base])  # , sample_weight=weight_train[index_base]
    #                 pred = svr.predict(X_valid)
    #
    #             elif param['task'] == "reg_skl_ridge":
    #                 ## regression with sklearn ridge regression
    #                 ridge = Ridge(alpha=param["alpha"], normalize=True)
    #                 ridge.fit(X_train[index_base], labels_train[index_base])  # , sample_weight=weight_train[index_base]
    #                 pred = ridge.predict(X_valid)
    #
    #             elif param['task'] == "reg_skl_lasso":
    #                 ## regression with sklearn lasso
    #                 lasso = Lasso(alpha=param["alpha"], normalize=True)
    #                 lasso.fit(X_train[index_base], labels_train[index_base])
    #                 pred = lasso.predict(X_valid)
    #
    #             elif param['task'] == 'reg_libfm':
    #                 ## regression with factorization machine (libfm)
    #                 ## to array
    #                 X_train = X_train.toarray()
    #                 X_valid = X_valid.toarray()
    #
    #                 ## scale
    #                 scaler = StandardScaler()
    #                 X_train[index_base] = scaler.fit_transform(X_train[index_base])
    #                 X_valid = scaler.transform(X_valid)
    #
    #                 ## dump feat
    #                 dump_svmlight_file(X_train[index_base], labels_train[index_base], feat_train_path + ".tmp")
    #                 dump_svmlight_file(X_valid, labels_valid, feat_valid_path + ".tmp")
    #
    #                 ## train fm
    #                 cmd = "%s -task r -train %s -test %s -out %s -dim '1,1,%d' -iter %d > libfm.log" % ( \
    #                     libfm_exe, feat_train_path + ".tmp", feat_valid_path + ".tmp", raw_pred_valid_path, \
    #                     param['dim'], param['iter'])
    #                 os.system(cmd)
    #                 os.remove(feat_train_path + ".tmp")
    #                 os.remove(feat_valid_path + ".tmp")
    #
    #                 ## extract libfm prediction
    #                 pred = np.loadtxt(raw_pred_valid_path, dtype=float)
    #                 ## labels are in [0,1,2,3]
    #                 pred += 1
    #
    #             # elif param['task'] == "reg_keras_dnn":
    #             #     ## regression with keras' deep neural networks
    #             #     model = Sequential()
    #             #     ## input layer
    #             #     model.add(Dropout(param["input_dropout"]))
    #             #     ## hidden layers
    #             #     first = True
    #             #     hidden_layers = param['hidden_layers']
    #             #     while hidden_layers > 0:
    #             #         if first:
    #             #             dim = X_train.shape[1]
    #             #             first = False
    #             #         else:
    #             #             dim = param["hidden_units"]
    #             #         model.add(Dense(dim, param["hidden_units"], init='glorot_uniform'))
    #             #         if param["batch_norm"]:
    #             #             model.add(BatchNormalization((param["hidden_units"],)))
    #             #         if param["hidden_activation"] == "prelu":
    #             #             model.add(PReLU((param["hidden_units"],)))
    #             #         else:
    #             #             model.add(Activation(param['hidden_activation']))
    #             #         model.add(Dropout(param["hidden_dropout"]))
    #             #         hidden_layers -= 1
    #             #
    #             #     ## output layer
    #             #     model.add(Dense(param["hidden_units"], 1, init='glorot_uniform'))
    #             #     model.add(Activation('linear'))
    #             #
    #             #     ## loss
    #             #     model.compile(loss='mean_squared_error', optimizer="adam")
    #             #
    #             #     ## to array
    #             #     X_train = X_train.toarray()
    #             #     X_valid = X_valid.toarray()
    #             #
    #             #     ## scale
    #             #     scaler = StandardScaler()
    #             #     X_train[index_base] = scaler.fit_transform(X_train[index_base])
    #             #     X_valid = scaler.transform(X_valid)
    #             #
    #             #     ## train
    #             #     model.fit(X_train[index_base], labels_train[index_base],
    #             #                 nb_epoch=param['nb_epoch'], batch_size=param['batch_size'],
    #             #                 validation_split=0, verbose=0)
    #             #
    #             #     ##prediction
    #             #     pred = model.predict(X_valid, verbose=0)
    #             #     pred.shape = (X_valid.shape[0],)
    #
    #             elif param['task'] == "reg_rgf":
    #                 ## regression with regularized greedy forest (rgf)
    #                 ## to array
    #                 X_train, X_valid = X_train.toarray(), X_valid.toarray()
    #
    #                 train_x_fn = feat_train_path + ".x"
    #                 train_y_fn = feat_train_path + ".y"
    #                 valid_x_fn = feat_valid_path + ".x"
    #                 valid_pred_fn = feat_valid_path + ".pred"
    #
    #                 model_fn_prefix = "rgf_model"
    #
    #                 np.savetxt(train_x_fn, X_train[index_base], fmt="%.6f", delimiter='\t')
    #                 np.savetxt(train_y_fn, labels_train[index_base], fmt="%d", delimiter='\t')
    #                 np.savetxt(valid_x_fn, X_valid, fmt="%.6f", delimiter='\t')
    #                 # np.savetxt(valid_y_fn, labels_valid, fmt="%d", delimiter='\t')
    #
    #
    #                 pars = [
    #                     "train_x_fn=", train_x_fn, "\n",
    #                     "train_y_fn=", train_y_fn, "\n",
    #                     # "train_w_fn=",weight_train_path,"\n",
    #                     "model_fn_prefix=", model_fn_prefix, "\n",
    #                     "reg_L2=", param['reg_L2'], "\n",
    #                     # "reg_depth=", 1.01, "\n",
    #                     "algorithm=", "RGF", "\n",
    #                     "loss=", "LS", "\n",
    #                     # "opt_interval=", 100, "\n",
    #                     "valid_interval=", param['max_leaf_forest'], "\n",
    #                     "max_leaf_forest=", param['max_leaf_forest'], "\n",
    #                     "num_iteration_opt=", param['num_iteration_opt'], "\n",
    #                     "num_tree_search=", param['num_tree_search'], "\n",
    #                     "min_pop=", param['min_pop'], "\n",
    #                     "opt_interval=", param['opt_interval'], "\n",
    #                     "opt_stepsize=", param['opt_stepsize'], "\n",
    #                     "NormalizeTarget"
    #                 ]
    #                 pars = "".join([str(p) for p in pars])
    #
    #                 rfg_setting_train = "./rfg_setting_train"
    #                 with open(rfg_setting_train + ".inp", "wb") as f:
    #                     f.write(pars)
    #
    #                 ## train fm
    #                 cmd = "perl %s %s train %s >> rgf.log" % (
    #                     call_exe, rgf_exe, rfg_setting_train)
    #                 # print cmd
    #                 os.system(cmd)
    #
    #                 model_fn = model_fn_prefix + "-01"
    #                 pars = [
    #                     "test_x_fn=", valid_x_fn, "\n",
    #                     "model_fn=", model_fn, "\n",
    #                     "prediction_fn=", valid_pred_fn
    #                 ]
    #
    #                 pars = "".join([str(p) for p in pars])
    #
    #                 rfg_setting_valid = "./rfg_setting_valid"
    #                 with open(rfg_setting_valid + ".inp", "wb") as f:
    #                     f.write(pars)
    #                 cmd = "perl %s %s predict %s >> rgf.log" % (
    #                     call_exe, rgf_exe, rfg_setting_valid)
    #                 # print cmd
    #                 os.system(cmd)
    #
    #                 pred = np.loadtxt(valid_pred_fn, dtype=float)
    #
    #             ## weighted averageing over different models
    #             pred_valid = pred
    #             ## this bagging iteration
    #             preds_bagging[:, n] = pred_valid  # preds_bagging的第n+1列为pred_valid
    #             pred_raw = np.mean(preds_bagging[:, :(n + 1)], axis=1)  # 按行（同行多列）进行平均值
    #             # pred_rank = pred_raw.argsort().argsort()    # argsort: 获取排序的索引值（index），但索引值本身不排序，第二次是归位
    #             # pred_score, cutoff = getScore(pred_rank, cdf_valid, valid=True) # 根据cdf来生成分数
    #             # kappa_valid = quadratic_weighted_kappa(pred_score, Y_valid) # 计算kappa分数
    #             log_loss_valid = elementwise.log_loss(Y_valid, pred_raw)
    #             print('Y_valid mean:', np.mean(Y_valid))
    #             print('pred_raw mean:', np.mean(pred_raw))
    #             if (n + 1) != bagging_size:
    #                 print("              {:>3}   {:>3}   {:>3}   {:>6}   {} x {}".format(
    #                     run, fold, n + 1, np.round(log_loss_valid, 6), X_train.shape[0], X_train.shape[1]))
    #             else:
    #                 print("                    {:>3}       {:>3}      {:>3}    {:>8}  {} x {}".format(
    #                     run, fold, n + 1, np.round(log_loss_valid, 6), X_train.shape[0], X_train.shape[1]))
    #         log_loss_cv[run - 1, fold - 1] = log_loss_valid
    #         ## save this prediction 保存的是单行的预测值
    #         dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_raw})
    #         dfPred.to_csv(raw_pred_valid_path, index=False, header=True, columns=["target", "prediction"])
    #         # save this prediction 保存的是根据预测值排序之后，然后使用cdf来生成的预测值
    #         # dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_rank})
    #         # dfPred.to_csv(rank_pred_valid_path, index=False, header=True, columns=["target", "prediction"])
    #
    # log_loss_cv_mean = np.mean(log_loss_cv)
    # log_loss_cv_std = np.std(log_loss_cv)
    # if verbose_level >= 1:
    #     print("              Mean: %.6f" % log_loss_cv_mean)
    #     print("              Std: %.6f" % log_loss_cv_std)

    ####################
    #### Retraining ####
    ####################
    #### all the path
    path = "%s/All" % (feat_folder)
    save_path = "%s/All" % output_path
    subm_path = "%s/Subm" % output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(subm_path):
        os.makedirs(subm_path)
    # feat
    feat_train_path = "%s/train.feat" % path
    feat_test_path = "%s/test.feat" % path
    # weight
    # weight_train_path = "%s/train.feat.weight" % path
    # info
    info_train_path = "%s/train.info" % path
    info_test_path = "%s/test.info" % path
    # cdf
    # cdf_test_path = "%s/test.cdf" % path
    # raw prediction path (rank)
    raw_pred_test_path = "%s/test.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
    rank_pred_test_path = "%s/test.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
    # submission path (is_duplicate as in [0, 1])
    # subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % (subm_path, feat_name, trial_counter, log_loss_cv_mean, log_loss_cv_std)

    #### load data
    ## load feat
    # X_train, labels_train = load_svmlight_file(feat_train_path)
    X_train, labels_train = X_train_all, labels_train_all
    X_test, labels_test = load_svmlight_file(feat_test_path)
    # if X_test.shape[1] < X_train.shape[1]:
    #     X_test = hstack([X_test, np.zeros((X_test.shape[0], X_train.shape[1]-X_test.shape[1]))])
    # elif X_test.shape[1] > X_train.shape[1]:
    #     X_train = hstack([X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))])
    # X_train = X_train.tocsr()
    # X_test = X_test.tocsr()
    # 缩小训练数据比例以使训练数据和测试数据比例一致

    ## load train weight
    # weight_train = np.loadtxt(weight_train_path, dtype=float)
    ## load test info
    info_train = pd.read_csv(info_train_path)
    numTrain = X_train.shape[0]
    info_test = pd.read_csv(info_test_path)
    numTest = info_test.shape[0]
    id_test = info_test["test_id"]
    numValid = info_test.shape[0]
    ## load cdf
    # cdf_test = np.loadtxt(cdf_test_path, dtype=float)
    # ## 评价函数
    # evalerror_regrank_test = lambda preds,dtrain: evalerror_regrank_cdf(preds, dtrain, cdf_test)
    # evalerror_softmax_test = lambda preds,dtrain: evalerror_softmax_cdf(preds, dtrain, cdf_test)
    # evalerror_softkappa_test = lambda preds,dtrain: evalerror_softkappa_cdf(preds, dtrain, cdf_test)
    # evalerror_ebc_test = lambda preds,dtrain: evalerror_ebc_cdf(preds, dtrain, cdf_test, ebc_hard_threshold)
    # evalerror_cocr_test = lambda preds,dtrain: evalerror_cocr_cdf(preds, dtrain, cdf_test)
    rng = np.random.RandomState(datetime.datetime.now().year + 1000 * 1 + 10 * 1)
    ## bagging
    preds_bagging = np.zeros((numTest, bagging_size), dtype=float)
    for n in range(bagging_size):
        if bootstrap_replacement:
            sampleSize = int(numTrain*bootstrap_ratio)
            #index_meta = rng.randint(numTrain, size=sampleSize)
            #index_base = [i for i in range(numTrain) if i not in index_meta]
            index_base = rng.randint(numTrain, size=sampleSize)
            index_meta = [i for i in range(numTrain) if i not in index_base]
        else:
            randnum = rng.uniform(size=numTrain)
            index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
            index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]

        # 如果是xgb则先把数据转换成xgb需要的格式
        if "booster" in param:
            dtest = xgb.DMatrix(X_test, label=labels_test)
            dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base])   # , weight=weight_train[index_base]

            watchlist = []
            if verbose_level >= 2:
                watchlist  = [(dtrain, 'train')]

        ## train
        if param["task"] in ["regression", "ranking"]:
            bst = xgb.train(param, dtrain, param['num_round'], watchlist)   # , feval=evalerror_regrank_test
            pred = bst.predict(dtest)

        if param["task"] in ["classification"]:
            ## regression & pairwise ranking with xgboost
            bst = xgb.train(param, dtrain, param['num_round'], watchlist)  # , feval=evalerror_softmax_test
            pred = bst.predict(dtest)

        elif param["task"] in ["softmax"]:
            bst = xgb.train(param, dtrain, param['num_round'], watchlist)   # , feval=evalerror_softmax_test
            pred = bst.predict(dtest)
            w = np.asarray(range(1,numValid))
            pred = pred * w[np.newaxis,:]
            pred = np.sum(pred, axis=1)

        elif param["task"] in ["softkappa"]:
            #  自定义损失函数
            # obj = lambda preds, dtrain: softkappaObj(preds, dtrain, hess_scale=param['hess_scale'])
            bst = xgb.train(param, dtrain, param['num_round'], watchlist)   # , obj=obj, feval=evalerror_softkappa_test
            pred = softmax(bst.predict(dtest))
            w = np.asarray(range(1,numValid))
            pred = pred * w[np.newaxis,:]
            pred = np.sum(pred, axis=1)

        elif param["task"]  in ["ebc"]:
            #  自定义损失函数
            # obj = lambda preds, dtrain: ebcObj(preds, dtrain)
            bst = xgb.train(param, dtrain, param['num_round'], watchlist)   # , obj=obj, feval=evalerror_ebc_test
            pred = sigmoid(bst.predict(dtest))
            pred = applyEBCRule(pred, hard_threshold=ebc_hard_threshold)

        elif param["task"]  in ["cocr"]:
            #  自定义损失函数
            obj = lambda preds, dtrain: cocrObj(preds, dtrain)
            bst = xgb.train(param, dtrain, param['num_round'], watchlist)   # , obj=obj, feval=evalerror_cocr_test
            pred = bst.predict(dtest)
            pred = applyCOCRRule(pred)

        elif param['task'] == "reg_skl_rf":
            ## random forest regressor
            rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                                       max_features=param['max_features'],
                                       n_jobs=param['n_jobs'],
                                       random_state=param['random_state'])
            rf.fit(X_train[index_base], labels_train[index_base]) # , sample_weight=weight_train[index_base]
            pred = rf.predict(X_test)

        elif param['task'] == "reg_skl_etr":
            ## extra trees regressor
            etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                                      max_features=param['max_features'],
                                      n_jobs=param['n_jobs'],
                                      random_state=param['random_state'])
            etr.fit(X_train[index_base], labels_train[index_base])    # , sample_weight=weight_train[index_base]
            pred = etr.predict(X_test)

        elif param['task'] == "reg_skl_gbm":
            ## gradient boosting regressor
            gbm = GradientBoostingRegressor(n_estimators=param['n_estimators'],
                                            max_features=param['max_features'],
                                            learning_rate=param['learning_rate'],
                                            max_depth=param['max_depth'],
                                            subsample=param['subsample'],
                                            random_state=param['random_state'])
            gbm.fit(X_train.toarray()[index_base], labels_train[index_base])  #, sample_weight=weight_train[index_base]
            pred = gbm.predict(X_test.toarray())

        elif param['task'] == "clf_skl_lr":
            lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                    C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                    class_weight='auto', random_state=param['random_state'])
            lr.fit(X_train[index_base], labels_train[index_base])
            pred = lr.predict_proba(X_test)
            w = np.asarray(range(1,numValid))
            pred = pred * w[np.newaxis,:]
            pred = np.sum(pred, axis=1)

        elif param['task'] == "reg_skl_svr":
            ## regression with sklearn support vector regression
            X_train, X_test = X_train.toarray(), X_test.toarray()
            scaler = StandardScaler()
            X_train[index_base] = scaler.fit_transform(X_train[index_base])
            X_test = scaler.transform(X_test)
            svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
                                    degree=param['degree'], kernel=param['kernel'])
            svr.fit(X_train[index_base], labels_train[index_base])    # , sample_weight=weight_train[index_base]
            pred = svr.predict(X_test)

        elif param['task'] == "reg_skl_ridge":
            ridge = Ridge(alpha=param["alpha"], normalize=True)
            ridge.fit(X_train[index_base], labels_train[index_base])  # , sample_weight=weight_train[index_base]
            pred = ridge.predict(X_test)

        elif param['task'] == "reg_skl_lasso":
            lasso = Lasso(alpha=param["alpha"], normalize=True)
            lasso.fit(X_train[index_base], labels_train[index_base])
            pred = lasso.predict(X_test)

        elif param['task'] == 'reg_libfm':
            ## to array
            X_train, X_test = X_train.toarray(), X_test.toarray()

            ## scale
            scaler = StandardScaler()
            X_train[index_base] = scaler.fit_transform(X_train[index_base])
            X_test = scaler.transform(X_test)

            ## dump feat
            dump_svmlight_file(X_train[index_base], labels_train[index_base], feat_train_path+".tmp")
            dump_svmlight_file(X_test, labels_test, feat_test_path+".tmp")

            ## train fm
            cmd = "%s -task r -train %s -test %s -out %s -dim '1,1,%d' -iter %d > libfm.log" % ( \
                        libfm_exe, feat_train_path+".tmp", feat_test_path+".tmp", raw_pred_test_path, \
                        param['dim'], param['iter'])
            os.system(cmd)
            os.remove(feat_train_path+".tmp")
            os.remove(feat_test_path+".tmp")

            ## extract libfm prediction
            pred = np.loadtxt(raw_pred_test_path, dtype=float)
            ## labels are in [0,1,2,3]
            pred += 1

        # elif param['task'] == "reg_keras_dnn":
        #     ## regression with keras deep neural networks
        #     model = Sequential()
        #     ## input layer
        #     model.add(Dropout(param["input_dropout"]))
        #     ## hidden layers
        #     first = True
        #     hidden_layers = param['hidden_layers']
        #     while hidden_layers > 0:
        #         if first:
        #             dim = X_train.shape[1]
        #             first = False
        #         else:
        #             dim = param["hidden_units"]
        #         model.add(Dense(dim, param["hidden_units"], init='glorot_uniform'))
        #         if param["batch_norm"]:
        #             model.add(BatchNormalization((param["hidden_units"],)))
        #         if param["hidden_activation"] == "prelu":
        #             model.add(PReLU((param["hidden_units"],)))
        #         else:
        #             model.add(Activation(param['hidden_activation']))
        #         model.add(Dropout(param["hidden_dropout"]))
        #         hidden_layers -= 1
        #
        #     ## output layer
        #     model.add(Dense(param["hidden_units"], 1, init='glorot_uniform'))
        #     model.add(Activation('linear'))
        #
        #     ## loss
        #     model.compile(loss='mean_squared_error', optimizer="adam")
        #
        #     ## to array
        #     X_train = X_train.toarray()
        #     X_test = X_test.toarray()
        #
        #     ## scale
        #     scaler = StandardScaler()
        #     X_train[index_base] = scaler.fit_transform(X_train[index_base])
        #     X_test = scaler.transform(X_test)
        #
        #     ## train
        #     model.fit(X_train[index_base], labels_train[index_base],
        #                 nb_epoch=param['nb_epoch'], batch_size=param['batch_size'], verbose=0)
        #
        #     ##prediction
        #     pred = model.predict(X_test, verbose=0)
        #     pred.shape = (X_test.shape[0],)

        elif param['task'] == "reg_rgf":
            ## to array
            X_train, X_test = X_train.toarray(), X_test.toarray()

            train_x_fn = feat_train_path+".x"
            train_y_fn = feat_train_path+".y"
            test_x_fn = feat_test_path+".x"
            test_pred_fn = feat_test_path+".pred"

            model_fn_prefix = "rgf_model"

            np.savetxt(train_x_fn, X_train[index_base], fmt="%.6f", delimiter='\t')
            np.savetxt(train_y_fn, labels_train[index_base], fmt="%d", delimiter='\t')
            np.savetxt(test_x_fn, X_test, fmt="%.6f", delimiter='\t')
            # np.savetxt(valid_y_fn, labels_valid, fmt="%d", delimiter='\t')


            pars = [
                "train_x_fn=",train_x_fn,"\n",
                "train_y_fn=",train_y_fn,"\n",
                #"train_w_fn=",weight_train_path,"\n",
                "model_fn_prefix=",model_fn_prefix,"\n",
                "reg_L2=", param['reg_L2'], "\n",
                #"reg_depth=", 1.01, "\n",
                "algorithm=","RGF","\n",
                "loss=","LS","\n",
                "test_interval=", param['max_leaf_forest'],"\n",
                "max_leaf_forest=", param['max_leaf_forest'],"\n",
                "num_iteration_opt=", param['num_iteration_opt'], "\n",
                "num_tree_search=", param['num_tree_search'], "\n",
                "min_pop=", param['min_pop'], "\n",
                "opt_interval=", param['opt_interval'], "\n",
                "opt_stepsize=", param['opt_stepsize'], "\n",
                "NormalizeTarget"
            ]
            pars = "".join([str(p) for p in pars])

            rfg_setting_train = "./rfg_setting_train"
            with open(rfg_setting_train+".inp", "wb") as f:
                f.write(pars)

            ## train fm
            cmd = "perl %s %s train %s >> rgf.log" % (
                    call_exe, rgf_exe, rfg_setting_train)
            #print cmd
            os.system(cmd)


            model_fn = model_fn_prefix + "-01"
            pars = [
                "test_x_fn=",test_x_fn,"\n",
                "model_fn=", model_fn,"\n",
                "prediction_fn=", test_pred_fn
            ]

            pars = "".join([str(p) for p in pars])

            rfg_setting_test = "./rfg_setting_test"
            with open(rfg_setting_test+".inp", "wb") as f:
                f.write(pars)
            cmd = "perl %s %s predict %s >> rgf.log" % (
                    call_exe, rgf_exe, rfg_setting_test)
            #print cmd
            os.system(cmd)

            pred = np.loadtxt(test_pred_fn, dtype=float)

        ## weighted averageing over different models
        pred_test = pred
        preds_bagging[:,n] = pred_test
    pred_raw = np.mean(preds_bagging, axis=1)
    # pred_rank = pred_raw.argsort().argsort()
    #
    ## write
    output = pd.DataFrame({"test_id": id_test, "is_duplicate": pred_raw})
    output.to_csv(raw_pred_test_path, index=False)

    ## write
    # output = pd.DataFrame({"test_id": id_test, "is_duplicate": pred_rank})
    # output.to_csv(rank_pred_test_path, index=False)

    ## write score
    # pred_score = getScore(pred, cdf_test)
    # output = pd.DataFrame({"test_id": id_test, "is_duplicate": pred_score})
    # output.to_csv(subm_path, index=False)
    #"""

    return log_loss_cv_mean, log_loss_cv_std


####################
## Model Buliding ##
####################

def check_model(models, feat_name):
    if models == "all":
        return True
    for model in models:
        if model in feat_name:
            return True
    return False


if __name__ == "__main__":
    # 训练的模型需要传入说明，否则终止训练
    # specified_models = sys.argv[1:]
    specified_models = ['bclf_xgb_tree']
    if len(specified_models) == 0:
        print("You have to specify which model to train.\n")
        print("Usage: python ./train_model_library_lsa.py model1 model2 model3 ...\n")  # cmd中启动命令
        print("Example: python ./train_model_library_lsa.py reg_skl_ridge reg_skl_lasso reg_skl_svr\n")
        print("See model_library_config_lsa.py for a list of available models (i.e., Model@model_name)")
        sys.exit()
    # 日志，关键点，因为记录了训练过程，可以通过该日志了解训练的情况，对分析模型非常有用
    log_path = "%s/Log" % output_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # zip是你内置函数，可以同时遍历多个变量，以最短的那个作为终止
    for feat_name, feat_folder in zip(feat_names, feat_folders):
        if not check_model(specified_models, feat_name):
            continue
        param_space = param_spaces[feat_name]
        # """

        # 把文件header写入日志
        log_file = "%s/%s_hyperopt_[Run_Time@%s].log" % (log_path, feat_name, time.strftime("%Y%m%d%H%M%S", time.localtime()))
        log_handler = open(log_file, 'w')
        writer = csv.writer(log_handler)
        headers = ['trial_counter', 'log_loss_mean', 'log_loss_std', 'spend_time']
        for k, v in sorted(param_space.items()):
            headers.append(k)
        print(headers)
        writer.writerow(headers)
        log_handler.flush()

        print("************************************************************")
        print("Search for the best params")
        # global trial_counter
        trial_counter = 0
        trials = Trials()
        # lambda在这一步并不会运行，只是定义一个函数而已
        objective = lambda p: hyperopt_wrapper(p, feat_folder, feat_name)
        # objective放到fmin中，会被调用，且传进三个参数
        best_params = fmin(objective, param_space, algo=tpe.suggest,
                           trials=trials, max_evals=param_space["max_evals"])
        for f in int_feat:
            if f in best_params:
                best_params[f] = int(best_params[f])
        print("************************************************************")
        print("Best params")
        for k, v in best_params.items():
            print("        %s: %s" % (k, v))
        trial_log_losss = -np.asarray(trials.losses(), dtype=float)
        best_log_loss_mean = max(trial_log_losss)
        ind = np.where(trial_log_losss == best_log_loss_mean)[0][0]
        best_log_loss_std = trials.trial_attachments(trials.trials[ind])['std']
        print("log_loss stats")
        print("        Mean: %.6f\n        Std: %.6f" % (best_log_loss_mean, best_log_loss_std))