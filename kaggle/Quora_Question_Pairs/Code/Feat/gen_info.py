
"""
__file__

    gen_info.py

__description__

    This file generates the following info for each run and fold, and for the entire training and testing set.

        1. training and validation/testing data

        2. sample weight

        3. cdf of the median_relevance
        
        4. the group info for pairwise ranking in XGBoost

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import os
import sys
import pickle
import dill
import numpy as np
import pandas as pd
sys.path.append("../")
from param_config import config

def gen_info(feat_path_name):
    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = dill.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = dill.load(f)
    dfTrain_original = pd.read_csv(config.original_train_data_path).fillna("")
    dfTest_original = pd.read_csv(config.original_test_data_path).fillna("")
    ## insert fake label for test
    dfTest_original["is_duplicate"] = np.ones((dfTest_original.shape[0]))
    # dfTest_original["relevance_variance"] = np.zeros((dfTest_original.shape[0]))
    # change it to zero-based for classification
    Y = dfTrain_original["is_duplicate"].values

    ## load pre-defined stratified k-fold index
    with open(config.cv_info_path, "rb") as f:
        skf = pickle.load(f, encoding='latin1')
        
    #######################
    ## Generate Features ##
    #######################
    print("Generate info...")
    print("For cross-validation...")
    for run in range(config.n_runs):
        ## use 33% for training and 67 % for validation
        ## so we switch train_index and valid_index
        for fold, (train_index, valid_index) in skf[run]:
            print("Run: %d, Fold: %d" % (run + 1, fold + 1))
            path = "%s/%s/Run%d/Fold%d" % (config.feat_folder, feat_path_name, run + 1, fold + 1)
            if not os.path.exists(path):
                os.makedirs(path)
            ##########################
            ## get and dump weights ##
            ##########################
            # raise_to = 0.5
            # var = dfTrain["relevance_variance"].values
            # max_var = np.max(var[train_index]**raise_to)
            # weight = (1 + np.power(((max_var - var**raise_to) / max_var),1)) / 2.
            # # weight = (max_var - var**raise_to) / max_var, weight 在 1/2 和 1 之间，方差（多位rater进行打分不一致所产生的）越大，权重越小，当方差为0时，权重为1
            # np.savetxt("%s/train.feat.weight" % path, weight[train_index], fmt="%.6f")
            # np.savetxt("%s/valid.feat.weight" % path, weight[valid_index], fmt="%.6f")

            #############################    
            ## get and dump group info ##
            #############################
            np.savetxt("%s/train.feat.group" % path, [len(train_index)], fmt="%d")
            np.savetxt("%s/valid.feat.group" % path, [len(valid_index)], fmt="%d")
            
            ######################
            ## get and dump cdf ##
            ######################
            hist = np.bincount(Y[train_index])  # 统计数出现的频率，类似直方图hist
            overall_cdf_valid = np.cumsum(hist) / float(sum(hist))  # CDF（cumulative distribution function）累积分布函数
            np.savetxt("%s/valid.cdf" % path, overall_cdf_valid)
                
            #############################
            ## dump all the other info ##
            #############################
            dfTrain_original.iloc[train_index].to_csv("%s/train.info" % path, index=False, header=True, encoding='utf-8')
            dfTrain_original.iloc[valid_index].to_csv("%s/valid.info" % path, index=False, header=True, encoding='utf-8')
    print("Done.")

    print("For training and testing...")
    path = "%s/%s/All" % (config.feat_folder, feat_path_name)
    if not os.path.exists(path):
        os.makedirs(path)

    # ## weight
    # max_var = np.max(var**raise_to)
    # weight = (1 + np.power(((max_var - var**raise_to) / max_var),1)) / 2.
    # np.savetxt("%s/train.feat.weight" % path, weight, fmt="%.6f")
    
    ## group
    np.savetxt("%s/%s/All/train.feat.group" % (config.feat_folder, feat_path_name), [dfTrain.shape[0]], fmt="%d")
    np.savetxt("%s/%s/All/test.feat.group" % (config.feat_folder, feat_path_name), [dfTest.shape[0]], fmt="%d")
    ## cdf 这个地方需要改动，因为Train Set与Test Set的正反例比例不一致，是线性下降的，如果修改代码呢？
    # hist_full = np.bincount(Y)
    # print (hist_full) / float(sum(hist_full))
    # overall_cdf_full = np.cumsum(hist_full) / float(sum(hist_full))
    overall_cdf_full = [0.835, 1.0]
    np.savetxt("%s/%s/All/test.cdf" % (config.feat_folder, feat_path_name), overall_cdf_full)
    ## info        
    dfTrain_original.to_csv("%s/%s/All/train.info" % (config.feat_folder, feat_path_name), index=False, header=True)
    dfTest_original.to_csv("%s/%s/All/test.info" % (config.feat_folder, feat_path_name), index=False, header=True)
    
    print("All Done.")