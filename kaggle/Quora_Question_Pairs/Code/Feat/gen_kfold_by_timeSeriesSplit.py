
"""
__file__

    gen_kfold_by_stratifiedKFold.py

__description__

    This file generates the StratifiedKFold indices which will be kept fixed in
    ALL the following model building parts.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import sys
import dill
import datetime
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
sys.path.append("../")
from param_config import config


if __name__ == "__main__":

    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = dill.load(f)

    skf = np.zeros((config.n_runs, config.n_folds), dtype=object) # 因为TimeSeriesSplit是按时间来切割的，所以没有随机因子，所以n_runs默认1
    for run in range(config.n_runs):
        tss = TimeSeriesSplit(n_splits=config.n_folds)    # n_splits: default = 3
        fold = 0
        for train_index, valid_index in tss.split(dfTrain):
            skf[run][fold] = fold, (train_index, valid_index)
            fold += 1
            print("================================")
            print("Index for run: %s, fold: %s" % (run + 1, fold))
            print("Train (num = %s)" % len(train_index))
            print(train_index[:10])
            print("Valid (num = %s)" % len(valid_index))
            print(valid_index[:10])
    with open("%s/timeSeriesSplit.pkl" % (config.data_folder), "wb") as f:
        dill.dump(skf, f, -1)