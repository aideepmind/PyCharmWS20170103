
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
from sklearn.model_selection import train_test_split
sys.path.append("../")
from param_config import config


if __name__ == "__main__":

    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = dill.load(f)

    skf = np.zeros((config.n_runs, config.n_folds), dtype=object) # 因为TimeSeriesSplit是按时间来切割的，所以没有随机因子，所以n_runs默认1
    year = datetime.datetime.now().year
    for run in range(config.n_runs):
        for fold in range(config.n_folds):
            random_seed = year + 1000 * (run + 1) + 100 * (fold + 1)
            train_data, valid_data = train_test_split(dfTrain, test_size=1.0 / config.n_folds, random_state=random_seed)
            train_index, valid_index = train_data.index, valid_data.index
            skf[run][fold] = fold, (train_index, valid_index)
            fold += 1
            print("================================")
            print("Index for run: %s, fold: %s" % (run + 1, fold))
            print("Train (num = %s)" % len(train_index))
            print(train_index[:10])
            print("Valid (num = %s)" % len(valid_index))
            print(valid_index[:10])
    with open("%s/random.pkl" % (config.data_folder), "wb") as f:
        dill.dump(skf, f, -1)