
"""
__file__

    preprocess.py

__description__

    This file preprocesses data.

__author__

    Chenglong Chen
    
"""

import sys
import pickle
import dill
import numpy as np
import pandas as pd
from nlp_utils import clean_text, pos_tag_text
sys.path.append("../")
from param_config import config

###############
## Load Data ##
###############
print("Load data...")

dfTrain = pd.read_csv(config.original_train_data_path).fillna("")   # , encoding='utf-8'
# dfTest = pd.read_csv(config.original_test_data_path).fillna("")
# number of train/test samples
num_train = dfTrain.shape[0]  # , num_test , dfTest.shape[0]

print("Done.")


######################
## Pre-process Data ##
######################
print("Pre-process data...")

## insert sample index
dfTrain["index"] = np.arange(num_train)
# dfTest["index"] = np.arange(num_test)

## clean text
clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)
dfTrain = dfTrain.apply(clean, axis=1)
# dfTest = dfTest.apply(clean, axis=1)

print("After clean text and the text is:")
print(dfTrain[['question1', 'question2']][:10])
print("Done.")


###############
## Save Data ##
###############
print("Save data...")

with open(config.processed_mild_train_data_path, "wb") as f:
    dill.dump(dfTrain, f, -1)
# with open(config.processed_test_data_path, "wb") as f:
#     dill.dump(dfTest, f, -1)
print("Done.")
