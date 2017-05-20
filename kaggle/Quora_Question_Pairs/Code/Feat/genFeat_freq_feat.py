
"""
__file__

    genFeat_basic_tfidf_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. basic tfidf features for query/title/description
            - use common vocabulary among query/title/description for further computation of cosine similarity

        2. cosine similarity between query & title, query & description, title & description pairs
            - just plain cosine similarity

        3. cosine similarity stats features for title/description
            - computation is carried out with regard to a pool of samples grouped by:
                - median_relevance (#4)
                - query (qid) & median_relevance (#4)
            - cosine similarity for the following pairs are computed for each sample
                - sample title        vs.  pooled sample titles
                - sample description  vs.  pooled sample descriptions
                Note that in the pool samples, we exclude the current sample being considered.
            - stats features include quantiles of cosine similarity and others defined in the variable "stats_func", e.g.,
                - mean value
                - standard deviation (std)
                - more can be added, e.g., moment features etc

        4. SVD version of the above features

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import re
import sys
import pickle
import dill
import numpy as np
import pandas as pd
from copy import copy
from scipy.sparse import vstack
from nlp_utils import stopwords, english_stemmer, stem_tokens
from feat_utils import get_sample_indices_by_relevance, dump_feat_name
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from collections import Counter
sys.path.append("../")
from param_config import config


#####################
## Helper function ##
#####################
## 根据字符串从字典中取出数字
def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0

## extract all features
def extract_feat(dfTrain, dfTest):
    df1 = dfTrain[['question1']].copy()
    df2 = dfTrain[['question2']].copy()
    df1_test = dfTest[['question1']].copy()
    df2_test = dfTest[['question2']].copy()

    df2.rename(columns={'question2': 'question1'}, inplace=True)
    df2_test.rename(columns={'question2': 'question1'}, inplace=True)

    train_questions = df1.append(df2)
    train_questions = train_questions.append(df1_test)
    train_questions = train_questions.append(df2_test)
    #train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
    train_questions.drop_duplicates(subset = ['question1'],inplace=True)

    train_questions.reset_index(inplace=True,drop=True)
    questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
    train_cp = dfTrain.copy()
    test_cp = dfTest.copy()
    train_cp.drop(['qid1','qid2'], axis=1, inplace=True)

    test_cp['is_duplicate'] = -1
    test_cp.rename(columns={'test_id':'id'},inplace=True)
    comb = pd.concat([train_cp,test_cp])

    comb['sentence_hash_of_question1'] = comb['question1'].map(questions_dict)
    comb['sentence_hash_of_question2'] = comb['question2'].map(questions_dict)

    question1_vc = comb.sentence_hash_of_question1.value_counts().to_dict()
    question2_vc = comb.sentence_hash_of_question2.value_counts().to_dict()

    # map to frequency space
    comb['sentence_freq_of_question1'] = comb['sentence_hash_of_question1'].map(lambda x: try_apply_dict(x, question1_vc) + try_apply_dict(x, question2_vc))
    comb['sentence_freq_of_question2'] = comb['sentence_hash_of_question2'].map(lambda x: try_apply_dict(x, question1_vc) + try_apply_dict(x, question2_vc))

    train_comb = comb[comb['is_duplicate'] >= 0][['sentence_hash_of_question1', 'sentence_hash_of_question2', 'sentence_freq_of_question1', 'sentence_freq_of_question2']]
    test_comb = comb[comb['is_duplicate'] < 0][['sentence_hash_of_question1', 'sentence_hash_of_question2', 'sentence_freq_of_question1', 'sentence_freq_of_question2']]

    dfTrain = pd.concat([dfTrain, train_comb], axis=1)
    dfTest = pd.concat([dfTest, test_comb], axis=1)

    return dfTrain, dfTest


if __name__ == "__main__":

    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = dill.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = dill.load(f)
    ## load pre-defined stratified k-fold index
    with open(config.cv_info_path, "rb") as f:
        skf = pickle.load(f, encoding='latin1')

    ## file to save feat names
    # feat_name_file = "%s/counting.feat_name" % config.feat_folder

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate freq and hash features...")

    dfTrain, dfTest = extract_feat(dfTrain, dfTest)
    feat_names = [
        name for name in dfTrain.columns \
        if "hash" in name \
        or "freq" in name
        ]

    print("For cross-validation...")
    for run in range(config.n_runs):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        for fold, (validInd, trainInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run + 1, fold + 1))
            path = "%s/Run%d/Fold%d" % (config.feat_folder, run + 1, fold + 1)

            #########################
            ## get word count feat ##
            #########################
            for feat_name in feat_names:
                X_train = dfTrain[feat_name].values[trainInd]
                X_valid = dfTrain[feat_name].values[validInd]
                with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    dill.dump(X_train, f, -1)
                with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    dill.dump(X_valid, f, -1)
    print("Done.")

    print("For training and testing...")
    path = "%s/All" % config.feat_folder

    for feat_name in feat_names:
        X_train = dfTrain[feat_name].values
        X_test = dfTest[feat_name].values
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
            dill.dump(X_train, f, -1)
        with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
            dill.dump(X_test, f, -1)

    # ## save feat names
    # print("Feature names are stored in %s" % feat_name_file)
    # ## dump feat name
    # dump_feat_name(feat_names, feat_name_file)

    print("All Done.")