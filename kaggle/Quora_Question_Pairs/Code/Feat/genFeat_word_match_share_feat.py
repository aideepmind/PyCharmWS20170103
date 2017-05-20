
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
## word match share
def word_match_share(w1, w2):
    if len(w1) == 0 or len(w2) == 0:
        return 0
    shared_words = [w for w in w1 if w in w2] + [w for w in w2 if w in w1]
    rate = len(shared_words) / (len(w1) + len(w2))
    return rate

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


## tfidf word match share
global weights
weights = None
def init_weights(dfTrain):
    ## unigram
    print("generate unigram")
    dfTrain["question2_unigram"] = list(dfTrain.apply(lambda x: preprocess_data2(x["question2"]), axis=1))
    dfTrain["question1_unigram"] = list(dfTrain.apply(lambda x: preprocess_data2(x["question1"]), axis=1))
    # eps = 5000
    words = []
    feat_names = ["question1", "question2"]
    for feat_name in feat_names:
        col = dfTrain["%s_unigram" % feat_name]
        for row in range(0, len(col)):
            for word in col[row]:
                words.append(word)
    counts = Counter(words)
    global weights
    weights = {word: get_weight(count) for word, count in counts.items()}


def tfidf_word_match_share(w1, w2):
    if len(w1) == 0 or len(w2) == 0:
        return 0
    shared_weights = [weights.get(w, 0) for w in w1 if w in w2] + [weights.get(w, 0) for w in w2 if w in w1]
    total_weights = [weights.get(w, 0) for w in w1] + [weights.get(w, 0) for w in w2]
    if np.sum(total_weights) == 0:
        return 0
    rate = 1.0 * np.sum(shared_weights) / np.sum(total_weights)
    return rate



######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"    # \b: 匹配一个单词边界，也就是指单词和空格间的位置
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
transform = config.count_feat_transform
def preprocess_data(line, token_pattern=token_pattern,
                     exclude_stopword=config.cooccurrence_word_exclude_stopword,
                     encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    ## tokenize
    tokens = [x.lower() for x in token_pattern.findall(line)]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed

def preprocess_data2(line):
    tokens = line.split()
    tokens = [x for x in tokens if x not in stopwords]
    return set(tokens)

## extract all features
def extract_feat(df):
    ## unigram
    print("generate unigram")
    df["question2_unigram"] = list(df.apply(lambda x: preprocess_data2(x["question2"]), axis=1))
    df["question1_unigram"] = list(df.apply(lambda x: preprocess_data2(x["question1"]), axis=1))

    # print('question1_unigram', df["question1_unigram"].head(10))
    # print('question2_unigram', df["question2_unigram"].head(10))
    ######################
    ## word match share ##
    ######################
    feat_names = ["question1", "question2"]
    for i in range(len(feat_names) - 1):
        for j in range(i + 1, len(feat_names)):
            target_name = feat_names[i]
            obs_name = feat_names[j]
            df["ratio_of_%s_%s_%s_share" % (target_name, obs_name, "unigram")] = list(df.apply(
                lambda x: word_match_share(x[target_name + "_unigram"], x[obs_name + "_unigram"]),
                axis=1))


    ############################
    ## tfidf word match share ##
    ############################
    for i in range(len(feat_names) - 1):
        for j in range(i + 1, len(feat_names)):
            target_name = feat_names[i]
            obs_name = feat_names[j]
            df["ratio_of_%s_%s_%s_share_tfidf" % (target_name, obs_name, "unigram")] = list(df.apply(
                lambda x: tfidf_word_match_share(x[target_name + "_unigram"], x[obs_name + "_unigram"]),
                axis=1))


if __name__ == "__main__":

    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(config.processed_mild_train_data_path, "rb") as f:
        dfTrain = dill.load(f)
    with open(config.processed_mild_test_data_path, "rb") as f:
        dfTest = dill.load(f)
    ## load pre-defined stratified k-fold index
    with open(config.cv_info_path, "rb") as f:
        skf = pickle.load(f, encoding='latin1')

    ## file to save feat names
    feat_name_file = "%s/counting.feat_name" % config.feat_folder

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate counting features...")

    # init weight for tfidf
    init_weights(dfTrain)

    extract_feat(dfTrain)
    feat_names = [name for name in dfTrain.columns if "ratio" in name]

    print("For cross-validation...")
    # for run in range(config.n_runs):
    #     ## use 33% for training and 67 % for validation
    #     ## so we switch trainInd and validInd
    #     for fold, (validInd, trainInd) in enumerate(skf[run]):
    #         print("Run: %d, Fold: %d" % (run + 1, fold + 1))
    #         path = "%s/Run%d/Fold%d" % (config.feat_folder, run + 1, fold + 1)
    #
    #         #########################
    #         ## get word count feat ##
    #         #########################
    #         for feat_name in feat_names:
    #             X_train = dfTrain[feat_name].values[trainInd]
    #             X_valid = dfTrain[feat_name].values[validInd]
    #             with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
    #                 dill.dump(X_train, f, -1)
    #             with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
    #                 dill.dump(X_valid, f, -1)
    # print("Done.")

    print("For training and testing...")
    path = "%s/All" % config.feat_folder
    ## use full version for X_train
    extract_feat(dfTest)
    for feat_name in feat_names:
        X_train = dfTrain[feat_name].values
        X_test = dfTest[feat_name].values
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
            dill.dump(X_train, f, -1)
        with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
            dill.dump(X_test, f, -1)
    #
    # ## save feat names
    # print("Feature names are stored in %s" % feat_name_file)
    # ## dump feat name
    # dump_feat_name(feat_names, feat_name_file)

    print("All Done.")