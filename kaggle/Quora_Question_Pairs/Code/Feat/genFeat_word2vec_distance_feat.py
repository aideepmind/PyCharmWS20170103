
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
import gensim
import pandas as pd
from copy import copy
from scipy.sparse import vstack
from nltk import word_tokenize
from collections import defaultdict
from collections import Counter
import argparse
import functools
from fuzzywuzzy import fuzz
from tqdm import tqdm   # 进度条
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nlp_utils import stopwords, english_stemmer, stem_tokens
from feat_utils import get_sample_indices_by_relevance, dump_feat_name
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from collections import Counter
sys.path.append("../")
from param_config import config


# model = gensim.models.KeyedVectors.load_word2vec_format('E:\安装软件\Python相关/GoogleNews-vectors-negative300.bin', binary=True)

norm_model = gensim.models.KeyedVectors.load_word2vec_format('E:\安装软件\Python相关/GoogleNews-vectors-negative300.bin', binary=True)
norm_model.init_sims(replace=True)

#####################
## Helper function ##
#####################
# 使用word2vec来计算两问题之间的距离
# def wmdistance(s1, s2):
#     s1 = str(s1).lower().split()
#     s2 = str(s2).lower().split()
#     s1 = [w for w in s1 if w not in stopwords]
#     s2 = [w for w in s2 if w not in stopwords]
#     return model.wmdistance(s1, s2)


def norm_wmdistance(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in stopwords]
    s2 = [w for w in s2 if w not in stopwords]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stopwords]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)   # 按行累加，即所有行相加，列不变
    return v / np.sqrt((v ** 2).sum())  # 模
# word share
def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R

# jaccard距离
def jaccard_by_words(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

# 共同词数量
def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))

# 唯一单词数量
def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

# 唯一但不包括停词单词数量
def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

# 单词数量差绝对值
def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

# 单词数量比率
def wc_ratio(row):
    l1 = len(row['question1']) * 1.0
    l2 = len(row['question2'])
    if l2 == 0:
        return 0
    if l1 / l2: # 两者都不等于0时，以l1为分母
        return l2 / l1
    else:
        return l1 / l2

# 唯一单词数量差绝对值
def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

# 唯一单词数量比率
def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return 0
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

# 唯一但不包括单词数量差绝对值
def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len(
        [x for x in set(row['question2']) if x not in stops]))

# 唯一但不包括单词数量比率
def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops]) * 1.0
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return 0
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

# 首单词是否相同
def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return 0
    return int(row['question1'][0] == row['question2'][0])

# 字符数量差
def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

# 字符数量比率
def char_ratio(row):
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return 0
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

# 唯一但不包括字符数量差绝对值
def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(
        ''.join([x for x in set(row['question2']) if x not in stops])))

# 获取单词权重
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

# tfidf
def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

# tfidf 包括停词
def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

## extract all features
def extract_feat(df):
    df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True)
    # count 处理
    # print("generate char length features")
    # df['len_of_question1'] = df.question1.apply(lambda x: len(str(x)))
    # df['len_of_question2'] = df.question2.apply(lambda x: len(str(x)))
    # df['diff_len_of_question1_question2'] = df.len_of_question1 - df.len_of_question2
    # df['len_of_char_question1'] = df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    # df['len_of_char_question2'] = df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    # df['len_of_word_question1'] = df.question1.apply(lambda x: len(str(x).split()))
    # df['len_of_word_question2'] = df.question2.apply(lambda x: len(str(x).split()))
    # df['common_words_question1_question2'] = df.apply(
    #     lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),
    #     axis=1)
    # print("generate fuzz features")
    # Fuzzywuzzy是一个可以对字符串进行模糊匹配的小工具
    # 简单比
    # df['fuzz_qratio_question1_question2'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    # df['fuzz_WRatio_question1_question2'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    # # 部分比
    # df['fuzz_partial_ratio_question1_question2'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])),
    #                                         axis=1)
    # # 单词集合比
    # df['fuzz_partial_token_set_ratio_question1_question2'] = df.apply(
    #     lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    # # 单词排序比
    # df['fuzz_partial_token_sort_ratio_question1_question2'] = df.apply(
    #     lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    # df['fuzz_token_set_ratio_question1_question2'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
    #                                           axis=1)
    # df['fuzz_token_sort_ratio_question1_question2'] = df.apply(
    #     lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    #
    #
    # print("generate wmdistance features")
    # df['wmdistance_question1_question2'] = df.apply(lambda x: wmdistance(x['question1'], x['question2']), axis=1)
    df['norm_wmdistance_question1_question2'] = df.apply(lambda x: norm_wmdistance(x['question1'], x['question2']), axis=1)
    #
    #
    # question1_vectors = np.zeros((df.shape[0], 300))
    # # Tqdm 是一个快速，可扩展的Python进度条
    # for i, q in tqdm(enumerate(df.question1.values)):
    #     question1_vectors[i, :] = sent2vec(q)
    #
    # question2_vectors = np.zeros((df.shape[0], 300))
    # for i, q in tqdm(enumerate(df.question2.values)):
    #     question2_vectors[i, :] = sent2vec(q)
    #
    # # 各种距离
    # print("generate distance features")
    # df['cosine_distance_between_question1_question2'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                                     np.nan_to_num(question2_vectors))]
    # df['cityblock_distance_between_question1_question2'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                                           np.nan_to_num(question2_vectors))]
    # df['jaccard_distance_between_question1_question2'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                                       np.nan_to_num(question2_vectors))]
    # df['canberra_distance_between_question1_question2'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                                         np.nan_to_num(question2_vectors))]
    # df['euclidean_distance_between_question1_question2'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                                           np.nan_to_num(question2_vectors))]
    # df['minkowski_distance_between_question1_question2'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                                              np.nan_to_num(question2_vectors))]
    # df['braycurtis_distance_between_question1_question2'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                                             np.nan_to_num(question2_vectors))]
    #
    # df.fillna(0)

    # # 三偏四峰
    # print("generate skew and kur features")
    # df['skew_of_question1_vec'] = [skew(x) for x in 0_to_num(question1_vectors)]
    # df['skew_of_question2_vec'] = [skew(x) for x in 0_to_num(question2_vectors)]
    # df['kur_of_question1_vec'] = [kurtosis(x) for x in 0_to_num(question1_vectors)]
    # df['kur_of_question2_vec'] = [kurtosis(x) for x in 0_to_num(question2_vectors)]


    # train_qs = pd.Series(df['question1'].tolist() + df['question2'].tolist())
    # words = [x for y in train_qs for x in y]
    # counts = Counter(words)
    # weights = {word: get_weight(count) for word, count in counts.items()}
    #
    # # f = functools.partial(word_match_share, stops=stopwords)
    # # df['word_match'] = df.apply(f, axis=1, raw=True)  # 1
    #
    # f = functools.partial(tfidf_word_match_share, weights=weights)
    # df['tfidf_wm_question1_question2'] = df.apply(f, axis=1, raw=True)  # 2
    #
    # f = functools.partial(tfidf_word_match_share_stops, stops=stopwords, weights=weights)
    # df['tfidf_wm_stops_question1_question2'] = df.apply(f, axis=1, raw=True)  # 3
    #
    # df['jaccard_question1_question2'] = df.apply(jaccard_by_words, axis=1, raw=True)  # 4
    # df['wc_diff_question1_question2'] = df.apply(wc_diff, axis=1, raw=True)  # 5
    # df['wc_ratio_question1_question2'] = df.apply(wc_ratio, axis=1, raw=True)  # 6
    # df['wc_diff_unique_question1_question2'] = df.apply(wc_diff_unique, axis=1, raw=True)  # 7
    # df['wc_ratio_unique_question1_question2'] = df.apply(wc_ratio_unique, axis=1, raw=True)  # 8
    #
    # f = functools.partial(wc_diff_unique_stop, stops=stopwords)
    # df['wc_diff_unq_stop_question1_question2'] = df.apply(f, axis=1, raw=True)  # 9
    # f = functools.partial(wc_ratio_unique_stop, stops=stopwords)
    # df['wc_ratio_unique_stop_question1_question2'] = df.apply(f, axis=1, raw=True)  # 10
    #
    # df['same_start_question1_question2'] = df.apply(same_start_word, axis=1, raw=True)  # 11
    # df['char_diff_question1_question2'] = df.apply(char_diff, axis=1, raw=True)  # 12
    #
    # f = functools.partial(char_diff_unique_stop, stops=stopwords)
    # df['char_diff_unq_stop_question1_question2'] = df.apply(f, axis=1, raw=True)  # 13
    #
    # #     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    # df['total_unique_words_question1_question2'] = df.apply(total_unique_words, axis=1, raw=True)  # 15
    #
    # f = functools.partial(total_unq_words_stop, stops=stopwords)
    # df['total_unq_words_stop_question1_question2'] = df.apply(f, axis=1, raw=True)  # 16
    #
    # df['char_ratio_question1_question2'] = df.apply(char_ratio, axis=1, raw=True)  # 17


if __name__ == "__main__":

    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(config.processed_mild_train_data_path, "rb") as f:
        dfTrain = dill.load(f)
    # with open(config.processed_mild_test_data_path, "rb") as f:
    #     dfTest = dill.load(f)
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


    extract_feat(dfTrain)
    feat_names = [
        name for name in dfTrain.columns \
        if "len" in name \
        or "diff" in name \
        or "common" in name \
        or "fuzz" in name \
        or "wmd" in name \
        or "distance" in name \
        or "skew" in name \
        or "kur" in name \
        or "total" in name \
        or "char" in name \
        or "same" in name \
        or "ratio" in name \
        or "tfidf" in name \
        or "jaccard" in name
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

    # print("For training and testing...")
    # path = "%s/All" % config.feat_folder
    # ## use full version for X_train
    # extract_feat(dfTest)
    # for feat_name in feat_names:
    #     X_train = dfTrain[feat_name].values
    #     X_test = dfTest[feat_name].values
    #     with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
    #         dill.dump(X_train, f, -1)
    #     with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
    #         dill.dump(X_test, f, -1)
    #
    # ## save feat names
    # print("Feature names are stored in %s" % feat_name_file)
    # ## dump feat name
    # dump_feat_name(feat_names, feat_name_file)

    print("All Done.")