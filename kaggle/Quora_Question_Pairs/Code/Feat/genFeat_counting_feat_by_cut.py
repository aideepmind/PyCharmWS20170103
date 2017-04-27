"""
__file__

    genFeat_counting_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. Basic Counting Features

            1. Count of n-gram in query/title/description

            2. Count & Ratio of Digit in query/title/description

            3. Count & Ratio of Unique n-gram in query/title/description

        2. Intersect Counting Features

            1. Count & Ratio of a's n-gram in b's n-gram

        3. Intersect Position Features

            1. Statistics of Positions of a's n-gram in b's n-gram

            2. Statistics of Normalized Positions of a's n-gram in b's n-gram

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import re
import sys
import ngram
import pickle
import dill
import numpy as np
import pandas as pd
from nlp_utils import stopwords, english_stemmer, stem_tokens
from feat_utils import try_divide, dump_feat_name

sys.path.append("../")
from param_config import config


def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j, w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"


# token_pattern = r'\w{1,}'
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=config.cooccurrence_word_exclude_stopword,
                    encode_digit=False):
    token_pattern = re.compile(token_pattern,
                               flags=re.UNICODE | re.LOCALE)  # re.RegexFlag.UNICODE and re.RegexFlag.LOCALE
    ## tokenize
    tokens = [x.lower() for x in token_pattern.findall(line)]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed


def extract_feat(df):
    # ## unigram
    # print("generate unigram")
    # df["question1_unigram"] = list(df.apply(lambda x: preprocess_data(x["question1"]), axis=1))
    # with open("%s/train.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question1_unigram'), "wb") as f:
    #     dill.dump(df["question1_unigram"], f, -1)
    #     print('dump df at unigram step')
    # df["question2_unigram"] = list(df.apply(lambda x: preprocess_data(x["question2"]), axis=1))
    # with open("%s/train.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question2_unigram'), "wb") as f:
    #     dill.dump(df["question2_unigram"], f, -1)
    #     print('dump df at unigram step')
    # ## bigram
    # print("generate bigram")
    # join_str = "_"
    # df["question1_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["question1_unigram"], join_str), axis=1))
    # with open("%s/train.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question1_bigram'), "wb") as f:
    #     dill.dump(df["question1_bigram"], f, -1)
    #     print('dump df at unigram step')
    # df["question2_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["question2_unigram"], join_str), axis=1))
    # with open("%s/train.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question2_bigram'), "wb") as f:
    #     dill.dump(df["question2_bigram"], f, -1)
    #     print('dump df at bigram step')
    # ## trigram
    # print("generate trigram")

    with open("%s/test.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question1_unigram'), "rb") as f:
        df["question1_unigram"] = dill.load(f)
        print('load question1_unigram')
    with open("%s/test.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question2_unigram'), "rb") as f:
        df["question2_unigram"] = dill.load(f)
        print('load question2_unigram')
    # with open("%s/test.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question1_bigram'), "rb") as f:
    #     df["question1_bigram"] = dill.load(f)
    #     print('load question1_bigram')
    # with open("%s/test.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question2_bigram'), "rb") as f:
    #     df["question2_bigram"] = dill.load(f)
    #     print('load question2_bigram')
    # with open("%s/test.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question1_trigram'), "rb") as f:
    #     df["question1_trigram"] = dill.load(f)
    #     print('load question1_trigram')
    # with open("%s/test.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question2_trigram'), "rb") as f:
    #     df["question2_trigram"] = dill.load(f)
    #     print('load question2_trigram')

    with open("%s/test.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'count_of_question1_unigram'), "rb") as f:
        df["count_of_question1_unigram"] = dill.load(f)
        print('load count_of_question1_unigram')
    with open("%s/test.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'count_of_question2_unigram'), "rb") as f:
        df["count_of_question2_unigram"] = dill.load(f)
        print('load count_of_question2_unigram')

    # join_str = "_"
    # df["question1_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["question1_unigram"], join_str), axis=1))
    # with open("%s/train.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question1_trigram'), "wb") as f:
    #     dill.dump(df["question1_trigram"], f, -1)
    #     print('dump df at trigram step')
    # df["question2_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["question2_unigram"], join_str), axis=1))
    # with open("%s/train.%s.feat.pkl" % ("%s/All" % config.feat_folder, 'question2_trigram'), "wb") as f:
    #     dill.dump(df["question2_trigram"], f, -1)
    #     print('dump df at trigram step')
    ################################
    ## word count and digit count ##
    ################################
    print("generate word counting features")
    feat_names = ["question1", "question2"]
    grams = ["unigram"]  # , "bigram", "trigram"
    # count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    # for feat_name in feat_names:
    #     for gram in grams:
    #         ## word count 单词数量
    #         count = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
    #         df['count_of_%s_%s' % (feat_name, gram)] = count
    #         count_of_unique = list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
    #         ratio_of_unique = list(map(try_divide, count_of_unique, count))
    #         with open("%s/test.count_of_%s_%s.feat.pkl" % ("%s/All" % config.feat_folder, feat_name, gram), "wb") as f:
    #             dill.dump(count, f, -1)
    #         with open("%s/test.count_of_unique_%s_%s.feat.pkl" % ("%s/All" % config.feat_folder, feat_name, gram), "wb") as f:
    #             dill.dump(count_of_unique, f, -1)
    #         with open("%s/test.ratio_of_unique_%s_%s.feat.pkl" % ("%s/All" % config.feat_folder, feat_name, gram), "wb") as f:
    #             dill.dump(ratio_of_unique, f, -1)
    #         #
    #         # ## digit count 数字数量
    #         # count_of_digit_in = list( map(lambda x: count_digit(x), df[feat_name + '_' + gram]))
    #         # ratio_of_digit_in = list(
    #         #     map(try_divide, count_of_digit_in, count))
    #         # with open("%s/test.count_of_digit_in_%s.feat.pkl" % ("%s/All" % config.feat_folder, feat_name), "wb") as f:
    #         #     dill.dump(count_of_digit_in, f, -1)
    #         # with open("%s/test.ratio_of_digit_in_%s.feat.pkl" % ("%s/All" % config.feat_folder, feat_name), "wb") as f:
    #         #     dill.dump(ratio_of_digit_in, f, -1)
    #
    # ##############################
    # ## intersect word count ##
    # ##############################
    # print("generate intersect word counting features")
    # #### unigram
    # for gram in grams:
    #     for obs_name in feat_names:
    #         for target_name in feat_names:
    #             if target_name != obs_name:
    #                 ## query
    #                 count_of_= list(df.apply(
    #                     lambda x: sum([1. for w in x[obs_name + "_" + gram] if w in set(x[target_name + "_" + gram])]),
    #                     axis=1))
    #                 ratio_of_ = list(
    #                     map(try_divide, count_of_,
    #                         df["count_of_%s_%s" % (obs_name, gram)]))
    #                 with open("%s/test.count_of_%s_%s_in_%s.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
    #                           "wb") as f:
    #                     dill.dump(count_of_, f, -1)
    #                 with open("%s/test.ratio_of_%s_%s_in_%s.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
    #                           "wb") as f:
    #                     dill.dump(ratio_of_, f, -1)

    ######################################
    ## intersect word position feat ##
    ######################################
    print("generate intersect word position features")
    for gram in grams:
        for target_name in feat_names:
            for obs_name in feat_names:
                if target_name != obs_name:
                    pos = list(
                        df.apply(lambda x: get_position_list(x[target_name + "_" + gram], obs=x[obs_name + "_" + gram]),
                                 axis=1))
                    ## stats feat on pos
                    min = list(map(np.min, pos))
                    mean = list(map(np.mean, pos))
                    median = list(map(np.median, pos))
                    max = list(map(np.max, pos))
                    std = list(map(np.std, pos))
                    ## stats feat on normalized_pos
                    min_normalized = list(
                        map(try_divide, min,
                            df["count_of_%s_%s" % (obs_name, gram)]))
                    mean_normalized = list(
                        map(try_divide, mean,
                            df["count_of_%s_%s" % (obs_name, gram)]))
                    median_normalized = list(
                        map(try_divide, median,
                            df["count_of_%s_%s" % (obs_name, gram)]))
                    max_normalized = list(
                        map(try_divide, max,
                            df["count_of_%s_%s" % (obs_name, gram)]))
                    std_normalized = list(
                        map(try_divide, std,
                            df["count_of_%s_%s" % (obs_name, gram)]))

                    with open("%s/test.pos_of_%s_%s_in_%s_min.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                              "wb") as f:
                        dill.dump(min, f, -1)
                    with open("%s/test.pos_of_%s_%s_in_%s_mean.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                              "wb") as f:
                        dill.dump(mean, f, -1)
                    with open("%s/test.pos_of_%s_%s_in_%s_median.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                              "wb") as f:
                        dill.dump(median, f, -1)
                    with open("%s/test.pos_of_%s_%s_in_%s_max.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                              "wb") as f:
                        dill.dump(max, f, -1)
                    with open("%s/test.pos_of_%s_%s_in_%s_std.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                              "wb") as f:
                        dill.dump(std, f, -1)

                    # with open("%s/test.normalized_pos_of_%s_%s_in_%s_min.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                    #           "wb") as f:
                    #     dill.dump(min_normalized, f, -1)
                    # with open("%s/test.normalized_pos_of_%s_%s_in_%s_mean.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                    #           "wb") as f:
                    #     dill.dump(mean_normalized, f, -1)
                    # with open("%s/test.normalized_pos_of_%s_%s_in_%s_median.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                    #           "wb") as f:
                    #     dill.dump(median_normalized, f, -1)
                    # with open("%s/test.normalized_pos_of_%s_%s_in_%s_max.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                    #           "wb") as f:
                    #     dill.dump(max_normalized, f, -1)
                    # with open("%s/test.normalized_pos_of_%s_%s_in_%s_std.feat.pkl" % ("%s/All" % config.feat_folder, obs_name, gram, target_name),
                    #           "wb") as f:
                    #     dill.dump(std_normalized, f, -1)


if __name__ == "__main__":

    ###############
    ## Load Data ##
    ###############
    ## load data
    # with open(config.processed_train_data_path, "rb") as f:
    #     dfTrain = dill.load(f)
    # with open(config.processed_test_data_path, "rb") as f:
    #     dfTest = dill.load(f)
    # ## load pre-defined stratified k-fold index
    # with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
    #     skf = pickle.load(f, encoding='latin1')

    ## file to save feat names
    feat_name_file = "%s/counting.feat_name" % config.feat_folder

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate counting features...")

    # extract_feat(dfTrain)
    # feat_names = [
    #     name for name in dfTrain.columns \
    #     if "count" in name \
    #     or "ratio" in name \
    #     or "div" in name \
    #     or "pos_of" in name
    #     ]

    feat_names = ['count_of_question1_unigram',
                  'count_of_unique_question1_unigram',
                  'ratio_of_unique_question1_unigram',
                  'count_of_question1_bigram',
                  'count_of_unique_question1_bigram',
                  'ratio_of_unique_question1_bigram',
                  'count_of_question1_trigram',
                  'count_of_unique_question1_trigram',
                  'ratio_of_unique_question1_trigram',
                  'count_of_digit_in_question1',
                  'ratio_of_digit_in_question1',
                  'count_of_question2_unigram',
                  'count_of_unique_question2_unigram',
                  'ratio_of_unique_question2_unigram',
                  'pos_of_question1_unigram_in_question2_min',
                  'pos_of_question1_unigram_in_question2_mean',
                  'pos_of_question1_unigram_in_question2_median',
                  'pos_of_question1_unigram_in_question2_max',
                  'pos_of_question1_unigram_in_question2_std',
                  'normalized_pos_of_question1_unigram_in_question2_min',
                  'normalized_pos_of_question1_unigram_in_question2_mean',
                  'normalized_pos_of_question1_unigram_in_question2_median',
                  'normalized_pos_of_question1_unigram_in_question2_max',
                  'normalized_pos_of_question1_unigram_in_question2_std',
                  'pos_of_question2_unigram_in_question1_min',
                  'pos_of_question2_unigram_in_question1_mean',
                  'pos_of_question2_unigram_in_question1_median',
                  'pos_of_question2_unigram_in_question1_max',
                  'pos_of_question2_unigram_in_question1_std',
                  'normalized_pos_of_question2_unigram_in_question1_min',
                  'normalized_pos_of_question2_unigram_in_question1_mean',
                  'normalized_pos_of_question2_unigram_in_question1_median',
                  'normalized_pos_of_question2_unigram_in_question1_max',
                  'normalized_pos_of_question2_unigram_in_question1_std']

    # print("For cross-validation...")
    # for run in range(config.n_runs):
    #     ## use 33% for training and 67 % for validation
    #     ## so we switch trainInd and validInd
    #     for fold, (validInd, trainInd) in enumerate(skf[run]):
    #         print("Run: %d, Fold: %d" % (run+1, fold+1))
    #         path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
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
    dfTest = pd.DataFrame()
    extract_feat(dfTest)
    # for feat_name in feat_names:
    #     # X_train = dfTrain[feat_name].values
    #     X_test = dfTest[feat_name].values
    #     # with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
    #     #     dill.dump(X_train, f, -1)
    #     with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
    #         dill.dump(X_test, f, -1)

    ## save feat names
    print("Feature names are stored in %s" % feat_name_file)
    ## dump feat name
    dump_feat_name(feat_names, feat_name_file)

    print("All Done.")