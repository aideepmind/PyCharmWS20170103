
"""
__file__

    genFeat_cooccurrence_tfidf.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. tfidf for the following cooccurrence terms
            - question1 unigram/bigram & question2 unigram/bigram
            - question1 unigram/bigram & description unigram/bigram
            - question1 id & question2 unigram/bigram
            - question1 id & description unigram/bigram

        2. corresponding lsa (svd) version features

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import re
import sys
import pickle
import dill
import ngram
from feat_utils import dump_feat_name
from sklearn.decomposition import TruncatedSVD
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
sys.path.append("../")
from param_config import config

######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
def preprocess_data(line,
                    token_pattern=token_pattern,
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


########################
## Cooccurrence terms ##
########################
def cooccurrence_terms(lst1, lst2, join_str):
    terms = [""] * len(lst1) * len(lst2)
    cnt =  0
    for item1 in lst1:
        for item2 in lst2:
            terms[cnt] = item1 + join_str + item2
            cnt += 1
    res = " ".join(terms)
    return res


##################
## Extract feat ##
##################
def extract_feat(df):
    ## unigram
    print("generate unigram")
    df["question1_unigram"] = list(df.apply(lambda x: preprocess_data(x["question1"]), axis=1))
    df["question2_unigram"] = list(df.apply(lambda x: preprocess_data(x["question2"]), axis=1))
    ## bigram
    print("generate bigram")
    join_str = "_"
    df["question1_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["question1_unigram"], join_str), axis=1))
    df["question2_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["question2_unigram"], join_str), axis=1))
    # ## trigram
    # join_str = "_"
    # df["question1_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["question1_unigram"], join_str), axis=1))
    # df["question2_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["question2_unigram"], join_str), axis=1))

    ## cooccurrence terms
    join_str = "X"
    # question1 unigram
    df["question1_unigram_question2_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["question1_unigram"], x["question2_unigram"], join_str), axis=1))
    df["question1_unigram_question2_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["question1_unigram"], x["question2_bigram"], join_str), axis=1))
    # question1 bigram
    df["question1_bigram_question2_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["question1_bigram"], x["question2_unigram"], join_str), axis=1))
    df["question1_bigram_question2_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["question1_bigram"], x["question2_bigram"], join_str), axis=1))

        

if __name__ == "__main__":

    ############
    ## Config ##
    ############
    ## cooccurrence terms column names
    column_names = [
        "question1_unigram_question2_unigram",
        "question1_unigram_question2_bigram",
        "question1_bigram_question2_unigram",
        "question1_bigram_question2_bigram"
    ]
    ## feature names
    feat_names = [ name+"_tfidf" for name in column_names ]
    ## file to save feat names
    feat_name_file = "%s/intersect_tfidf.feat_name" % config.feat_folder

    ngram_range = config.cooccurrence_tfidf_ngram_range

    svd_n_components = 100

    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = dill.load(f)
    # with open(config.processed_test_data_path, "rb") as f:
    #     dfTest = dill.load(f)
    ## load pre-defined stratified k-fold index
    with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
            skf = pickle.load(f, encoding='latin1')

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate co-occurrence tfidf features...")

    ## get cooccurrence terms
    extract_feat(dfTrain)
    # extract_feat(dfTest)

    ######################
    ## Cross validation ##
    ######################
    print("For cross-validation...")
    for run in range(config.n_runs):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        for fold, (validInd, trainInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
                
            for feat_name,column_name in zip(feat_names, column_names):
                print("generate %s feat" % feat_name)
                ## tfidf
                tfv = getTFV(ngram_range=ngram_range)
                X_tfidf_train = tfv.fit_transform(dfTrain.iloc[trainInd][column_name])
                X_tfidf_valid = tfv.transform(dfTrain.iloc[validInd][column_name])
                with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    dill.dump(X_tfidf_train, f, -1)
                with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    dill.dump(X_tfidf_valid, f, -1)

                ## svd
                svd = TruncatedSVD(n_components=svd_n_components, n_iter=15)
                X_svd_train = svd.fit_transform(X_tfidf_train)
                X_svd_test = svd.transform(X_tfidf_valid)
                with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
                    dill.dump(X_svd_train, f, -1)
                with open("%s/valid.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
                    dill.dump(X_svd_test, f, -1)

    print("Done.")


    #################
    ## Re-training ##
    #################
    # print("For training and testing...")
    # path = "%s/All" % config.feat_folder
    # for feat_name,column_name in zip(feat_names, column_names):
    #     print("generate %s feat" % feat_name)
    #     tfv = getTFV(ngram_range=ngram_range)
    #     X_tfidf_train = tfv.fit_transform(dfTrain[column_name])
    #     X_tfidf_test = tfv.transform(dfTest[column_name])
    #     with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
    #         dill.dump(X_tfidf_train, f, -1)
    #     with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
    #         dill.dump(X_tfidf_test, f, -1)
    #
    #     ## svd
    #     svd = TruncatedSVD(n_components=svd_n_components, n_iter=15)
    #     X_svd_train = svd.fit_transform(X_tfidf_train)
    #     X_svd_test = svd.transform(X_tfidf_test)
    #     with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
    #         dill.dump(X_svd_train, f, -1)
    #     with open("%s/test.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
    #         dill.dump(X_svd_test, f, -1)
    #
    # print("Done.")
    #
    # ## save feat names
    # print("Feature names are stored in %s" % feat_name_file)
    # feat_names += [ "%s_individual_svd%d"%(f, svd_n_components) for f in feat_names ]
    # dump_feat_name(feat_names, feat_name_file)

    print("All Done.")