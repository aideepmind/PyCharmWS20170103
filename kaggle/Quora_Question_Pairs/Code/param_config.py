
"""
__file__

    param_config.py

__description__

    This file provides global parameter configurations for the project.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import os
import numpy as np


############
## Config ##
############
class ParamConfig:
    def __init__(self,
                 feat_folder,
                 drop_html_flag,
                 basic_tfidf_ngram_range=(1,3),
                 basic_tfidf_vocabulary_type="common",
                 cooccurrence_tfidf_ngram_range=(1,1),
                 cooccurrence_word_exclude_stopword=False,
                 stemmer_type="snowball",
                 count_feat_transform=np.sqrt):
    
        self.n_classes = 2

        ## CV params
        self.n_runs = 1
        self.n_folds = 1
        self.stratified_label = "duplicate"

        ## path
        self.data_folder = "../../Data"
        self.feat_folder = feat_folder
        self.original_train_data_path = "%s/train.csv" % self.data_folder
        self.original_test_data_path = "%s/test.csv" % self.data_folder
        self.processed_train_data_path = "%s/train.processed.csv.pkl" % self.feat_folder
        self.processed_mild_train_data_path = "%s/train.processed_mild.csv.pkl" % self.feat_folder
        self.processed_test_data_path = "%s/test.processed.csv.pkl" % self.feat_folder
        self.processed_mild_test_data_path = "%s/test.processed_mild.csv.pkl" % self.feat_folder
        # self.pos_tagged_train_data_path = "%s/train.pos_tagged.csv.pkl" % self.feat_folder
        # self.pos_tagged_test_data_path = "%s/test.pos_tagged.csv.pkl" % self.feat_folder
        # cv path
        self.cv_info_path = "%s/stratifiedKFold.%s.pkl" % (self.data_folder, self.stratified_label)
        # self.cv_info_path = "%s/timeSeriesSplit.pkl" % self.data_folder
        # self.cv_info_path = "%s/random.pkl" % self.data_folder


        ## nlp related
        self.drop_html_flag = drop_html_flag
        self.basic_tfidf_ngram_range = basic_tfidf_ngram_range  #
        self.basic_tfidf_vocabulary_type = basic_tfidf_vocabulary_type  # 词汇类型
        self.cooccurrence_tfidf_ngram_range = cooccurrence_tfidf_ngram_range  #
        self.cooccurrence_word_exclude_stopword = cooccurrence_word_exclude_stopword
        self.stemmer_type = stemmer_type

        ## transform for count features
        self.count_feat_transform = count_feat_transform

        ## create feat folder
        if not os.path.exists(self.feat_folder):
            os.makedirs(self.feat_folder)

        ## creat folder for the training and testing feat
        if not os.path.exists("%s/All" % self.feat_folder):
            os.makedirs("%s/All" % self.feat_folder)

        ## creat folder for each run and fold
        for run in range(1, self.n_runs+1):
            for fold in range(1, self.n_folds+1):
                path = "%s/Run%d/Fold%d" % (self.feat_folder, run, fold)
                if not os.path.exists(path):
                    os.makedirs(path)


## initialize a param config					
config = ParamConfig(feat_folder="../../Feat/solution", # 特征保存路径 feat_folder="../../Feat/solution"
                     drop_html_flag=True,
                     stemmer_type="porter", # 提取词干方式：porter和snowball
                     cooccurrence_word_exclude_stopword=False)