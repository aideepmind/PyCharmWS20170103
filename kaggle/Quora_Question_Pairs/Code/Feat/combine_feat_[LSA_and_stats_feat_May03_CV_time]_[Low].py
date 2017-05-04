
"""
__file__
    
    combine_feat_[LSA_and_stats_feat_Apr25]_[Low].py

__description__

    This file generates one combination of feature set (Low).

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import sys
sys.path.append("../")
from param_config import config
from gen_info import gen_info
from combine_feat import combine_feat, SimpleTransform

            
if __name__ == "__main__":

    feat_names = [

        ################
        ## Word count ##
        ################
        # ('count_of_question1_unigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question1_unigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question1_unigram', SimpleTransform()),
        #
        # ('count_of_question1_bigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question1_bigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question1_bigram', SimpleTransform()),
        #
        # ('count_of_question1_trigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question1_trigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question1_trigram', SimpleTransform()),
        #
        # ('count_of_digit_in_question1', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_digit_in_question1', SimpleTransform()),
        #
        # ('count_of_question2_unigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question2_unigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question2_unigram', SimpleTransform()),
        #
        # ('count_of_question2_bigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question2_bigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question2_bigram', SimpleTransform()),
        #
        # ('count_of_question2_trigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question2_trigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question2_trigram', SimpleTransform()),
        #
        # ('count_of_digit_in_question2', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_digit_in_question2', SimpleTransform()),
        #
        # ('count_of_question1_unigram_in_question2', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_question1_unigram_in_question2', SimpleTransform()),
        #
        # ('count_of_question2_unigram_in_question1', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_question2_unigram_in_question1', SimpleTransform()),
        #
        # ('count_of_question1_bigram_in_question2', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_question1_bigram_in_question2', SimpleTransform()),
        #
        # ('count_of_question2_bigram_in_question1', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_question2_bigram_in_question1', SimpleTransform()),
        #
        # ('count_of_question1_trigram_in_question2', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_question1_trigram_in_question2', SimpleTransform()),
        #
        # ('count_of_question2_trigram_in_question1', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_question2_trigram_in_question1', SimpleTransform()),
        #
        # ('count_of_question1_digit_subtract_question2', SimpleTransform()),
        # ('count_of_question1_unigram_subtract_question2', SimpleTransform()),
        #
        # ('count_of_question2_digit_subtract_question1', SimpleTransform()),
        # ('count_of_question2_unigram_subtract_question1', SimpleTransform()),

        ##############
        ## Position ##
        ##############
        # ('pos_of_question1_unigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_unigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_unigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_unigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_unigram_in_question2_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question1_unigram_in_question2_min', SimpleTransform()),
        # ('normalized_pos_of_question1_unigram_in_question2_mean', SimpleTransform()),
        # ('normalized_pos_of_question1_unigram_in_question2_median', SimpleTransform()),
        # ('normalized_pos_of_question1_unigram_in_question2_max', SimpleTransform()),
        # ('normalized_pos_of_question1_unigram_in_question2_std', SimpleTransform()),

        # ('pos_of_question2_unigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_unigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_unigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_unigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_unigram_in_question1_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question2_unigram_in_question1_min', SimpleTransform()),
        # ('normalized_pos_of_question2_unigram_in_question1_mean', SimpleTransform()),
        # ('normalized_pos_of_question2_unigram_in_question1_median', SimpleTransform()),
        # ('normalized_pos_of_question2_unigram_in_question1_max', SimpleTransform()),
        # ('normalized_pos_of_question2_unigram_in_question1_std', SimpleTransform()),

        # ('pos_of_question1_bigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question1_bigram_in_question2_min', SimpleTransform()),
        # ('normalized_pos_of_question1_bigram_in_question2_mean', SimpleTransform()),
        # ('normalized_pos_of_question1_bigram_in_question2_median', SimpleTransform()),
        # ('normalized_pos_of_question1_bigram_in_question2_max', SimpleTransform()),
        # ('normalized_pos_of_question1_bigram_in_question2_std', SimpleTransform()),

        # ('pos_of_question2_bigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question2_bigram_in_question1_min', SimpleTransform()),
        # ('normalized_pos_of_question2_bigram_in_question1_mean', SimpleTransform()),
        # ('normalized_pos_of_question2_bigram_in_question1_median', SimpleTransform()),
        # ('normalized_pos_of_question2_bigram_in_question1_max', SimpleTransform()),
        # ('normalized_pos_of_question2_bigram_in_question1_std', SimpleTransform()),

        # ('pos_of_question1_trigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question1_trigram_in_question2_min', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_mean', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_median', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_max', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_std', SimpleTransform()),

        # ('pos_of_question2_trigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question2_trigram_in_question1_min', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_mean', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_median', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_max', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_std', SimpleTransform()),

        ##################
        ## jaccard coef ##
        ##################
        # ('jaccard_coef_of_unigram_between_question1_question2', SimpleTransform()),
        # ('jaccard_coef_of_bigram_between_question1_question2', SimpleTransform()),
        # ('jaccard_coef_of_trigram_between_question1_question2', SimpleTransform()),

        ################
        ## jdice dist ##
        ################
        # ('dice_dist_of_unigram_between_question1_question2', SimpleTransform()),
        # ('dice_dist_of_bigram_between_question1_question2', SimpleTransform()),
        # ('dice_dist_of_trigram_between_question1_question2', SimpleTransform()),

        ############
        ## TF-IDF ##
        ############
        # ('question1_tfidf_common_vocabulary', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary', SimpleTransform()),

        # ('question1_tfidf_common_vocabulary_question2_tfidf_common_vocabulary_tfidf_cosine_sim', SimpleTransform()),
        # ('question1_bow_common_vocabulary_question2_bow_common_vocabulary_bow_cosine_sim', SimpleTransform()),

        # ('question1_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('question1_tfidf_common_vocabulary_common_svd150', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_common_svd150', SimpleTransform()),

        # ('question1_bow_common_vocabulary_common_svd100', SimpleTransform()),
        # ('question2_bow_common_vocabulary_common_svd100', SimpleTransform()),
        # ('question1_bow_common_vocabulary_common_svd150', SimpleTransform()),
        # ('question2_bow_common_vocabulary_common_svd150', SimpleTransform()),

        # ('question1_tfidf_common_vocabulary_question2_tfidf_common_vocabulary_tfidf_common_svd100_cosine_sim', SimpleTransform()),
        # ('question1_tfidf_common_vocabulary_question2_tfidf_common_vocabulary_tfidf_common_svd150_cosine_sim', SimpleTransform()),

        # ('question1_bow_common_vocabulary_question2_bow_common_vocabulary_bow_common_svd100_cosine_sim', SimpleTransform()),
        # ('question1_bow_common_vocabulary_question2_bow_common_vocabulary_bow_common_svd150_cosine_sim', SimpleTransform()),

        # ('question1_tfidf_common_vocabulary_individual_svd100', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_individual_svd100', SimpleTransform()),
        # ('question1_tfidf_common_vocabulary_individual_svd150', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_individual_svd150', SimpleTransform()),

        # ('question1_bow_common_vocabulary_individual_svd100', SimpleTransform()),
        # ('question2_bow_common_vocabulary_individual_svd100', SimpleTransform()),
        # ('question1_bow_common_vocabulary_individual_svd150', SimpleTransform()),
        # ('question2_bow_common_vocabulary_individual_svd150', SimpleTransform()),

        #########################
        ## Cooccurrence TF-IDF ##
        #########################
        # ('question1_unigram_question2_unigram', SimpleTransform()),
        # ('question1_unigram_question2_bigram', SimpleTransform()),
        # ('question1_bigram_question2_unigram', SimpleTransform()),
        # ('question1_bigram_question2_bigram', SimpleTransform()),

        ######################
        ## word match share ##
        ######################
        ('ratio_of_question1_question2_unigram_share', SimpleTransform()),
        ('ratio_of_question1_question2_unigram_share_tfidf', SimpleTransform()),
    ]

    gen_info(feat_path_name="LSA_and_stats_feat_May03_CV_Time")
    combine_feat(feat_names, feat_path_name="LSA_and_stats_feat_May03_CV_Time")