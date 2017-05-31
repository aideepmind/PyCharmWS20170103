
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
        ('count_of_question1_unigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question1_unigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question1_unigram', SimpleTransform()),

        ('count_of_question1_bigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question1_bigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question1_bigram', SimpleTransform()),

        ('count_of_question1_trigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_question1_trigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question1_trigram', SimpleTransform()),

        # ('count_of_digit_in_question1', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_digit_in_question1', SimpleTransform()),

        ('count_of_question2_unigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question2_unigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question2_unigram', SimpleTransform()),

        ('count_of_question2_bigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question2_bigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question2_bigram', SimpleTransform()),

        # ('count_of_question2_trigram', SimpleTransform(config.count_feat_transform)),
        # ('count_of_unique_question2_trigram', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_unique_question2_trigram', SimpleTransform()),

        # ('count_of_digit_in_question2', SimpleTransform(config.count_feat_transform)),
        # ('ratio_of_digit_in_question2', SimpleTransform()),

        # ('count_of_question1_unigram_in_question2', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question1_unigram_in_question2', SimpleTransform()),

        # ('count_of_question2_unigram_in_question1', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question2_unigram_in_question1', SimpleTransform()),

        ('count_of_question1_bigram_in_question2', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question1_bigram_in_question2', SimpleTransform()),

        # ('count_of_question2_bigram_in_question1', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question2_bigram_in_question1', SimpleTransform()),

        # ('count_of_question1_trigram_in_question2', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question1_trigram_in_question2', SimpleTransform()),

        # ('count_of_question2_trigram_in_question1', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question2_trigram_in_question1', SimpleTransform()),

        ##############
        ## Position ##
        ##############
        # ('pos_of_question1_unigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_unigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_unigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_unigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_unigram_in_question2_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question1_unigram_in_question2_min', SimpleTransform()),
        ('normalized_pos_of_question1_unigram_in_question2_mean', SimpleTransform()),
        ('normalized_pos_of_question1_unigram_in_question2_median', SimpleTransform()),
        ('normalized_pos_of_question1_unigram_in_question2_max', SimpleTransform()),
        ('normalized_pos_of_question1_unigram_in_question2_std', SimpleTransform()),

        # ('pos_of_question2_unigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_unigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_unigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_unigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question2_unigram_in_question1_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question2_unigram_in_question1_min', SimpleTransform()),
        ('normalized_pos_of_question2_unigram_in_question1_mean', SimpleTransform()),
        # ('normalized_pos_of_question2_unigram_in_question1_median', SimpleTransform()),
        ('normalized_pos_of_question2_unigram_in_question1_max', SimpleTransform()),
        ('normalized_pos_of_question2_unigram_in_question1_std', SimpleTransform()),

        ('pos_of_question1_bigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question1_bigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question1_bigram_in_question2_std', SimpleTransform(config.count_feat_transform)),

        ('normalized_pos_of_question1_bigram_in_question2_min', SimpleTransform()),
        ('normalized_pos_of_question1_bigram_in_question2_mean', SimpleTransform()),
        ('normalized_pos_of_question1_bigram_in_question2_median', SimpleTransform()),
        ('normalized_pos_of_question1_bigram_in_question2_max', SimpleTransform()),
        ('normalized_pos_of_question1_bigram_in_question2_std', SimpleTransform()),

        # ('pos_of_question2_bigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question2_bigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question2_bigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question2_bigram_in_question1_std', SimpleTransform(config.count_feat_transform)),

        ('normalized_pos_of_question2_bigram_in_question1_min', SimpleTransform()),
        ('normalized_pos_of_question2_bigram_in_question1_mean', SimpleTransform()),
        ('normalized_pos_of_question2_bigram_in_question1_median', SimpleTransform()),
        ('normalized_pos_of_question2_bigram_in_question1_max', SimpleTransform()),
        ('normalized_pos_of_question2_bigram_in_question1_std', SimpleTransform()),

        # ('pos_of_question1_trigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question1_trigram_in_question2_min', SimpleTransform()),
        ('normalized_pos_of_question1_trigram_in_question2_mean', SimpleTransform()),
        ('normalized_pos_of_question1_trigram_in_question2_median', SimpleTransform()),
        ('normalized_pos_of_question1_trigram_in_question2_max', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_std', SimpleTransform()),

        # ('pos_of_question2_trigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_std', SimpleTransform(config.count_feat_transform)),

        # ('normalized_pos_of_question2_trigram_in_question1_min', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_mean', SimpleTransform()),
        ('normalized_pos_of_question2_trigram_in_question1_median', SimpleTransform()),
        ('normalized_pos_of_question2_trigram_in_question1_max', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_std', SimpleTransform()),

        ##################
        ## jaccard coef ##
        ##################
        ('jaccard_coef_of_unigram_between_question1_question2', SimpleTransform()),
        ('jaccard_coef_of_bigram_between_question1_question2', SimpleTransform()),
        ('jaccard_coef_of_trigram_between_question1_question2', SimpleTransform()),

        ################
        ## jdice dist ##
        ################
        ('dice_dist_of_unigram_between_question1_question2', SimpleTransform()),
        ('dice_dist_of_bigram_between_question1_question2', SimpleTransform()),
        ('dice_dist_of_trigram_between_question1_question2', SimpleTransform()),

        ############
        ## TF-IDF ##
        ############
        # ('question1_tfidf_common_vocabulary', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary', SimpleTransform()),

        ('question1_tfidf_common_vocabulary_question2_tfidf_common_vocabulary_tfidf_cosine_sim', SimpleTransform()),
        # ('question1_bow_common_vocabulary_question2_bow_common_vocabulary_bow_cosine_sim', SimpleTransform()),

        # ('question1_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('question1_tfidf_common_vocabulary_common_svd150', SimpleTransform()),+
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

        ###############################################
        ## sentence hash and frequency and intersect ##
        ###############################################
        ('sentence_hash_of_question1', SimpleTransform()),
        ('sentence_hash_of_question2', SimpleTransform()),

        ('sentence_freq_of_question1', SimpleTransform()),
        ('sentence_freq_of_question2', SimpleTransform()),

        ('sentence_sum_of_question1_question2', SimpleTransform()),
        ('sentence_mean_of_question1_question2', SimpleTransform()),
        ('sentence_div_of_question1_question2', SimpleTransform()),

        ('sentence_intersect_of_question1_question2', SimpleTransform()),
        ('sentence_intersect_ratio_of_question1_question2_div_sum', SimpleTransform()),

        #########################################
        ## word2vec distance and some features ##
        #########################################
        ('len_of_question1', SimpleTransform()),
        ('len_of_question2', SimpleTransform()),
        ('diff_len_of_question1_question2', SimpleTransform()),
        ('len_of_char_question1', SimpleTransform()),
        ('len_of_char_question2', SimpleTransform()),
        ('len_of_word_question1', SimpleTransform()),
        ('len_of_word_question2', SimpleTransform()),
        ('common_words_question1_question2', SimpleTransform()),
        ('fuzz_qratio_question1_question2', SimpleTransform()),
        ('fuzz_WRatio_question1_question2', SimpleTransform()),
        ('fuzz_partial_ratio_question1_question2', SimpleTransform()),
        ('fuzz_partial_token_set_ratio_question1_question2', SimpleTransform()),
        ('fuzz_partial_token_sort_ratio_question1_question2', SimpleTransform()),
        ('fuzz_token_set_ratio_question1_question2', SimpleTransform()),
        ('fuzz_token_sort_ratio_question1_question2', SimpleTransform()),
        ('wmdistance_question1_question2', SimpleTransform()),
        ('norm_wmdistance_question1_question2', SimpleTransform()),
        ('cosine_distance_between_question1_question2', SimpleTransform()),
        ('cityblock_distance_between_question1_question2', SimpleTransform()),
        ('jaccard_distance_between_question1_question2', SimpleTransform()),
        ('canberra_distance_between_question1_question2', SimpleTransform()),
        ('euclidean_distance_between_question1_question2', SimpleTransform()),
        ('minkowski_distance_between_question1_question2', SimpleTransform()),
        ('braycurtis_distance_between_question1_question2', SimpleTransform()),
        ('skew_of_question1_vec', SimpleTransform()),
        ('skew_of_question2_vec', SimpleTransform()),
        ('kur_of_question1_vec', SimpleTransform()),
        ('kur_of_question2_vec', SimpleTransform()),
        ('tfidf_wm_stops_question1_question2', SimpleTransform()),
        ('jaccard_question1_question2', SimpleTransform()),
        ('wc_diff_question1_question2', SimpleTransform()),
        ('wc_ratio_question1_question2', SimpleTransform()),
        ('wc_diff_unique_question1_question2', SimpleTransform()),
        ('wc_ratio_unique_question1_question2', SimpleTransform()),
        ('wc_diff_unq_stop_question1_question2', SimpleTransform()),
        ('wc_ratio_unique_stop_question1_question2', SimpleTransform()),
        ('same_start_question1_question2', SimpleTransform()),
        ('char_diff_unq_stop_question1_question2', SimpleTransform()),
        ('total_unique_words_question1_question2', SimpleTransform()),
        ('total_unq_words_stop_question1_question2', SimpleTransform()),
        ('char_ratio_question1_question2', SimpleTransform()),


    ]

    gen_info(feat_path_name="LSA_and_stats_feat_May30")
    combine_feat(feat_names, feat_path_name="LSA_and_stats_feat_May30")