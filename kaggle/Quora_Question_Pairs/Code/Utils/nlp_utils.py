
"""
__file__

    nlp_utils.py

__description__

    This file provides functions to perform NLP task, e.g., TF-IDF and POS tagging.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import re
import sys
import nltk
from bs4 import BeautifulSoup
from replacer import CsvWordReplacer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
sys.path.append("../")
from param_config import config

################
## Stop Words ##
################
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)


##############
## Stemming ##
##############
if config.stemmer_type == "porter":
    english_stemmer = nltk.stem.PorterStemmer()
elif config.stemmer_type == "snowball":
    english_stemmer = nltk.stem.SnowballStemmer('english')
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

if __name__ == '__main__':
    stem_tokens(['aed'], english_stemmer)


#############
## POS Tag ##
#############
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
def pos_tag_text(line,
                 token_pattern=token_pattern,
                 exclude_stopword=config.cooccurrence_word_exclude_stopword,
                 encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    for name in ["query", "product_title", "product_description"]:
        l = line[name]
        ## tokenize
        tokens = [x.lower() for x in token_pattern.findall(l)]
		## stem
        tokens = stem_tokens(tokens, english_stemmer)
        if exclude_stopword:
            tokens = [x for x in tokens if x not in stopwords]
        tags = pos_tag(tokens)
        tags_list = [t for w,t in tags]
        tags_str = " ".join(tags_list)
        #print tags_str
        line[name] = tags_str
    return line

    
############
## TF-IDF ##
############
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
   
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 3
def getTFV(token_pattern = token_pattern,
           norm = tfidf__norm,
           max_df = tfidf__max_df,
           min_df = tfidf__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words = stop_words, norm=norm, vocabulary=vocabulary)
    return tfv
   

#########
## BOW ##
#########
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
   
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
bow__max_df = 0.75
bow__min_df = 3
def getBOW(token_pattern = token_pattern,
           max_df = bow__max_df,
           min_df = bow__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    bow = StemmedCountVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range,
                                 stop_words = stop_words, vocabulary=vocabulary)
    return bow


################
## Text Clean ##
################
## synonym replacer
replacer = CsvWordReplacer('%s/synonyms.csv' % config.data_folder)
## other replace dict
## such dict is found by exploring the training data
replace_dict = {
    r"what's": "what is ",
    r"What's": "what is ",
    r"\'s": " ",
    r"\'ve": " have ",
    r"can't": "cannot ",
    r"n't": " not ",
    r"I'm": "I am",
    r" m ": " am ",
    r"\'re": " are ",
    r"\'d": " would ",
    r"\'ll": " will ",
    r"60k": " 60000 ",
    r" e g ": " eg ",
    r" b g ": " bg ",
    r"\0s": "0",
    r" 9 11 ": "911",
    r"e-mail": "email",
    r"\s{2,}": " ",
    r"quikly": "quickly",
    r" usa ": " America ",
    r" USA ": " America ",
    r" u s ": " America ",
    r" uk ": " England ",
    r" UK ": " England ",
    r"india": "India",
    r"switzerland": "Switzerland",
    r"china": "China",
    r"chinese": "Chinese",
    r"imrovement": "improvement",
    r"intially": "initially",
    r"quora": "Quora",
    r" dms ": "direct messages ",
    r"demonitization": "demonetization",
    r"actived": "active",
    r"kms": " kilometers ",
    r"KMs": " kilometers ",
    r" cs ": " computer science ",
    r" upvotes ": " up votes ",
    r" iPhone ": " phone ",
    r"\0rs ": " rs ",
    r"calender": "calendar",
    r"ios": "operating system",
    r"gps": "GPS",
    r"gst": "GST",
    r"programing": "programming",
    r"bestfriend": "best friend",
    r"dna": "DNA",
    r"III": "3",
    r"the US": "America",
    r"Astrology": "astrology",
    r"Method": "method",
    r"Find": "find",
    r"banglore": "Banglore",
    r" J K ": " JK ",
    r"[^A-Za-z0-9^,!.\/'+-=]": " "
}

def clean_text(line, drop_html_flag=False):
    names = ["question1", "question2"]
    for name in names:
        l = line[name]
        if drop_html_flag:
            l = drop_html(l)
        l = l.lower()

        ## replace other words
        for k,v in replace_dict.items():
            l = re.sub(k, v, l)
        l = l.split(" ")

        ## replace synonyms
        l = replacer.replace(l)
        l = " ".join(l)
        line[name] = l
    return line
    

###################
## Drop html tag ##
###################
def drop_html(html):
    return BeautifulSoup(html).get_text(separator=" ")