import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import datetime
# import seaborn as sns
# %matplotlib inline

# pal = sns.color_palette()

# print('# File sizes')
# for f in os.listdir('../input'):
#     if 'zip' not in f:
#         print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')

df_train = pd.read_csv('kaggle/Quora_Question_Pairs/Data/train.csv')
# df_train.head()

# print('Total number of question pairs for training: {}'.format(len(df_train)))
# print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
# qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
# print('Total number of questions in the training data: {}'.format(len(
#     np.unique(qids))))
# print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))
#
# plt.figure(figsize=(12, 5))
# plt.hist(qids.value_counts(), bins=50)
# plt.yscale('log', nonposy='clip')
# plt.title('Log-Histogram of question appearance counts')
# plt.xlabel('Number of occurences of question')
# plt.ylabel('Number of questions')

# from sklearn.metrics import log_loss
#
# p = df_train['is_duplicate'].mean() # Our predicted probability
# print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))
#
df_test = pd.read_csv('kaggle/Quora_Question_Pairs/Data/test.csv')
# sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})
# sub.to_csv('naive_submission.csv', index=False)
# sub.head()
#
# df_test = pd.read_csv('../input/test.csv')
# df_test.head()
#
# print('Total number of question pairs for testing: {}'.format(len(df_test)))

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
# test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

# dist_train = train_qs.apply(len)
# dist_test = test_qs.apply(len)
# plt.figure(figsize=(15, 10))
# plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
# plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
# plt.title('Normalised histogram of character count in questions', fontsize=15)
# plt.legend()
# plt.xlabel('Number of characters', fontsize=15)
# plt.ylabel('Probability', fontsize=15)
#
# print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(),
#                           dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
#
# dist_train = train_qs.apply(lambda x: len(x.split(' ')))
# dist_test = test_qs.apply(lambda x: len(x.split(' ')))

# plt.figure(figsize=(15, 10))
# plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')
# plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='test')
# plt.title('Normalised histogram of word count in questions', fontsize=15)
# plt.legend()
# plt.xlabel('Number of words', fontsize=15)
# plt.ylabel('Probability', fontsize=15)
#
# print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(),
#                           dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
#
# from wordcloud import WordCloud
# cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
# plt.figure(figsize=(20, 15))
# plt.imshow(cloud)
# plt.axis('off')


# qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
# math = np.mean(train_qs.apply(lambda x: '[math]' in x))
# fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
# capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
# capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
# numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))
#
# print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
# print('Questions with [math] tags: {:.2f}%'.format(math * 100))
# print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
# print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
# print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
# print('Questions with numbers: {:.2f}%'.format(numbers * 100))

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

replace_dict = {
    r"what's": "what is ",
    r"\'s": " ",
    r"\'ve": " have ",
    r"can't": "can not ",
    r"n't": " not ",
    r"i'm": "i am",
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
    r" usa ": " america ",
    r" u s ": " america ",
    r" uk ": " england ",
    r" us ": "america",
    # r"india": "India",
    # r"switzerland": "Switzerland",
    # r"china": "China",
    # r"chinese": "Chinese",
    r"imrovement": "improvement",
    r"intially": "initially",
    # r"quora": "Quora",
    r" dms ": "direct messages ",
    r"demonitization": "demonetization",
    r"actived": "active",
    r"kms": " kilometers ",
    r" cs ": " computer science ",
    r" upvotes ": " up votes ",
    r" iphone ": " phone ",
    r"\0rs ": " rs ",
    r"calender": "calendar",
    # r"ios": "operating system",
    # r"gps": "GPS",
    # r"gst": "GST",
    r"programing": "programming",
    r"bestfriend": "best friend",
    # r"dna": "DNA",
    r"III": "3",
    # r"Astrology": "astrology",
    # r"Method": "method",
    # r"Find": "find",
    # r"banglore": "Banglore",
    r" j k ": " jk ",
    r"[^A-Za-z0-9^,!.\/'+-=]": " "  # 删除特殊字符
}
import re
def clean_text(line):
    names = ["question1", "question2"]
    for name in names:
        l = line[name]
        l = str(l).lower()

        ## replace other words
        for k,v in replace_dict.items():
            l = re.sub(k, v, l)

        line[name] = l
    return line

import nltk
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, WhitespaceTokenizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

stemmer_type = "snowball"
if stemmer_type == "porter":
    english_stemmer = nltk.stem.PorterStemmer()
elif stemmer_type == "snowball":
    english_stemmer = nltk.stem.SnowballStemmer('english')
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        try:
            stemmed.append(stemmer.stem(token))
        except (e) as e:    # python 3.6 可能会报错，3.5.2 没问题
            print('exception is: ', e)
    return stemmed
# 1.     CC      Coordinating conjunction
# 2.     CD     Cardinal number
# 3.     DT     Determiner
# 4.     EX     Existential there
# 5.     FW     Foreign word
# 6.     IN     Preposition or subordinating conjunction
# 7.     JJ     Adjective
# 8.     JJR     Adjective, comparative
# 9.     JJS     Adjective, superlative
# 10.     LS     List item marker
# 11.     MD     Modal
# 12.     NN     Noun, singular or mass
# 13.     NNS     Noun, plural
# 14.     NNP     Proper noun, singular
# 15.     NNPS     Proper noun, plural
# 16.     PDT     Predeterminer
# 17.     POS     Possessive ending
# 18.     PRP     Personal pronoun
# 19.     PRP$     Possessive pronoun
# 20.     RB     Adverb
# 21.     RBR     Adverb, comparative
# 22.     RBS     Adverb, superlative
# 23.     RP     Particle
# 24.     SYM     Symbol
# 25.     TO     to
# 26.     UH     Interjection
# 27.     VB     Verb, base form
# 28.     VBD     Verb, past tense
# 29.     VBG     Verb, gerund or present participle
# 30.     VBN     Verb, past participle
# 31.     VBP     Verb, non-3rd person singular present
# 32.     VBZ     Verb, 3rd person singular present
# 33.     WDT     Wh-determiner
# 34.     WP     Wh-pronoun
# 35.     WP$     Possessive wh-pronoun
# 36.     WRB     Wh-adverb
wordnet_tags = ['n', 'v']
lemmatizer = WordNetLemmatizer()
def lemmatize(token, tag):
    if tag[0].lower() in wordnet_tags:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token

token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r'\w{1,}'
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
def preprocess_data(line, token_pattern=token_pattern, encode_digit=False):
    line = str(line).lower()
    # token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    ## tokenize
    # tokens = [x.lower() for x in token_pattern.findall(line)]
    # tokens = line.split()
    # tokens = WhitespaceTokenizer().tokenize(line)
    # # stem
    # tokens_stemmed = stem_tokens(tokens, english_stemmer)
    # if True:
    #     tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]

    tagged_corpus = pos_tag(word_tokenize(line))
    tokens_lemmatize = [lemmatize(token, tag) for token, tag in tagged_corpus]
    return tokens_lemmatize
global ind
ind = 0
def word_match_share(row):
    row = clean_text(row)
    global ind
    q1words = {}
    q2words = {}
    if ind < 5:
        ind += 1
        print(preprocess_data(str(row['question1'])))
    for word in preprocess_data(str(row['question1'])):# str(row['question1']).lower().split()
        if word not in stops:
            q1words[word] = 1
    for word in preprocess_data(str(row['question2'])):
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

# plt.figure(figsize=(15, 5))
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
# plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
# plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
# plt.legend()
# plt.title('Label distribution over word_match_share', fontsize=15)
# plt.xlabel('word_match_share', fontsize=15)

from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=3):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

# print('Most common words and weights: \n')
# print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
# print('\nLeast common words and weights: ')
# (sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])


def tfidf_word_match_share(row):
    row = clean_text(row)
    q1words = {}
    q2words = {}
    for word in preprocess_data(str(row['question1'])):
        if word not in stops:
            q1words[word] = 1
    for word in preprocess_data(str(row['question2'])):
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

# plt.figure(figsize=(15, 5))
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
# plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
# plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
# plt.legend()
# plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
# plt.xlabel('word_match_share', fontsize=15)

# from sklearn.metrics import roc_auc_score
# print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))
# print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))

# First we create our training and testing data
x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)



# train_comb = train_comb.drop(['id', 'is_duplicate'], axis = 1)
# test_comb = test_comb.drop(['id'], axis=1)
# x_train = pd.concat([x_train, train_comb], axis=1)
# x_test = pd.concat([x_test, test_comb], axis=1)


y_train = df_train['is_duplicate'].values
# pos_new/(pos_new + neg) = 0.165 ->  need reduce (pos - pos_new)
import numpy as np
need_reduce = y_train[y_train == 1].shape[0] - np.round(0.165 * y_train[y_train == 0].shape[0] / (1 - 0.165))
pos_ids = df_train['id'][y_train == 1]
random_seed = datetime.datetime.now().year + 1000 * (2 + 1)
pos_ids = pos_ids.sample(int(need_reduce), random_state=random_seed)
flags = df_train['id'].apply(lambda x: False if x in pos_ids else True)
y_train = df_train['is_duplicate'][flags].values
x_train = x_train[flags]






# pos_train = x_train[y_train == 1]
# neg_train = x_train[y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
# p = 0.165
# scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
# while scale > 1:
#     neg_train = pd.concat([neg_train, neg_train])
#     scale -=1
# neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
# print(len(pos_train) / (len(pos_train) + len(neg_train)))
#
# x_train = pd.concat([pos_train, neg_train])
# y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
# del pos_train, neg_train

# Finally, we split some of the data off for validation
from sklearn.cross_validation import train_test_split
# weight = np.abs(y_train - 1.2 + y_train.mean())
x_train2, x_valid, y_train2, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4



d_train = xgb.DMatrix(x_train2, label=y_train2)
d_valid = xgb.DMatrix(x_valid, label=y_valid)   # , weight=w_valid
d_test = xgb.DMatrix(x_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 600, watchlist, early_stopping_rounds=50, verbose_eval=10)


preds = bst.predict(d_test)
result = pd.DataFrame({"test_id": df_test['test_id'], "is_duplicate": preds})
result.to_csv("kaggle/Quora_Question_Pairs/temp/xgb_train_result.csv", index=False)
result.head()


from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import log_loss
from sklearn.preprocessing import  MinMaxScaler
# param_grid_lasso = {
#     'alpha': [0.0005, 0.001, 0.005, 0.006]
# }
# lasso = GridSearchCV(
#     estimator=Lasso(
#         alpha=0.0005,
#         fit_intercept=True,
#         normalize=False,
#         precompute=False,
#         copy_X=True,
#         max_iter=2000,
#         tol=1e-4,
#         warm_start=False,
#         positive=False,
#         random_state=None,
#         selection='cyclic'),
#     param_grid=param_grid_lasso,
#     # scoring='neg_log_loss',
#     n_jobs=4,
#     iid=False,
#     cv=5)
# lasso.fit(x_train, y_train)
# # lasso.predict_proba()
# y_preds = lasso.predict(x_train)
# lasso.grid_scores_, lasso.best_params_, lasso.best_score_
# min_max_scaler = MinMaxScaler()
# y_preds = min_max_scaler.fit_transform(y_preds)
# score = log_loss(y_train, y_preds)

from sklearn.linear_model import LogisticRegression
param_grid = {
    "C": [0.01, 0.05, 0.1, 0.5, 1.0]}
logistic = GridSearchCV(
    estimator=LogisticRegression(
        penalty='l2', dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver='liblinear',
        max_iter=100,
        multi_class='ovr',
        verbose=0,
        warm_start=False,
        n_jobs=1
    ),
    cv=5,
    param_grid=param_grid)
logistic.fit(x_train, y_train)
logistic.grid_scores_, logistic.best_params_, logistic.best_score_