# @file: sentiment.py
# @author: Hongxiao Jin
# @creat_time: 2020/7/23 17:57

import csv
import re
import sys
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------------- read file -------------------------------------
train_file_name = sys.argv[1]  # get train file name
test_file_name = sys.argv[2]  # get train test name

# train_file_name = 'training.tsv'
# test_file_name = 'test.tsv'
train_file = open(train_file_name, 'r', encoding='gb18030', errors='ignore')
test_file = open(test_file_name, 'r', encoding='gb18030', errors='ignore')

# get train content
train_instance_number = []  # line[0]: index number
train_tweet_text = []  # line[1]: the content in tweet
train_sentiment = []  # line[2]: result show people's sentiment(positive, negative or neutral)
read_train_file = csv.reader(train_file, delimiter='\t')
for line in read_train_file:
    train_instance_number.append(line[0])
    train_tweet_text.append(line[1])
    train_sentiment.append(line[2])

# get test content
test_instance_number = []  # line[0]: index number
test_tweet_text = []  # line[1]: the content in tweet
test_sentiment = []  # line[2]: result show people's sentiment(positive, negative or neutral)
read_test_file = csv.reader(test_file, delimiter='\t')
for line in read_test_file:
    test_instance_number.append(line[0])
    test_tweet_text.append(line[1])
    test_sentiment.append(line[2])


# function: remove urls
def remove_urls(text):
    words = text.split()
    for index in range(0, len(words)):
        if len(words[index]) > 7:
            if words[index][:7] == 'http://' or words[index][:8] == 'https://':
                words[index] = ''
    remove_url = ' '.join(words)
    return remove_url


# function: remove junk characters use Regular expressions
def remove_junk_characters(text):
    text = re.sub(r'[^#@_$%a-zA-Z0-9\s]', '', text)
    return text


# ------------------------------------------ preprocess train_tweet_text -----------------------------------------
# remove urls
for i in range(0, len(train_tweet_text)):
    train_tweet_text[i] = remove_urls(train_tweet_text[i])
# remove junk characters
for i in range(0, len(train_tweet_text)):
    train_tweet_text[i] = remove_junk_characters(train_tweet_text[i])

formal_train_tweet = np.array(train_tweet_text)
# ------------------------------------------ preprocess test_tweet_text -----------------------------------------
# remove urls
for i in range(0, len(test_tweet_text)):
    test_tweet_text[i] = remove_urls(test_tweet_text[i])
# remove junk characters
for i in range(0, len(test_tweet_text)):
    test_tweet_text[i] = remove_junk_characters(test_tweet_text[i])

formal_test_tweet = np.array(test_tweet_text)

# --------------------- create count vectorizer and fit it with training data ----------------------
count = CountVectorizer(token_pattern=r'[#@_$%\b\w\w+\b]{2,}', lowercase=True, max_features=2000)
tweet_train_bag_of_words = count.fit_transform(formal_train_tweet)

# --------------------------- transform the test data with fit_transform ---------------------------
tweet_test_bag_of_words = count.transform(formal_test_tweet)

# ------------------------------------- Multinomial Naive Bayes set model ----------------------------
clf = MultinomialNB()
model = clf.fit(tweet_train_bag_of_words, train_sentiment)
MNB_predicted_sentiment = model.predict(tweet_test_bag_of_words)
for i in range(0, len(MNB_predicted_sentiment)):
    print(test_instance_number[i], MNB_predicted_sentiment[i])
