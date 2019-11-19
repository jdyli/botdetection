import xmltodict
import os
from feature_extraction import *
from preprocessing import preprocess_domain
import pandas as pd
import numpy as np
from collections import defaultdict
from lsa import latent_semantic_analysis, frequency_distribution
import re

def create_dataframe(column_names):
    dirname = os.path.abspath(os.path.curdir) + '/pan19-author-profiling-training-2019-02-18/en/'
    tweets_df = pd.DataFrame(columns = column_names)
    
    tables = [line.rstrip('\n').replace(':', ' ').replace(':', ' ').split() for line in open(dirname + 'truth-train.txt')]
    tweets_df['username'] = pd.Series(table[0] for table in tables)
    tweets_df['label'] = pd.Series(table[1] for table in tables)
    return (tweets_df, tables)

# used for the statistical features
def open_for_exploring():
    dirname = os.path.abspath(os.path.dirname(__file__) + 'pan19-author-profiling-training-2019-02-18/en/')
    (tweets_df, tables) = create_dataframe(['username', 'hashtags', 'retweets', 'url', 'punctuations', 'sentence_length','emoji', 'label'])
    for table in tables:
        files = [dirname + '/' + i for i in os.listdir(dirname) if i.endswith(table[0] + ".xml")]
        for file in files:
            with open(file, encoding="utf-8") as fd:
                tweets = xmltodict.parse(fd.read(), encoding='unicode-escape')
                tweets = pd.Series(tweets['author']['documents']['document'])
                tweets_df['hashtags'].loc[tweets_df['username'] == table[0]] = count_hashtags(tweets)
                tweets_df['retweets'].loc[tweets_df['username'] == table[0]] = count_retweets(tweets)
                tweets_df['url'].loc[tweets_df['username'] == table[0]] = count_url(tweets)
                tweets_df['punctuations'].loc[tweets_df['username'] == table[0]] = count_punctuations(tweets)
                tweets_df['sentence_length'].loc[tweets_df['username'] == table[0]] = count_sentence_length(tweets)
                tweets_df['emoji'].loc[tweets_df['username'] == table[0]] = count_emojis(tweets)
    print(tweets_df)

    df = pd.DataFrame(tweets_df)
    df.columns = ["username", "hashtags", 'retweets', 'url', 'punctuations', 'sentence_length','emoji', "label"]
    df.to_csv("test_raw.csv", index=0)

# used for the NLP features
def open_for_content_features():
    
    (tweets_df, tables) = create_dataframe(['username', 'levenshtein', 'polarity', 'subjectivity', 'possession', 'flesch_reading', 'label'])
    dirname = os.path.abspath(os.path.curdir) + '/pan19-author-profiling-training-2019-02-18/en/'
    for table in tables:
        files = [dirname + i for i in os.listdir(dirname) if i.endswith(table[0] + ".xml")]
        for file in files:
            with open(file, encoding="utf-8") as fd:
                tweets = xmltodict.parse(fd.read(), encoding='unicode-escape')
                tweets = tweets['author']['documents']['document']
                
                tweets2 = pd.Series(preprocess_domain(tweets))
                tweets_df['levenshtein'].loc[tweets_df['username'] == table[0]] = levenshtein_distance(tweets2)
                (polarity, subjectivity) = sentiment_analysis(tweets2)
                tweets_df['polarity'].loc[tweets_df['username'] == table[0]] = polarity
                tweets_df['subjectivity'].loc[tweets_df['username'] == table[0]] = subjectivity
                tweets_df['possession'].loc[tweets_df['username'] == table[0]] = possession_analysis(tweets2)
                tweets_df['flesch_reading'].loc[tweets_df['username'] == table[0]] = flesch_reading_ease(tweets2)
    print(tweets_df)

    df = pd.DataFrame(tweets_df)
    df.columns = ["username", 'levenshtein', 'polarity', 'subjectivity', 'possession', 'flesch_reading', "label"]
    df.to_csv("test_sentence.csv", index=0)

# (un)comment based on what kind of features you would like to run
open_for_content_features()
#open_for_exploring()