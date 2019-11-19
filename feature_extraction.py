import string
from Levenshtein import distance
import pandas as pd
import itertools
from textblob import TextBlob
import nltk
import re
import textstat

def count_hashtags(tweets):
    count_hashtags = sum(tweets.str.count('#')) #take average or not?
    return count_hashtags

def count_retweets(tweets):
    count_retweet = sum(tweets.str.count('RT @'))
    return count_retweet

def count_url(tweet):
    count_url = sum(tweet.str.count('http://'))
    count_url += sum(tweet.str.count('https://'))
    return count_url

def count_punctuations(tweets):
    a_punct = 0
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    for element in tweets:
        #a_chars +=  count(element, string.ascii_letters) 
        a_punct += count(element, string.punctuation)
    return a_punct / len(tweets)

def levenshtein_distance(tweets):
    result = 0
    tweet_combos = itertools.combinations(tweets,2)
    num_combos = len(list(itertools.combinations(tweets, 2)))
    for each in tweet_combos:
        result += distance(each[0], each[1])
    return result / num_combos

def count_sentence_length(tweets):
    count_length = sum(tweets.str.count(' ')) / len(tweets)
    return count_length

def count_emojis(tweets):
    result = 0
    for i in tweets:
        result += len(re.findall('(\\\\[u,U][0-9a-zA-Z]+)', i))
    return result

def pos_tagging(tweet, types):
    tokens = nltk.word_tokenize(tweet)
    tagged = nltk.pos_tag(tokens)
    if (types == 'sentiment'):
        a = [item[0] for item in tagged if item[1] == 'JJ' or item[1] == 'JJR' or item[1] == 'JJS']
        a = ' '.join(str(i) for i in a)
    else:
        a = [item[0] for item in tagged if item[1] == 'PRP$' or item[1] == 'PRP']
    return a

def sentiment_analysis(tweets):
    polarity = 0
    subjectivity = 0
    count_correction = 0
    for i in tweets:
        tagged = pos_tagging(i, 'sentiment')
        testimonial = TextBlob(tagged)
        polarity += testimonial.sentiment.polarity
        subjectivity += testimonial.sentiment.subjectivity
    return ((polarity / len(tweets)), (subjectivity / len(tweets)))

def possession_analysis(tweets):
    for i in tweets:
        tagged = pos_tagging(i, 'possession')
        count = tagged.count('I')
        count += tagged.count('my')
        count += tagged.count('We')
        count += tagged.count('Our')
    return count / len(tweets)

def flesch_reading_ease(tweets):
    score = 0 
    new_string = "\n".join(tweets)
    score = textstat.flesch_reading_ease(new_string)
    return score
