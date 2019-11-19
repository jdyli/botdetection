import numpy as np 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import re

def removal_words(sentence):
    new = ' '.join(re.sub("(\\\\[u,U][0-9a-zA-Z]+)", "", sentence).split())
    new = ' '.join(re.sub('http*\S+','',new).split())
    new = ' '.join(re.sub('RT*@\S+','', new).split())
    return new 


def preprocess_text(sentence):
    stop_words = set(stopwords.words('english'))
    sentence = word_tokenize(sentence)
    sentence = [w for w in sentence if not w in stop_words]
    return " ".join(sentence)

def preprocess_domain(tweets):
    preprocessed_list = []
    for i in tweets:
        #preprocessed_list.append(preprocess_text(i))
        preprocessed_list.append(removal_words(i))
    return preprocessed_list