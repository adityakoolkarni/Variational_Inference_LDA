import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

def get_top_ten():
    num_topics = 5
    words = list(set(reuters.words(reuters.fileids()[:10])))
    total_words = len(words)
    word_lookup = {i:word for i,word in enumerate(words)}
    topic_probability = np.random.dirichlet(np.ones(total_words),(num_topics))
    for topic in range(num_topics):
        for k,v in word_lookup.items():
            if k in sorted(range(len(topic_probability)), key=lambda i: topic[i])[-10:]:
                print(topic, v)

if __name__ == '__main__':
    get_top_ten()

