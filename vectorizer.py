# vectorizer.py

import os
import pickle
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load stopwords and porter stemmer
stop = pickle.load(open(os.path.join('pkl_objects', 'stopwords.pkl'), 'rb'))
porter = pickle.load(open(os.path.join('pkl_objects', 'porter.pkl'), 'rb'))

def preprocess_text(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def tokenizer(text):
    return [porter.stem(word) for word in text.split() if word not in stop]

def get_vectorizer():
    df = pd.read_csv('movie_data.csv')
    X = df['review'].apply(preprocess_text)

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None,
                            tokenizer=tokenizer,
                            ngram_range=(1,1),
                            stop_words=None)
    
    tfidf.fit(X)
    return tfidf
