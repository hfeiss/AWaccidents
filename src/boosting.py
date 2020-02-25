import joblib
import numpy as np
import pandas as pd
from filepaths import Root
import nlp_holdout_scorer as holdout
from nlp_scorer import binary, categorical
from tokenator import tokenize_and_lemmatize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

ada = AdaBoostClassifier(n_estimators=50)

vectorizer = CountVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)


if __name__ == "__main__":

    binary(ada, vectorizer)
    # joblib.dump(ada, models + 'ada.joblib')
    # ada = joblib.load(models + 'ada.joblib')

    categorical(ada, vectorizer)
    # joblib.dump(ada, models + 'ada.joblib')
    # ada = joblib.load(models + 'ada.joblib')
    
    holdout.binary(ada, vectorizer)
    holdout.categorical(ada, vectorizer)