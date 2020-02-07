
import numpy as np
import pandas as pd
from pprint import pprint
from filepaths import paths
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from tokenator import tokenize_and_lemmatize


paths = paths(0)
clean = paths.data.clean.path
images = paths.images.path

df = pd.read_pickle(clean + '/clean.pkl')

X = df['description']
y_class = np.array(df['target'])
y_binary = np.array(df['F'])

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)


def split_data(X, y):
    return train_test_split(X, y, shuffle=True, random_state=42)


def vector(data):
    return vectorizer.transform(data)


def binary(model):
    X_test, X_train, y_test, y_train = split_data(X, y_binary)
    vectorizer.fit(X_train)
    model.fit(vector(X_train), y_train)
    predict = model.predict(vector(X_test))
    acc = accuracy_score(y_test, predict)
    rec = recall_score(y_test, predict)
    pre = precision_score(y_test, predict)
    print(f'Accuracy:  {acc}')
    print(f'Recall:    {rec}')
    print(f'Precision: {pre}')


def categorical(model):
    X_test, X_train, y_test, y_train = split_data(X, y_class)
    vectorizer.fit(X_train)
    labels = ['Medical', 'Injury', 'Fatality']
    model.fit(vector(X_train), y_train)
    predict = model.predict(vector(X_test))
    results = classification_report(y_test,
                                    predict,
                                    target_names=labels)
    pprint(results)


def get_features():
    return vectorizer.get_feature_names()
