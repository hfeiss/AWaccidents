import numpy as np
import pandas as pd
from pprint import pprint
from filepaths import Root
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tokenator import tokenize_and_lemmatize


paths = Root(0).paths()
clean = paths.data.clean.path

df = pd.read_pickle(clean + '/clean.pkl')
X = df['description']
y = np.array(df['target'])
# y = np.array(df['F'])


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    shuffle=True,
                                                    random_state=42
                                                    )

vectorizer = CountVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

vectorizer.fit(X_train)
features = vectorizer.get_feature_names()


def vector(data):
    return vectorizer.transform(data)


def print_important_words(binary=True):
    if binary:
        labels = ['Not Fatal', 'Fatal']
    else:
        labels = ['Medical', 'Injury', 'Death']
    coefs = bayes.coef_
    for i, outcome in enumerate(coefs):
        coefs_sorted = np.argsort(outcome)[-1:-16:-1]
        print(f'Top words for {labels[i]}:')
        print([features[coef] for coef in coefs_sorted])
        print('\n')


def print_anti_important_words(binary=True):
    if binary:
        labels = ['Not Fatal', 'Fatal']
    else:
        labels = ['Medical', 'Injury', 'Death']
    coefs = bayes.coef_
    for i, outcome in enumerate(coefs):
        coefs_sorted = np.argsort(outcome)[0:16]
        print(f'Top anti-words for {labels[i]}:')
        print([features[coef] for coef in coefs_sorted])
        print('\n')


bayes = MultinomialNB()
bayes.fit(vector(X_train), y_train)


def binary(model):
    model.fit(vector(X_train), y_train)
    predict = model.predict(vector(X_test))
    acc = accuracy_score(y_test, predict)
    rec = recall_score(y_test, predict)
    pre = precision_score(y_test, predict)
    print(f'Accuracy:  {acc}')
    print(f'Recall:    {rec}')
    print(f'Precision: {pre}')


def categorical(model):
    labels = ['Medical', 'Injury', 'Fatality']
    model.fit(vector(X_train), y_train)
    predict = model.predict(vector(X_test))
    results = classification_report(y_test,
                                    predict,
                                    target_names=labels)
    pprint(results)


if __name__ == "__main__":
    # binary(bayes)
    # categorical(bayes)
    print_important_words(binary=False)
    print_anti_important_words(binary=False)