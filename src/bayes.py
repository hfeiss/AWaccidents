import numpy as np
import pandas as pd
from pprint import pprint
from filepaths import Root
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tokenator import tokenize_and_lemmatize


paths = Root(0).paths()
clean = paths.data.clean.path

df = pd.read_pickle(clean + '/clean.pkl')
X = df['description']
y = np.array(df['target'])

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


def print_important_words():
    labels = ['Medical', 'Injury', 'Death']
    coefs = bayes.coef_
    for i, outcome in enumerate(coefs):
        coefs_sorted = np.argsort(outcome)[-1:-16:-1]
        print(f'Top words for {labels[i]}:')
        print([features[coef] for coef in coefs_sorted])
        print('\n')


def print_anti_important_words():
    labels = ['Medical', 'Injury', 'Death']
    coefs = bayes.coef_
    for i, outcome in enumerate(coefs):
        coefs_sorted = np.argsort(outcome)[0:16]
        print(f'Top anti-words for {labels[i]}:')
        print([features[coef] for coef in coefs_sorted])
        print('\n')


bayes = MultinomialNB()
bayes.fit(vector(X_train), y_train)


if __name__ == "__main__":
    print_important_words()
    print_anti_important_words()
    labels = ['Medical', 'Injury', 'Fatality']
    predict = bayes.predict(vector(X_test))
    results = classification_report(y_test,
                                    predict,
                                    target_names=labels)
    pprint(results)