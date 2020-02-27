import joblib
import numpy as np
from filepaths import Root
import nlp_holdout_scorer as holdout
from nlp_scorer import binary, categorical
from tokenator import tokenize_and_lemmatize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


paths = Root(__file__, depth=1).paths()
models = paths.models.path

bayes = MultinomialNB()

vectorizer = CountVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)


def print_important_words(bayes, features, binary=True):
    if binary:
        labels = ['Not Fatal', 'Fatal']
    else:
        labels = ['Medical', 'Injury', 'Death']
    for i, outcome in enumerate(bayes.coef_):
        coefs_sorted = np.argsort(outcome)[-1:-16:-1]
        print(coefs_sorted.shape)
        print(f'Top words for {labels[i]}:')
        print([features[coef] for coef in coefs_sorted])
        print('\n')


def print_anti_important_words(bayes, features, binary=True):
    if binary:
        labels = ['Not Fatal', 'Fatal']
    else:
        labels = ['Medical', 'Injury', 'Death']
    for i, outcome in enumerate(bayes.coef_):
        coefs_sorted = np.argsort(outcome)[0:16]
        print(f'Top anti-words for {labels[i]}:')
        print([features[coef] for coef in coefs_sorted])
        print('\n')


if __name__ == "__main__":
    binary(bayes, vectorizer)
    # joblib.dump(bayes, models + 'bayes.joblib')
    # bayes = joblib.load(bayes + 'bayes.joblib')
    features = vectorizer.get_feature_names()
    print_important_words(bayes, features, binary=True)
    print_anti_important_words(bayes, features, binary=True)
    
    categorical(bayes, vectorizer)
    # joblib.dump(bayes, models + 'bayes.joblib')
    # bayes = joblib.load(bayes + 'bayes.joblib')
    features = vectorizer.get_feature_names()
    print_important_words(bayes, features, binary=False)
    print_anti_important_words(bayes, features, binary=False)

    # holdout.binary(bayes, vectorizer)
    # holdout.categorical(bayes, vectorizer)