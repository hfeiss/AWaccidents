import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from model_scorer import binary, categorical, get_features


bayes = MultinomialNB()


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


if __name__ == "__main__":
    binary(bayes)
    features = get_features()
    print_important_words(binary=True)
    print_anti_important_words(binary=True)
    
    categorical(bayes)
    features = get_features()
    print_important_words(binary=False)
    print_anti_important_words(binary=False)