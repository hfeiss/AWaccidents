import numpy as np
import pandas as pd
import joblib
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from tokenator import tokenize_and_lemmatize


df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')

X = df['description']
# y = np.array(df['target'])
y = np.array(df['F'])

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    shuffle=True,
                                                    random_state=42
                                                    )

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

vectorizer.fit(X_train)


def vector(data):
    return vectorizer.transform(data)


features = vectorizer.get_feature_names()

ada = AdaBoostClassifier(n_estimators=50)

ada.fit(vector(X_train), y_train)
importances = np.mean([tree.feature_importances_ for tree in ada.estimators_],
                      axis=0)

important_idx = importances.argsort()[-1:-16:-1]
important_val = importances[important_idx]
important_wrd = []

for feat in important_idx:
    important_wrd.append(features[feat])


def print_important():
    print(important_wrd)


def plot_important_features():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Top Words for Predicting Fatality', fontsize=16)
    ax.bar(range(len(important_val)),
           important_val,
           # yerr=important_std,
           align='center',
           color='#047495')
    ax.set_xticks(np.array(range(len(important_val))) - 0.15)
    ax.set_xticklabels(important_wrd, rotation=30, fontsize=12)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim([0, np.max(important_val) + 0.025])
    ax.tick_params(axis='both', which='both', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('/Users/hfeiss/dsi/capstone-2/images/ada.png')


def horiz_plot():
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_title('Top Words for Predicting Fatality', fontsize=16)
    y_pos = np.arange(len(important_wrd))
    ax.barh(y_pos,
            important_val,
            # xerr=important_std,
            align='center',
            color='#047495')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(important_wrd)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('/Users/hfeiss/dsi/capstone-2/images/ada.png')


def errors_vs_n():
    test_scores = []
    train_scores = []
    for n in range(1, 111, 10):
        print(n)
        ada = AdaBoostClassifier(n_estimators=n)
        ada.fit(vector(X_train), y_train)
        train = ada.score(vector(X_train), y_train)
        test = ada.score(vector(X_test), y_test)
        train_scores.append(train)
        test_scores.append(test)
    print(test_scores)
    print(train_scores)


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
    # print_important()
    # plot_important_features()
    # horiz_plot()
    # score = ada.score(vector(X_test), y_test)
    # print(f'Saving model with score: {score}')
    # joblib.dump(ada, '/Users/hfeiss/dsi/capstone-2/models/ada.joblib')
    errors_vs_n()
    # categorical(ada)
    # binary(ada)
