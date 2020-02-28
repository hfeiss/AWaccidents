import numpy as np
import pandas as pd
from pprint import pprint
from filepaths import Root
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report


paths = Root(__file__, 1).paths()
train = paths.data.train.path
holdout = paths.data.holdout.path
images = paths.images.path

train_df = pd.read_pickle(train + 'train.pkl')
holdout_df = pd.read_pickle(holdout + 'holdout.pkl')

train_X = train_df['description']
holdout_X = holdout_df['description']

train_y_class = np.array(train_df['target'])
holdout_y_class = np.array(holdout_df['target'])

train_y_binary = np.array(train_df['F'])
holdout_y_binary = np.array(holdout_df['F'])


def binary(model, vectorizer):

    X = vectorizer.fit_transform(train_X)

    model.fit(X, train_y_binary)

    predict = model.predict(vectorizer.transform(holdout_X))

    acc = accuracy_score(holdout_y_binary, predict)
    rec = recall_score(holdout_y_binary, predict)
    pre = precision_score(holdout_y_binary, predict)

    print(f'Accuracy:  {acc}')
    print(f'Recall:    {rec}')
    print(f'Precision: {pre}')


def categorical(model, vectorizer):

    X = vectorizer.fit_transform(train_X)

    model.fit(X, train_y_class)

    predict = model.predict(vectorizer.transform(holdout_X))

    labels = ['Medical', 'Injury', 'Fatality']
    predict = model.predict(vectorizer.transform(holdout_X))
    results = classification_report(holdout_y_class,
                                    predict,
                                    target_names=labels)
    pprint(results)
