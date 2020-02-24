
import numpy as np
import pandas as pd
from pprint import pprint
from filepaths import Root
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report


paths = Root(__file__, 1).paths()
train = paths.data.train.path
images = paths.images.path

df = pd.read_pickle(train + '/train.pkl')

X = df['description']
y_class = np.array(df['target'])
y_binary = np.array(df['F'])


def split_data(X, y):
    return train_test_split(X, y, shuffle=True, random_state=42)


def binary(model, vectorizer):
   
    X_train, X_test, y_train, y_test = split_data(X, y_binary)
    vectorizer.fit(X_train)
    model.fit(vectorizer.transform(X_train), y_train)
    predict = model.predict(vectorizer.transform(X_test))
    acc = accuracy_score(y_test, predict)
    rec = recall_score(y_test, predict)
    pre = precision_score(y_test, predict)
    print(f'Accuracy:  {acc}')
    print(f'Recall:    {rec}')
    print(f'Precision: {pre}')


def categorical(model, vectorizer):
 
    X_train, X_test, y_train, y_test = split_data(X, y_class)
    vectorizer.fit(X_train)
    labels = ['Medical', 'Injury', 'Fatality']
    model.fit(vectorizer.transform(X_train), y_train)
    predict = model.predict(vectorizer.transform(X_test))
    results = classification_report(y_test,
                                    predict,
                                    target_names=labels)
    pprint(results)
