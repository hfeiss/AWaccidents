import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tokenator import tokenize_and_lemmatize
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib


df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')
df.dropna(how='any', inplace=True)

X = df[['rellevel', 'age', 'kayak', 'commercial']].to_numpy()
docs = df['description'].to_numpy()
y = df['F'].to_numpy()

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=None,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

def vector(data):
    return vectorizer.transform(data)

bc = BaggingClassifier(base_estimator=None,
                       n_estimators=10,
                       max_samples=1.0,
                       max_features=1.0,
                       bootstrap=True,
                       bootstrap_features=False,
                       oob_score=False,
                       warm_start=False,
                       n_jobs=-1,
                       random_state=42,
                       verbose=1)

def sm_summary(X, docs, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)
    vectorizer.fit(docs)
    bc.fit(vector(docs), y)
    # bc_predict = bc.predict_proba(vector(docs))[:, 1:]
    bc_predict = np.reshape(bc.predict(vector(docs)), (-1, 1))
    X = np.append(X, bc_predict, axis=1)
    # X = ss.fit_transform(X)
    X = add_constant(X)

    model = Logit(y, X).fit()
    print(model.summary())


def k_fold():
    kfold = KFold(n_splits=10)

    accuracies = []
    precisions = []
    recalls = []


    for train_index, test_index in kfold.split(X):
        train_docs =  docs[train_index]
        test_docs = docs[test_index]
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        vectorizer.fit(train_docs)
        bc.fit(vector(train_docs), y_train)
        bc_predict = bc.predict(vector(train_docs))
        X_train = np.append(X_train, np.reshape(bc_predict, (-1, 1)), axis=1)

        model = LogisticRegression(solver="lbfgs")
        model.fit(X_train, y_train)

        bc_predict = bc.predict(vector(test_docs))
        X_test = np.append(X_test, np.reshape(bc_predict, (-1, 1)), axis=1)
        y_predict = model.predict(X_test)
        y_true = y_test

        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

    print("Accuracy:", np.average(accuracies))
    print("Precision:", np.average(precisions))
    print("Recall:", np.average(recalls))


if __name__ == "__main__":
    sm_summary(X, docs, y)