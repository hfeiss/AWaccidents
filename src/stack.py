import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tokenator import tokenize_and_lemmatize
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')
df.dropna(inplace=True)

docs = df['description']
y = df['F'].values

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=None,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

vectorizer.fit(docs)
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

bc.fit(vector(docs), y)

df['bagging_guess'] = bc.predict(vector(docs))
print(df['bagging_guess'][0:10])

X = df[['rellevel', 'age', 'kayak', 'commercial', 'bagging_guess']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

model = Logit(y, X).fit()
print(model.summary())


kfold = KFold(n_splits=10)

accuracies = []
precisions = []
recalls = []

X_train, X_test, y_train, y_test = train_test_split(X, y)

for train_index, test_index in kfold.split(X_train):
    model = LogisticRegression(solver="lbfgs")
    model.fit(X[train_index], y[train_index])
    y_predict = model.predict(X[test_index])
    y_true = y[test_index]
    accuracies.append(accuracy_score(y_true, y_predict))
    precisions.append(precision_score(y_true, y_predict))
    recalls.append(recall_score(y_true, y_predict))

print("Accuracy:", np.average(accuracies))
print("Precision:", np.average(precisions))
print("Recall:", np.average(recalls))


if __name__ == "__main__":
    pass