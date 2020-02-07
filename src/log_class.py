import numpy as np
import pandas as pd
from filepaths import Root
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


paths = Root(0).paths()
clean = paths.data.clean.path

df = pd.read_pickle(clean + '/clean.pkl')
df = df[df['age'] != 0]
df.dropna(inplace=True)

X = df[['rellevel', 'age', 'kayak']].values
y = df['F'].values

vif = variance_inflation_factor
print('VIF: ')
for i in range(X.shape[1]):
    print(vif(X, i))

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
