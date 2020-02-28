import numpy as np
import pandas as pd
from filepaths import Root
from statsmodels.tools import add_constant
from sklearn.ensemble import RandomForestClassifier
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


paths = Root(__file__, depth=1).paths()
train = paths.data.train.path
holdout = paths.data.holdout.path

train_df = pd.read_pickle(train + 'train.pkl')
train_df = train_df[train_df['age'] != 0]

holdout_df = pd.read_pickle(holdout + 'holdout.pkl')
holdout_df = holdout_df[holdout_df['age'] != 0]

ss = StandardScaler()

columns = ['rellevel', 'difficulty', 'experience', 'F']
names = ['const', 'rellevel', 'difficulty', 'experience']

rf = RandomForestClassifier(n_estimators=1000,
                            criterion='gini',
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features='auto',
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None,
                            bootstrap=True,
                            oob_score=True,
                            n_jobs=-1,
                            random_state=None,
                            verbose=1,
                            warm_start=False,
                            class_weight=None,
                            ccp_alpha=0.0,
                            max_samples=None)


def get_X_y(df):

    df = df[columns]
    df.dropna(inplace=True)

    X = df[columns[:-1]].values
    X = ss.fit_transform(X)
    y = df['F'].values

    return X, y


def score(df):

    X, y = get_X_y(df)

    vif = variance_inflation_factor
    print('VIF: ')
    for i in range(X.shape[1]):
        print(vif(X, i))

    model = rf.fit(X, y)

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold.split(X):
        model.fit(X[train_index], y[train_index])
        y_predict = model.predict(X[test_index])
        y_true = y[test_index]
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

    print("Accuracy:", np.average(accuracies))
    print("Precision:", np.average(precisions))
    print("Recall:", np.average(recalls))


def score_holdout(train_df, holdout_df):

    train_X, train_y = get_X_y(train_df)
    holdout_X, holdout_y = get_X_y(holdout_df)

    model = rf
    model.fit(train_X, train_y)
    predict = model.predict(holdout_X)

    acc = accuracy_score(holdout_y, predict)
    rec = recall_score(holdout_y, predict)
    pre = precision_score(holdout_y, predict)

    print(f'Accuracy:  {acc}')
    print(f'Recall:    {rec}')
    print(f'Precision: {pre}')


if __name__ == "__main__":

    # score(train_df)
    score_holdout(train_df, holdout_df)
