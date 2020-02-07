import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model_scorer import *


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


def print_features():
    importances = rf.feature_importances_
    short_list = importances.argsort()[-1:-16:-1]
    for feat in short_list:
        print(features[feat])


if __name__ == "__main__":
    # binary(rf)
    # features = get_features()
    # print_features()

    categorical(rf)
    features = get_features()
    print_features()
