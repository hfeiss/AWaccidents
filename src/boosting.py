import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier
from model_scorer import binary, categorical


ada = AdaBoostClassifier(n_estimators=50)

if __name__ == "__main__":
    binary(ada)
    categorical(ada)
