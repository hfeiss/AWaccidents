from sklearn.ensemble import BaggingClassifier
from model_scorer import binary, categorical


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


if __name__ == "__main__":
    binary(bc)
    categorical(bc)
