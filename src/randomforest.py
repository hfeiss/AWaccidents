import numpy as np
from sklearn.ensemble import RandomForestClassifier
from nlp_scorer import binary, categorical
import nlp_holdout_scorer as holdout
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tokenator import tokenize_and_lemmatize
import joblib
from filepaths import Root


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

vectorizer = CountVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)


def print_features(n=15):

    features = vectorizer.get_feature_names()
    importances = rf.feature_importances_
    short_list = importances.argsort()[-1:-(n + 1):-1]
    for feat in short_list:
        print(features[feat])


if __name__ == "__main__":

    # rf = joblib.load(models + 'rf.joblib')
    binary(rf, vectorizer)
    # joblib.dump(rf, models + 'rf.joblib')

    # rf = joblib.load(models + 'rf.joblib')
    categorical(rf, vectorizer)
    # joblib.dump(rf, models + 'rf.joblib')
    
    holdout.binary(rf, vectorizer)
    holdout.categorical(rf, vectorizer)

    print_features()