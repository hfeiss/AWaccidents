import joblib
from filepaths import Root
import nlp_holdout_scorer as holdout
from nlp_scorer import binary, categorical
from tokenator import tokenize_and_lemmatize
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

paths = Root(__file__, depth=1).paths()
models = paths.models.path

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

vectorizer = CountVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

if __name__ == "__main__":

    binary(bc, vectorizer)
    # joblib.dump(bc, models + 'bagging.joblib')
    # bc = joblib.load(models + 'bagging.joblib')

    categorical(bc, vectorizer)
    # joblib.dump(bc, models + 'bagging.joblib')
    # bc = joblib.load(models + 'bagging.joblib')

    holdout.binary(bc, vectorizer)
    holdout.categorical(bc, vectorizer)
