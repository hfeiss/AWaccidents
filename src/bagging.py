from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
from tokenator import tokenize_and_lemmatize


df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')

X = df['description']
y = np.array(df['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=None,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

vectorizer.fit(X_train)
def vector(data):
    return vectorizer.transform(data)
features = vectorizer.get_feature_names()

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

bc.fit(vector(X_train), y_train)

print(bc.score(vector(X_test), y_test))

importances = np.mean([tree.feature_importances_ for tree in bc.estimators_], axis=0)

short_list = importances.argsort()[-1:-42:-1]
for feat in short_list:
    print(features[feat])