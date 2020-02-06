import numpy as np
import pandas as pd
from filepaths import Root
from tokenator import tokenize_and_lemmatize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


paths = Root(1).paths()
clean = paths.data.clean.path

df = pd.read_pickle(clean + '/clean.pkl')

X = df['description']
y = np.array(df['target'])

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    shuffle=True,
                                                    random_state=42
                                                    )

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

vectorizer.fit(X_train)


def vector(data):
    return vectorizer.transform(data)


features = vectorizer.get_feature_names()

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

rf.fit(vector(X_train), y_train)

print(rf.score(vector(X_test), y_test))

importances = rf.feature_importances_
short_list = importances.argsort()[-1:-16:-1]
for feat in short_list:
    print(features[feat])
