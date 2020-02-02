import numpy as np
import pandas as pd
from pprint import pprint
from time import time
# from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
# from nltk.stem.snowball import SnowballStemmer
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')

X = df['description'].str.join(' ')
# features = vectorizer.get_feature_names()
# X = X.todense()

y = np.array(df['target'])


pipeline = Pipeline([('vect', CountVectorizer()),
                     ('bayes', MultinomialNB())])


parameters = {'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
              'vect__max_df': (0.5, 0.6, 0.7, 0.8, 0.9),
              'vect__min_df': (1, 5, 50),
              'vect__max_features': (None, 1000, 10000, 100000)}
'''
vectorizer = CountVectorizer(ngram_range=(1, 3),
                             analyzer='word',
                             max_df=0.65,
                             min_df=10,
                             max_features=None,
                             vocabulary=None,
                             binary=False)
'''

grid_search = GridSearchCV(pipeline,
                           parameters,
                           n_jobs=-1,
                           verbose=1)


print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print()
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


def print_important_words():
    coefs = bayes.coef_
    print(coefs.shape)
    for outcome in coefs:
        coefs_sorted = np.argsort(outcome)[-1:-21:-1]
        print(coefs_sorted)
        print([features[coef] for coef in coefs_sorted])

# print(bayes.predict(X_test))

# print(bayes.score(X_test, y_test))