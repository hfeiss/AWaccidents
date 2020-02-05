import numpy as np
import pandas as pd
from pprint import pprint
from time import time
from tokenator import tokenize_and_lemmatize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')

X = df['description'].str.join(' ')
y = np.array(df['target'])

pipeline = Pipeline([('vect', CountVectorizer(token_pattern=None,
                                              tokenizer=tokenize_and_lemmatize)),
                     ('bayes', MultinomialNB())])

parameters = {
              'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
              'vect__max_df': (0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
              'vect__min_df': (1, 5, 10, 50),
              'vect__max_features': (1000, 10000, 50000, 100000, None)
              }

grid_search = GridSearchCV(pipeline,
                           parameters,
                           n_jobs=-1,
                           verbose=1)


def run_grid_search(parameters):
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


if __name__ == "__main__":
    run_grid_search(parameters)