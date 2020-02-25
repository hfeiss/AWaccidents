import numpy as np
import pandas as pd
import joblib
from filepaths import Root
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tokenator import tokenize_and_lemmatize


paths = Root(__file__, 1).paths()
clean = paths.data.clean.path
models = paths.models.path

df = pd.read_pickle(clean + 'clean.pkl')
X = df['description']

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

num_topics = 20
nmf = NMF(n_components=num_topics,
          init=None,
          solver='cd',
          beta_loss='frobenius',
          tol=0.0001,
          max_iter=200,
          random_state=42,
          alpha=0.0,
          l1_ratio=0.0,
          verbose=0,
          shuffle=False)


if __name__ == "__main__":

    vectorizer.fit(X)
    features = vectorizer.get_feature_names()
    
    probs = nmf.fit_transform(vectorizer.transform(X))
    # joblib.dump(probs, '/Users/hfeiss/dsi/capstone-2/models/nmf.joblib')
    # probs = joblib.load('/Users/hfeiss/dsi/capstone-2/models/nmf.joblib')
    probs = np.array(probs)

    features = np.array(vectorizer.get_feature_names())
    sorted_topics = nmf.components_.argsort(axis=1)[:, ::-1][:, :20]

    top_doc_idx = probs.argsort(axis=1)[-1:-201:-1, :]
    joblib.dump(top_doc_idx, models + 'doc_idx.joblib')

    print(top_doc_idx.shape)
    for i, topic in enumerate(sorted_topics):
        print(f'Topic: {i} with closest article {top_doc_idx[i]}')
        print(features[topic])

    for i in range(num_topics):
        print(f'Topic {i} top docs')
        # print(X[doc])
        print(top_doc_idx[:, i])
