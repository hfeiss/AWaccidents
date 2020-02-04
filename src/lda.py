import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tokenator import tokenize_and_lemmatize


df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')

X = df['description']
# y = np.array(df['target'])

vectorizer = CountVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=None,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

vectorizer.fit(X)
def vector(data):
    return vectorizer.transform(data)
features = vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_components=10,
                                doc_topic_prior=None,
                                topic_word_prior=None,
                                learning_method='batch',
                                learning_decay=0.7,
                                learning_offset=10.0,
                                max_iter=10,
                                batch_size=128,
                                evaluate_every=-1,
                                total_samples=1000000.0,
                                perp_tol=0.1,
                                mean_change_tol=0.001,
                                max_doc_update_iter=100,
                                n_jobs=-1,
                                verbose=1,
                                random_state=42)

probs = lda.fit_transform(vector(X))
probs = np.array(probs)

features = np.array(vectorizer.get_feature_names())

sorted_topics = lda.components_.argsort(axis=1)[:, ::-1][:, :10]

for i, topic in enumerate(sorted_topics):
    print(f'Topic: {i}')
    print(features[topic])


top_doc_idx = probs.argsort(axis=0)[-1:-11:-1, :]

for i, doc in enumerate(top_doc_idx):
    print(f'Topic {i} top doc')
    print(X[doc])