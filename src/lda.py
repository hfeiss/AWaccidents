import numpy as np
import pandas as pd
import joblib
from filepaths import Root
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tokenator import tokenize_and_lemmatize


paths = Root(1).paths()
clean = paths.data.clean.path

df = pd.read_pickle(clean + '/clean.pkl')
X = df['description']

vectorizer = CountVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)


lda = LatentDirichletAllocation(n_components=20,
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


if __name__ == "__main__":

    vectorizer.fit(X)
    features = vectorizer.get_feature_names()

    probs = lda.fit_transform(vectorizer.transform(X))
    joblib.dump(probs, '/Users/hfeiss/dsi/capstone-2/models/lda.joblib')
    probs = joblib.load('/Users/hfeiss/dsi/capstone-2/models/lda.joblib')
    probs = np.array(probs)

    features = np.array(vectorizer.get_feature_names())
    sorted_topics = lda.components_.argsort(axis=1)[:, ::-1][:, :10]

    top_doc_idx = probs.argsort(axis=0)[-1, :]

    for i, topic in enumerate(sorted_topics):
        print(f'Topic: {i} with best article {top_doc_idx[i]}')
        print(features[topic])

    # for i, doc in enumerate(top_doc_idx):
        # print(f'Topic {i} top doc')
        # print(X[doc])
