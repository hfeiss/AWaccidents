import numpy as np
import pandas as pd
from filepaths import Root
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenator import tokenize_and_lemmatize


paths = Root(1).paths()
clean = paths.data.clean.path

df = pd.read_pickle(clean + '/clean.pkl')

data = df['description']

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

X = vectorizer.fit_transform(data)

range_n_clusters = [2, 3, 4, 5, 6, 8, 10]

scores = []
for n_clusters in range_n_clusters:

    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    scores.append(silhouette_avg)

print(scores)
