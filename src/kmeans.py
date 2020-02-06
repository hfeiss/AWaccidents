import numpy as np
import pandas as pd
import joblib
from filepaths import Root
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tokenator import tokenize_and_lemmatize
from sklearn.metrics import silhouette_score

paths = Root(1).paths()
clean = paths.data.clean.path

df = pd.read_pickle(clean + '/clean.pkl')
data = df['description'].apply(tokenize_and_lemmatize).str.join(' ')

vectorizer = TfidfVectorizer(stop_words='english',
                             min_df=3,
                             max_features=4000,
                             max_df=.7)

X = vectorizer.fit_transform(data)
features = vectorizer.get_feature_names()


def print_top_words():
    top_centroids = kmeans.cluster_centers_.argsort()[:, -1:-16:-1]
    print("Top features for each cluster with 1000 max features:")
    for num, centroid in enumerate(top_centroids):
        print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))


if __name__ == "__main__":
    K = [k for k in range(1, 11)]
    scores = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        scores.append(kmeans.score(X))
        print(scores)

    # joblib.dump(kmeans, '/Users/hfeiss/dsi/capstone-2/models/kmeans.joblib')
    # kmeans = joblib.load('/Users/hfeiss/dsi/capstone-2/models/kmeans.joblib')

    # print_top_words()
