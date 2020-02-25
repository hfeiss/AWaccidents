import joblib
import numpy as np
import pandas as pd
from filepaths import Root
from sklearn.cluster import KMeans
from tokenator import tokenize_and_lemmatize
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer


paths = Root(__file__, 1).paths()
clean = paths.data.clean.path

df = pd.read_pickle(clean + '/clean.pkl')

data = df['description']

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

X = vectorizer.fit_transform(data)

features = vectorizer.get_feature_names()


def print_top_words(model):
    top_centroids = model.cluster_centers_.argsort()[:, -1:-21:-1]
    print("Top features for each cluster with 1000 max features:")
    for num, centroid in enumerate(top_centroids):
        print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))


def sil_score(model, n=12):
    K = [k for k in range(2, n + 1)]
    scores = []
    for k in K:
        labels = model(n_clusters=k).fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(scores)


if __name__ == "__main__":

    # model = KMeans
    # sil_score(model, 20)
    model = KMeans(n_clusters=20).fit(X)
    print_top_words(model)

    # joblib.dump(kmeans, '/Users/hfeiss/dsi/capstone-2/models/kmeans.joblib')
    # kmeans = joblib.load('/Users/hfeiss/dsi/capstone-2/models/kmeans.joblib')

