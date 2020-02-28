import joblib
import numpy as np
import pandas as pd
from filepaths import Root
from sklearn.cluster import KMeans
from tokenator import tokenize_and_lemmatize
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from bayes import print_important_words, print_anti_important_words


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


def group_and_print(df, kmeans, n_clusters):
    for cluster in range(n_clusters):
        print(f'Topic: {cluster}:\n')
        mask = kmeans.labels_ == cluster
        data = df.iloc[list(mask)]
        docs = data['description']
        X = vectorizer.fit_transform(docs)
        y = data['F']
        print(y.value_counts())
        bayes = MultinomialNB().fit(X, y)
        # print_anti_important_words(bayes, features)
        features = vectorizer.get_feature_names()
        print_important_words(bayes, features)


if __name__ == "__main__":

    kmeans = KMeans(n_clusters=18).fit(X)
    print_top_words(kmeans)
    group_and_print(df, kmeans, 18)
