from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenator import tokenize_and_lemmatize


df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')
vectorizer = TfidfVectorizer(stop_words='english', min_df=3, max_features=4000, max_df=.7)

data = df['description'].apply(tokenize_and_lemmatize).str.join(' ')
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