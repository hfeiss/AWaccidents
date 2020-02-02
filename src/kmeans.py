import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')
vectorizer = TfidfVectorizer(stop_words='english', min_df=3, max_features=4000, max_df=.7)
X = vectorizer.fit_transform(df['description'].str.join(sep=' '))
features = vectorizer.get_feature_names()

kmeans = KMeans()
kmeans.fit(X)
top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
print("Top features for each cluster with 1000 max features:")
for num, centroid in enumerate(top_centroids):
    print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))