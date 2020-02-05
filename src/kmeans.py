import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tokenator import tokenize_and_lemmatize

df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')
vectorizer = TfidfVectorizer(stop_words='english', min_df=3, max_features=4000, max_df=.7)

data = df['description'].apply(tokenize_and_lemmatize).str.join(' ')
X = vectorizer.fit_transform(data)
features = vectorizer.get_feature_names()

def print_top_words():
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-16:-1]
    print("Top features for each cluster with 1000 max features:")
    for num, centroid in enumerate(top_centroids):
        print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))

if __name__ == "__main__":
    K = [k for k in range(1, 11)]
    intertias = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        intertias.append(kmeans.inertia_)
        print(intertias)
    
    # joblib.dump(kmeans, '/Users/hfeiss/dsi/capstone-2/models/kmeans.joblib')
    # kmeans = joblib.load('/Users/hfeiss/dsi/capstone-2/models/kmeans.joblib')

    # print_top_words()