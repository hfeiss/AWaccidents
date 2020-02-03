from sklearn import (
    cluster,
    decomposition, ensemble, manifold, 
    random_projection, preprocessing)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from tokenator import tokenize_and_lemmatize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')

X = df['description']
y = np.array(df['target'])

vectorizer = CountVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=None,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)

vectorizer.fit(X)
def vector(data):
    return vectorizer.transform(data)
features = vectorizer.get_feature_names()

X_vector = vector(X).todense()

ss = preprocessing.StandardScaler()
X_centered = ss.fit_transform(X_vector)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_centered)

def scree_plot(pca, n_components_to_plot=8, title=None):
    """Make a scree plot showing the variance explained (i.e. varaince of the projections) for the principal components in a fit sklearn PCA object.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    pca: sklearn.decomposition.PCA object.
      A fit PCA object.
      
    n_components_to_plot: int
      The number of principal components to display in the skree plot.
      
    title: str
      A title for the skree plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]), 
                   (ind[i]+0.2, vals[i]+0.005), 
                   va="bottom", 
                   ha="center", 
                   fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)

    plt.savefig('/Users/hfeiss/dsi/capstone-2/images/scree.png')


def plot_mnist_embedding(X, y, title):
    """Plot an embedding of the mnist dataset onto a plane.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
      
    y: numpy.array
      The labels of the datapoints.  Should be digits.
      
    title: str
      A title for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 str(digits.target[i]), 
                 color=plt.cm.Set1(y[i] / 10.), 
                 fontdict={'weight': 'bold', 'size': 12})

    ax.set_xticks([]), 
    ax.set_yticks([])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([-0.1,1.1])

    plt.savefig(f'/Users/hfeiss/dsi/capstone-2/images/{title}.png')

    if title is not None:
        ax.set_title(title, fontsize=16)



if __name__ == "__main__":
    scree_plot(pca, title="Scree Plot for Digits Principal Components")
    
    pca = decomposition.PCA(n_components=2)
    X_pca = pca.fit_transform(X_centered)
    plot_mnist_embedding(X_pca, y, 'mnist')  