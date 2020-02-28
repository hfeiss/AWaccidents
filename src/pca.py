import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from filepaths import Root
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenator import tokenize_and_lemmatize


paths = Root(__file__, 1).paths()
clean = paths.data.clean.path
models = paths.models.path

df = pd.read_pickle(clean + 'clean.pkl')

X = df['description']
y = np.array(df['target'])

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.55,
                             max_features=100000,
                             token_pattern=None,
                             tokenizer=tokenize_and_lemmatize)


def center_X():

    X_vector = vectorizer.fit_transform(X).todense()

    ss = preprocessing.StandardScaler()

    return ss.fit_transform(X_vector)


def scree_plot(pca, X_pca, n_components_to_plot=8, title=None):

    fig, ax = plt.subplots(figsize=(10, 6))

    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='#047495')
    ax.scatter(ind, vals, color='#047495', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]),
                    (ind[i]+0.2, vals[i]+0.005),
                    va="bottom",
                    ha="center",
                    fontsize=12)

    ax.set_xticklabels(ind, fontsize=14)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=16)

    plt.savefig('/Users/hfeiss/dsi/capstone-2/images/scree.png')


def plot_pca_target(X_pca, y, title):

    labels = ['M', 'I', 'F']
    colors = ['#047495', '#d2bd0a', '#f10c45']

    fig, ax = plt.subplots(figsize=(10, 6))

    x_min, x_max = np.min(X_pca, 0), np.max(X_pca, 0)
    X = (X_pca - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1],
                 str(labels[y[i]]),
                 color=colors[y[i]],
                 fontdict={'weight': 'bold', 'size': 12,
                           'alpha': 0.5})

    ax.scatter(1, 1, c='#f10c45', label='Fatality', alpha=.6)
    ax.scatter(1, 1, c='#d2bd0a', label='Injury', alpha=.6)
    ax.scatter(1, 1, c='#047495', label='Medical', alpha=.6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([0.025, 0.1])
    ax.set_xlim([0.0003, 0.0013])
    ax.legend(frameon=False)

    plt.savefig(f'/Users/hfeiss/dsi/capstone-2/images/{title}.png')

    if title is not None:
        ax.set_title(title, fontsize=16)


if __name__ == "__main__":

    X_centered = center_X()

    pca = PCA(n_components=10)
    joblib.dump(pca.fit(X_centered), models + '/pca_10.joblib')
    pca = joblib.load(models + '/pca_10.joblib')
    joblib.dump(pca.transform(X_centered), models + '/X_pca_10.joblib')
    X_pca = joblib.load(models + '/X_pca_10.joblib')
    scree_plot(pca,
               X_pca,
               title="Scree Plot for Description Principal Components")

    pca = PCA(n_components=2)
    pca.fit(X_centered)
    joblib.dump(pca, models + '/pca_2.joblib')
    pca = joblib.load(models + '/pca_10.joblib')
    joblib.dump(pca.transform(X_centered), models + '/X_pca_2.joblib')

    X_pca = joblib.load(models + '/X_pca_2.joblib')
    plot_pca_target(X_pca, y, 'pca_targets_idf')
