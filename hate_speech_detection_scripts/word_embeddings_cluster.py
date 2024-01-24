import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from time import time
import multiprocessing
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

class WordEmbeddingClusterer:
    def __init__(self, data_path, min_count=3, window=6, sample=2e-5, alpha=1.1, min_alpha=0.0078, negative=9, epochs=30, num_clusters=5):
        self.data_path = data_path
        self.min_count = min_count
        self.window = window
        self.sample = sample
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative
        self.epochs = epochs
        self.num_clusters = num_clusters
        self.sentences = None
        self.model = None

    def preprocess_data(self):
        df = pd.read_csv(self.data_path, sep=',', encoding='utf-8-sig', float_precision='round_trip')
        hate_data = df[df['model_result'] == '__label____label__HATE']

        hate_data['lemmatized'] = hate_data['lemmatized'].astype(str)

        hate_data['edited'] = hate_data['lemmatized'].apply(lambda row: " ,".join(re.sub('\W+', '', word) for word in row.split()))

        self.sentences = hate_data['edited'].apply(lambda row: row.split()).tolist()

    def build_word2vec_model(self):
        cores = multiprocessing.cpu_count()
        self.model = Word2Vec(min_count=self.min_count, window=self.window, sample=self.sample, alpha=self.alpha,
                              min_alpha=self.min_alpha, negative=self.negative, workers=cores - 1)
        self.model.build_vocab(self.sentences, progress_per=100)
        t = time()
        self.model.train(self.sentences, total_examples=self.model.corpus_count, epochs=self.epochs, report_delay=1)
        print('Time to train the Word2Vec model: {} mins'.format(round((time() - t) / 60, 2)))

    def cluster_words(self):
        X = self.model.wv.vectors
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(X)

        labels = kmeans.labels_
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

        print("Cluster id labels for inputted data:")
        print(labels)
        print("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
        print(kmeans.score(X))
        print("Silhouette_score:")
        print(silhouette_score)

        for i, word in enumerate(self.model.wv.index_to_key):
            print(word + ":" + str(labels[i]))

        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
        plt.show()

    def find_clusters(self):
        X = self.model.wv.vectors
        centers, labels = self._find_clusters(X, self.num_clusters)

        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
        plt.show()

    @staticmethod
    def _find_clusters(X, n_clusters, rseed=4):
        rng = np.random.RandomState(rseed)
        i = rng.permutation(X.shape[0])[:n_clusters]
        centers = X[i]

        while True:
            labels = pairwise_distances_argmin(X, centers)
            new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

            if np.all(centers == new_centers):
                break
            centers = new_centers

        return centers, labels

if __name__ == "__main__":
    data_path = './lemmatized_dataframe.csv'
    embedding_clusterer = WordEmbeddingClusterer(data_path)
    embedding_clusterer.preprocess_data()
    embedding_clusterer.build_word2vec_model()
    embedding_clusterer.cluster_words()
    embedding_clusterer.find_clusters()
