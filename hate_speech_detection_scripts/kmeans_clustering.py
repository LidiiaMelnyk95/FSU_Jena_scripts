from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict
import multiprocessing
import logging

class WordEmbeddingsClusterer:
    def __init__(self, data_path, min_count=2, window=10, sample=6e-5, alpha=0.07, min_alpha=0.0007, negative=15):
        self.data_path = data_path
        self.min_count = min_count
        self.window = window
        self.sample = sample
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative
        self.sentences = None
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.data_path, sep=',', encoding='utf-8-sig', float_precision='round_trip')
        return df[df['model_result'] == '__label____label__HATE']

    def preprocess_data(self, row):
        sent = []
        for word in row['lemmatized'].split(' '):
            words = re.findall('\w+', word)
            words = " ".join(words)
            sent.append(words)
        row['edited'] = " ,".join(sent)
        return row

    def build_word_embeddings_model(self):
        self.sentences = [row.split() for row in self.load_data()['edited'].astype(str)]
        self.model = Word2Vec(min_count=self.min_count,
                              window=self.window,
                              sample=self.sample,
                              alpha=self.alpha,
                              min_alpha=self.min_alpha,
                              negative=self.negative,
                              workers=multiprocessing.cpu_count() - 1)
        t = time()
        self.model.build_vocab(self.sentences, progress_per=10000)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        self.model.train(self.sentences, total_examples=self.model.corpus_count, epochs=30, report_delay=1)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    def vectorize_sentences(self):
        vectors = []
        for sentence in self.sentences:
            vectors.append(self.sent_vectorizer(sentence))
        return vectors

    def sent_vectorizer(self, sent):
        sent_vec = []
        numw = 0
        for w in sent:
            try:
                if numw == 0:
                    sent_vec = self.model[w]
                else:
                    sent_vec = np.add(sent_vec, self.model[w])
                numw += 1
            except:
                pass
        return np.asarray(sent_vec) / numw

    def cluster_sentences(self, num_clusters=5):
        kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance, repeats=10,
                                     avoid_empty_clusters=True)
        assigned_clusters = kclusterer.cluster(self.vectorize_sentences(), assign_clusters=True)
        return assigned_clusters

    def print_clusters(self, assigned_clusters):
        for index, sentence in enumerate(self.sentences):
            print(str(assigned_clusters[index]) + ":" + str(sentence))


if __name__ == "__main__":
    clusterer = WordEmbeddingsClusterer('/Users/lidiiamelnyk/Documents/korrespondent/lemmatized_dataframe_ua.csv')
    clusterer.load_data().apply(clusterer.preprocess_data, axis=1)
    clusterer.build_word_embeddings_model()
    assigned_clusters = clusterer.cluster_sentences()
    clusterer.print_clusters(assigned_clusters)
