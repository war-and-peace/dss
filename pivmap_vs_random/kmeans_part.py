# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 16.11.2020.
# ------------------------------------------------------------------------------

import numpy as np
from sklearn.cluster import KMeans


class KMeansPartitioner:
    def __init__(self, path_to_dataset, embeddings=None):
        if embeddings is None:
            self.dataset_path = path_to_dataset
        else:
            self.embeddings = embeddings.copy()

    def load(self):
        self.embeddings = np.load(self.dataset_path)

    def partition(self, n_partition=4):
        kmeans = KMeans(n_clusters=n_partition, random_state=0)
        kmeans.fit(self.embeddings)
        labels = kmeans.labels_
        partitions = [[] for _ in range(n_partition)]
        for i in range(labels.shape[0]):
            partitions[labels[i]].append(self.embeddings[i])

        res = []
        for part in partitions:
            res.append(np.array(part))

        print('Sizes of the partitions: ')
        for part in res:
            print(part.shape, end=' ')
        print()

        return res
