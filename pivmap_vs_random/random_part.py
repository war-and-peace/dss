# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 3.11.2020.
# ------------------------------------------------------------------------------

import numpy as np


class RandomPartitioner:
    def __init__(self, path_to_dataset, embeddings=None):
        if embeddings is None:
            self.dataset_path = path_to_dataset
        else:
            self.embeddings = embeddings.copy()

    def load(self):
        self.embeddings = np.load(self.dataset_path)

    def partition(self, n_partition=4):
        np.random.shuffle(self.embeddings)
        partitions = []
        start = 0
        t, addit = self.embeddings.shape[0] // n_partition, self.embeddings.shape[0] % n_partition

        for _ in range(n_partition):
            end = start + t
            if addit > 0:
                addit -= 1
                end += 1
            partitions.append(self.embeddings[start:end])
            start = end

        return partitions

