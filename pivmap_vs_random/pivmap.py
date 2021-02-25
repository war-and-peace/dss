# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 3.11.2020.
# ------------------------------------------------------------------------------

import numpy as np
import random


class PivotMapping:
    def __init__(self, path_to_dataset, embeddings=None):
        if embeddings is None:
            self.dataset_path = path_to_dataset
        else:
            self.embeddings = embeddings.copy()

        self.ids = []
        self.vectors = []

    def load(self):
        f = open(self.dataset_path, 'rb')
        self.embeddings = np.load(f)

    def calc_distances(self, p0, points):
        return np.sqrt(((p0 - points) ** 2).sum(axis=1))

    def graipher(self, pts, K):
        farthest_pts = np.zeros((K, pts.shape[1]))
        farthest_pts[0] = pts[np.random.randint(pts.shape[0])]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, K):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

    def transform(self, vec, pivots):
        return np.array([np.linalg.norm(vec - pivot) for pivot in pivots])

    def _partition(self, vectors, ids, n):

        if n == 1:
            return [vectors], [ids]

        a, b = (n + 1) // 2, n // 2
        dim = random.randint(0, self.vectors.shape[1] - 1)
        for i in range(len(ids) - 1):
            for j in range(i + 1, len(ids)):
                if vectors[i][dim] < vectors[j][dim]:
                    temp = vectors[i]
                    vectors[i] = vectors[j]
                    vectors[j] = temp
                    temp2 = ids[i]
                    ids[i] = ids[j]
                    ids[j] = temp2

        left = (len(vectors) // n) * a
        avec, aid = self._partition(vectors[:left], ids[:left], a)
        bvec, bid = self._partition(vectors[left:], ids[left:], b)
        vecs, iden = [], []
        vecs.extend(avec)
        vecs.extend(bvec)
        iden.extend(aid)
        iden.extend(bid)
        return vecs, iden

    def partition(self, n_partition=4, n_pivots=20):
        pivots = self.graipher(self.embeddings, n_pivots)
        self.vectors = np.array([self.transform(vec, pivots) for vec in self.embeddings])
        self.ids = np.array([i for i in range(self.vectors.shape[0])])
        vecs, ids = self._partition(self.vectors.copy(), self.ids.copy(), n_partition)
        partitions = []
        for id_list in ids:
            partitions.append(np.array([self.embeddings[id] for id in id_list]))

        return partitions
