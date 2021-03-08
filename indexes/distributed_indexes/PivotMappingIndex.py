# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 3.3.2021.
# ------------------------------------------------------------------------------

from indexes.single_indexes.hnsw_hnswlib import HnswHnswlib
from indexes.testers.TimeStats import QueryTimeStats, BuildTimeStats
from indexes.utils.dataset import BasicDataset
import Index

from sklearn.cluster import KMeans
from overrides import overrides
from typing import List, Tuple
from time import time
import numpy as np
import random


def is_in_mbb(query, mbb) -> bool:
    for i in range(query.shape[0]):
        if not (mbb[0][i] <= query[i] <= mbb[1][i]):
            return False
    return True


class PivotMappingIndex(Index.Index):
    def __init__(self, name, max_elements, dimensions, metric='l2', distance_function=None, num_threads=-1,
                 num_partitions=4, ef_construction=1000, ef=1000, m=64, num_pivots: int = 4):
        super().__init__(name, max_elements, dimensions, metric, distance_function, num_threads)
        self.num_partitions = num_partitions
        self.ef_construction = ef_construction
        self.ef = ef
        self.m = m
        self.num_pivots = num_pivots
        self.single_partitions_size = 0
        self.mapping = None
        self.pivots = None
        self.vectors = None
        self.ids = None
        self.MBBs = None

    def partition(self, dataset):
        data = dataset.get_data()

        self.pivots = self.get_farthest_pts(data, self.num_pivots)
        vectors = np.array([self.transform(vec, self.pivots) for vec in data])

        ids = np.array([i for i in range(vectors.shape[0])])
        v_ids = list(zip(vectors, ids))

        vectors_with_id = self._partition(v_ids, self.num_partitions)

        self.vectors = np.array([[x[0] for x in vector] for vector in vectors_with_id])
        self.ids = np.array([[x[1] for x in vector] for vector in vectors_with_id])

        partitions = [data[id_list] for id_list in self.ids]

        self.MBBs = [(np.min(vec, axis=0), np.max(vec, axis=0)) for vec in self.vectors]

        return partitions

    @overrides
    def build(self, dataset):
        partition_start_time = time()
        partitions = self.partition(dataset)
        elapsed_partition_time = time() - partition_start_time

        build_start_time = time()
        datasets = [BasicDataset(f'partition{idx}', '') for idx in range(self.num_partitions)]
        for idx, dataset in enumerate(datasets):
            dataset.load_dataset_from_numpy(partitions[idx])
        self.index = [HnswHnswlib('hnsw_hnswlib', datasets[idx].get_size(), datasets[idx].get_dimensions(), self.metric,
                                  self.distance_function, self.num_threads, self.ef_construction, self.ef, self.m) for
                      idx in range(self.num_partitions)]
        for idx, index in enumerate(self.index):
            index.build(datasets[idx])
        elapsed_build_time = time() - build_start_time
        return BuildTimeStats(elapsed_partition_time, elapsed_build_time)

    @overrides
    def search(self, query, k=5) -> Tuple[List[List[Tuple[float, int]]], QueryTimeStats, float]:
        query_start_time = time()
        queries = np.array([self.transform(q, self.pivots) for q in query])
        indices = [self.get_index_ids_by_mbb(query) for query in queries]
        results = [[self.index[idx].search([query[q_id]], k)[0] for idx in id_list] for q_id, id_list in enumerate(indices)]
        elapsed_query_time = time() - query_start_time

        avg_ind = sum([sum(t)for t in indices]) / len(query)
        merge_start_time = time()
        # Fix indexes

        results = [
            [[(t[0], self.ids[indices[idx][ind_id]][t[1]]) for t in index[0]] for ind_id, index in enumerate(res)] for
            idx, res in enumerate(results)]
        result = [list(sorted([t for index in res for t in index], key=lambda x: x[0]))[:k] for res in results]

        # print(results)
        # results_pair = [[result[res_id] for result in results] for res_id in range(len(query))]
        #
        # result = [list(sorted([elem for pair in rp for elem in pair], key=lambda x: x[0]))[:k] for rp in results_pair]
        elapsed_merge_time = time() - merge_start_time
        return result, QueryTimeStats(elapsed_query_time, elapsed_merge_time, len(query)), avg_ind

    def get_index_ids_by_mbb(self, query):
        return [idx for idx, mbb in enumerate(self.MBBs) if is_in_mbb(query, mbb)]

    @staticmethod
    def calc_distances(p0, points):
        return np.sqrt(((p0 - points) ** 2).sum(axis=1))

    def get_farthest_pts(self, pts, num_pivots: int = 8):
        farthest_pts = np.zeros((num_pivots, pts.shape[1]))
        farthest_pts[0] = pts[np.random.randint(pts.shape[0])]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, num_pivots):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

    @staticmethod
    def transform(vec, pivots):
        return np.array([np.linalg.norm(vec - pivot) for pivot in pivots])

    def _partition(self, vectors, num_partitions):

        if num_partitions == 1:
            return [vectors]

        part1, part2 = (num_partitions + 1) // 2, num_partitions // 2
        dim = random.randint(0, vectors[0][0].shape[0] - 1)

        vectors.sort(key=lambda x: x[0][dim])

        left = (len(vectors) // num_partitions) * part1
        vec1 = self._partition(vectors[:left], part1)
        vec2 = self._partition(vectors[left:], part2)

        vectors = []
        vectors.extend(vec1)
        vectors.extend(vec2)

        return vectors
