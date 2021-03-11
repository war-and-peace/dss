# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 26.2.2021.
# ------------------------------------------------------------------------------

from indexes.single_indexes.hnsw_hnswlib import HnswHnswlib
from indexes.testers.TimeStats import BuildTimeStats, QueryTimeStats
from indexes.utils.dataset import BasicDataset
from indexes.utils.distance_function import l2distance
import Index

from sklearn.cluster import KMeans
from overrides import overrides
from typing import List, Tuple

from time import time


class KMeansIndex(Index.Index):
    def __init__(self, name, max_elements, dimensions, metric='l2', distance_function=None, num_threads=-1,
                 num_partitions=4, num_partitions_to_search=2, ef_construction=1000, ef=1000, m=64):
        super().__init__(name, max_elements, dimensions, metric, distance_function, num_threads)
        self.num_partitions = num_partitions
        self.num_partitions_to_search = num_partitions_to_search
        self.ef_construction = ef_construction
        self.ef = ef
        self.m = m
        self.single_partitions_size = 0
        self.mapping = None
        self.centers = None

    def partition(self, dataset):
        data = dataset.get_data()
        k_means = KMeans(n_clusters=self.num_partitions, random_state=0)
        k_means.fit(data)
        labels = k_means.labels_
        self.centers = k_means.cluster_centers_
        self.mapping = dict([(i, []) for i in range(self.num_partitions)])
        p_indexes = [[] for _ in range(self.num_partitions)]
        for i in range(labels.shape[0]):
            p_indexes[labels[i]].append(i)
            self.mapping[labels[i]].append(i)

        partitions = [data[p_indexes[idx]] for idx in range(self.num_partitions)]

        return partitions

    @overrides
    def build(self, dataset) -> BuildTimeStats:
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
        avg_ind = self.num_partitions_to_search
        query_start_time = time()
        index_list = [
            list(sorted([(l2distance(q, self.centers[i]), i) for i in range(len(self.index))], key=lambda x: x[0]))[
            :self.num_partitions_to_search] for q in query]
        results = [[self.index[idx].search([query[q_id]], k)[0] for _, idx in id_list] for q_id, id_list in
                   enumerate(index_list)]
        elapsed_query_time = time() - query_start_time

        merge_start_time = time()
        # Fix indexes

        results = [[[(t[0], self.mapping[index_list[q_id][ind_id][1]][t[1]]) for t in index[0]] for ind_id, index in enumerate(res)]
                   for q_id, res in enumerate(results)]

        result = [list(sorted([t for index in res for t in index], key=lambda x: x[0]))[:k] for res in results]

        elapsed_merge_time = time() - merge_start_time
        return result, QueryTimeStats(elapsed_query_time, elapsed_merge_time, len(query)), avg_ind
