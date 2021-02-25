# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 26.2.2021.
# ------------------------------------------------------------------------------

from indexes.single_indexes.hnsw_hnswlib import HnswHnswlib
from indexes.utils.dataset import BasicDataset
import Index

from sklearn.cluster import KMeans
from overrides import overrides
from typing import List, Tuple
import numpy as np
import random


class KMeansIndex(Index.Index):
    def __init__(self, name, max_elements, dimensions, metric='l2', distance_function=None, num_threads=-1,
                 num_partitions=4, ef_construction=1000, ef=1000, m=64):
        super().__init__(name, max_elements, dimensions, metric, distance_function, num_threads)
        self.num_partitions = num_partitions
        self.ef_construction = ef_construction
        self.ef = ef
        self.m = m
        self.single_partitions_size = 0
        self.mapping = None

    def partition(self, dataset):
        data = dataset.get_data()
        k_means = KMeans(n_clusters=self.num_partitions, random_state=0)
        k_means.fit(data)
        labels = k_means.labels_
        self.mapping = dict([(i, []) for i in range(self.num_partitions)])
        p_indexes = [[] for _ in range(self.num_partitions)]
        for i in range(labels.shape[0]):
            p_indexes[labels[i]].append(i)
            self.mapping[labels[i]].append(i)

        partitions = [data[p_indexes[idx]] for idx in range(self.num_partitions)]

        # indexes = list(range(dataset.get_size()))
        # random.shuffle(indexes)
        # self.mapping = [elem for elem in indexes]
        # data = dataset.get_data()
        #
        # single_partition_size = (dataset.get_size() + self.num_partitions - 1) // self.num_partitions
        # partitions = []

        # start, end, finish = 0, min(single_partition_size, dataset.get_size()), dataset.get_size()
        # while start < finish:
        #     partitions.append(data[indexes[start:end]])
        #     start += single_partition_size
        #     end = min(end + single_partition_size, dataset.get_size())

        # print(f'{self.num_partitions} - {len(partitions)}')
        # assert (self.num_partitions == len(partitions),
        #         f"Error while creating partitions. Could not create {self.num_partitions} partitions")

        # self.single_partitions_size = single_partition_size
        return partitions

    @overrides
    def build(self, dataset):
        partitions = self.partition(dataset)
        datasets = [BasicDataset(f'partition{idx}', '') for idx in range(self.num_partitions)]
        for idx, dataset in enumerate(datasets):
            dataset.load_dataset_from_numpy(partitions[idx])
        self.index = [HnswHnswlib('hnsw_hnswlib', datasets[idx].get_size(), datasets[idx].get_dimensions(), self.metric,
                                  self.distance_function, self.num_threads, self.ef_construction, self.ef, self.m) for
                      idx in range(self.num_partitions)]
        for idx, index in enumerate(self.index):
            index.build(datasets[idx])

    @overrides
    def search(self, query, k=5) -> List[List[Tuple[float, int]]]:
        results = [index.search(query, k) for index in self.index]

        # Fix indexes
        results = [
            [[(dist, self.mapping[idx][ids]) for dist, ids in t] for t in res] for
            idx, res in enumerate(results)]

        results_pair = [[result[res_id] for result in results] for res_id in range(len(query))]

        return [list(sorted([elem for pair in rp for elem in pair], key=lambda x: x[0]))[:k] for rp in results_pair]
