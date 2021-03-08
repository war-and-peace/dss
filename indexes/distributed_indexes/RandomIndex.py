# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 25.2.2021.
# ------------------------------------------------------------------------------

from indexes.testers.TimeStats import BuildTimeStats, QueryTimeStats
from indexes.single_indexes.hnsw_hnswlib import HnswHnswlib
from indexes.utils.dataset import BasicDataset
import Index

from overrides import overrides
from typing import List, Tuple
from time import time
import random


class RandomIndex(Index.Index):
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

    def partition(self, dataset):
        indexes = list(range(dataset.get_size()))
        random.shuffle(indexes)
        self.mapping = [elem for elem in indexes]
        data = dataset.get_data()

        single_partition_size = (dataset.get_size() + self.num_partitions - 1) // self.num_partitions
        partitions = []

        start, end, finish = 0, min(single_partition_size, dataset.get_size()), dataset.get_size()
        while start < finish:
            partitions.append(data[indexes[start:end]])
            start += single_partition_size
            end = min(end + single_partition_size, dataset.get_size())

        self.single_partitions_size = single_partition_size
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
        avg_ind = self.num_partitions_to_search
        query_start_time = time()
        # results = [index.search(query, k)[0] for index in self.index]
        index_ids = random.sample(list(range(0, len(self.index))), self.num_partitions_to_search)
        # print(index_ids)
        results = [self.index[index_id].search(query, k)[0] for index_id in index_ids]
        elapsed_query_time = time() - query_start_time

        merge_start_time = time()
        # Fix indexes
        results = [
            [[(dist, self.mapping[int(ids + index_ids[idx] * self.single_partitions_size)]) for dist, ids in t] for t in
             res] for idx, res in enumerate(results)]

        results_pair = [[result[res_id] for result in results] for res_id in range(len(query))]

        result = [list(sorted([elem for pair in rp for elem in pair], key=lambda x: x[0]))[:k] for rp in results_pair]
        elapsed_merge_time = time() - merge_start_time
        return result, QueryTimeStats(elapsed_query_time, elapsed_merge_time, len(query)), avg_ind
