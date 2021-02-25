# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

from indexes.single_indexes.Index import Index
from indexes.testers.Report import SingleIndexTestReport

from indexes.utils.distance_function import get_distance_function
from indexes.utils.dataset import Dataset
from indexes.utils.config import Config

from typing import List, Tuple
from time import time
import random
import tqdm


def calc_recall(approximate, exact, ks):
    result = [0 for _ in range(len(ks))]
    for idx in range(len(exact)):
        for idk, k in enumerate(ks):
            exact_set = set(exact[idx][:k])
            result[idk] += sum([x in exact_set for x in approximate[idx][:k]]) / k
    return [(ks[idx], elem / len(exact)) for idx, elem in enumerate(result)]


class SingleTester:
    """
        Tests the given index on the given dataset for Recall, Build time while keeping track of dataset information:
        dimensionality of data and the size of the dataset.
    """

    def __init__(self, config: Config, index_list: List[str] = None, dataset_list: List[str] = None,
                 ready_indexes: List[Tuple[Index, Dataset]] = None):
        """
        Initializes the SingleTester object
        :param config: Object that contains the configuration information (e.g.: path for index and datasets)
        :param index_list: List of index names
        :param dataset_list: List of dataset names. Dataset will be loaded from the path stored in config object.
        :param ready_indexes: List of (Index, Dataset) pair. If this parameter is set, index_list and dataset_list
                is ignored.
        """

        self.config = config
        self.indexes = index_list
        self.datasets = dataset_list
        self.ready_indexes = ready_indexes
        self.default_queries = [5, 10, 20, 100]

        if self.ready_indexes is None:
            self.reports = [SingleIndexTestReport(r_id=idx, index_id=idx, dataset_id=d_id) for d_id in
                            range(len(self.datasets)) for idx in range(len(self.indexes))]
        else:
            self.reports = [SingleIndexTestReport(r_id=idx, index_id=idx, dataset_id=idx) for idx in
                            range(len(self.ready_indexes))]

    def test(self):
        if self.ready_indexes is None:
            for d_id, dataset_name in tqdm.tqdm(enumerate(self.datasets)):
                for index_id, index_name in tqdm.tqdm(enumerate(self.indexes)):
                    dataset = Dataset(name=dataset_name, dataset_path=self.config.get_dataset_path(dataset_name))
                    dataset.load_dataset()
                    index = self.config.get_index(index_name, dataset)
                    recall, build_time, query_time, data_dims, data_size = self.perform_single_test(index, dataset)
                    self.reports[index_id * len(self.indexes) + d_id].set_all_stats(recall, build_time, query_time,
                                                                                    data_dims, data_size)
        else:
            for idx, (index, dataset) in tqdm.tqdm(enumerate(self.ready_indexes)):
                recall, build_time, query_time, data_dims, data_size = self.perform_single_test(index, dataset)
                self.reports[idx].set_all_stats(recall, build_time, query_time, data_dims, data_size)

    def perform_single_test(self, index: Index, dataset: Dataset, queries: List[int] = None, query_num: int = 1):

        # Load dataset
        data = dataset.load_dataset()

        # Dimensionality of data
        if queries is None:
            queries = self.default_queries

        data_dims = dataset.get_dimensions()

        # Size of dataset
        data_size = dataset.get_size()

        # Measuring the build time
        build_start_time = time()
        index.build(dataset.get_data())
        build_finish_time = time()
        build_elapsed_time = build_finish_time - build_start_time

        # Query time and recall
        max_k = max(queries)

        ids = random.sample(range(1, data_size), query_num)
        qs = [data[query_id] for query_id in ids]

        query_start_time = time()
        results = index.search(query=qs, k=max_k)
        query_finish_time = time()

        query_time_total = (query_finish_time - query_start_time) / len(qs)

        # Get exact query results

        if index.distance_function is not None:
            exact_results = dataset.get_exact_query_results(qs, max_k, index.distance_function)
        else:
            exact_results = dataset.get_exact_query_results(qs, max_k, get_distance_function(index.metric))

        recall = calc_recall(results[0], exact_results, queries)
        return recall, build_elapsed_time, query_time_total, data_dims, data_size

    def report(self):
        # TODO: Fix the output format
        print(f'--------------------------------------------------------------------------------')
        print(f'\t\tReport')
        print(f'--------------------------------------------------------------------------------')
        print(f'|  No  | Recall                           | Build time | Query time | Data dimensions | Dataset size |')
        for report in self.reports:
            print(report.summary())
