# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

from Index import Index
from indexes.testers.Report import SingleIndexTestReport

from indexes.utils.distance_function import get_distance_function
from indexes.utils.dataset import Dataset
from indexes.utils.config import Config

from dataclasses import make_dataclass
from typing import List, Tuple, Union
from tabulate import tabulate
from time import time
import pandas as pd
import random
import tqdm


def calc_recall(approximate, exact, ks):
    result = [0 for _ in range(len(ks))]
    for idx in range(len(exact)):
        for idk, k in enumerate(ks):
            exact_set = set(exact[idx][:k])
            result[idk] += sum([x[1] in exact_set for x in approximate[idx][:k]]) / k
    return [(ks[idx], elem / len(exact)) for idx, elem in enumerate(result)]


class SingleTester:
    """
        Tests the given index on the given dataset for Recall, Build time while keeping track of dataset information:
        dimensionality of data and the size of the dataset.
    """

    def __init__(self, config: Config, index_list: List[str] = None, dataset_list: List[str] = None,
                 ready_indexes: List[Tuple[Index, Dataset]] = None, query_num: int = 10):
        """
        Initializes the SingleTester object
        :param config: Object that contains the configuration information (e.g.: path for index and datasets)
        :param index_list: List of index names
        :param dataset_list: List of dataset names. Dataset will be loaded from the path stored in config object.
        :param ready_indexes: List of (Index, Dataset) pair. If this parameter is set, index_list and dataset_list
                is ignored.
        """

        self.config = config
        self.query_num = query_num
        self.indexes = index_list
        self.datasets = dataset_list
        self.ready_indexes = ready_indexes
        self.default_queries = [5, 10, 20, 100]

        if self.ready_indexes is None:
            self.reports = [
                SingleIndexTestReport(r_id=idx, index_name=self.indexes[idx], dataset_name=self.datasets[d_id]) for d_id
                in range(len(self.datasets)) for idx in range(len(self.indexes))]
        else:
            self.reports = [SingleIndexTestReport(r_id=idx, index_name=self.ready_indexes[idx][0].name,
                                                  dataset_name=self.ready_indexes[idx][1].name) for idx in
                            range(len(self.ready_indexes))]

    def test(self):
        if self.ready_indexes is None:
            for d_id, dataset_name in tqdm.tqdm(enumerate(self.datasets)):
                for index_id, index_name in tqdm.tqdm(enumerate(self.indexes)):
                    dataset = Dataset(name=dataset_name, dataset_path=self.config.get_dataset_path(dataset_name))
                    dataset.load_dataset()
                    index = self.config.get_index(index_name, dataset)
                    recall, build_time, query_time, avg_ind, data_dims, data_size = self.perform_single_test(index, dataset)
                    self.reports[index_id * len(self.indexes) + d_id].set_all_stats(recall, build_time, query_time,
                                                                                    data_dims, data_size)
        else:
            for idx, (index, dataset) in tqdm.tqdm(enumerate(self.ready_indexes)):
                recall, build_time, query_time, avg_ind, data_dims, data_size = self.perform_single_test(index, dataset)
                self.reports[idx].set_all_stats(recall, build_time, query_time, avg_ind, data_dims, data_size)

    def perform_single_test(self, index: Index, dataset: Dataset, queries: List[int] = None):

        # Load dataset
        data = dataset.load_dataset()

        # Dimensionality of data
        if queries is None:
            queries = self.default_queries

        data_dims = dataset.get_dimensions()

        # Size of dataset
        data_size = dataset.get_size()

        build_stats = index.build(dataset)

        # Query time and recall
        max_k = max(queries)

        ids = random.sample(range(1, data_size), self.query_num)
        qs = [data[query_id] for query_id in ids]

        results, query_stats, avg_ind = index.search(query=qs, k=max_k)

        # Get exact query results

        if index.distance_function is not None:
            exact_results = dataset.get_exact_query_results(qs, max_k, index.distance_function)
        else:
            exact_results = dataset.get_exact_query_results(qs, max_k, get_distance_function(index.metric))
        er_indexes = [[elem[1] for elem in res] for res in exact_results]
        # print(exact_results)
        # print(results)
        recall = calc_recall(results, er_indexes, queries)
        return recall, build_stats, query_stats, avg_ind, data_dims, data_size

    def report(self, fmt: str = 'df', save_path: str = None) -> Union[pd.DataFrame, str]:
        summary_entry = make_dataclass("Entry", [('No', int), ('Index', str), ('Dataset', str), ('Recall', str),
                                                 ('Partition', float), ('Build', float),
                                                 ('Query', float), ('Merge', float), ('avg_ind', float),
                                                 ('dims', int), ('N', int)])
        df = pd.DataFrame([report.summary(summary_entry) for report in self.reports])
        if save_path:
            with open(save_path, 'w') as fd:
                fd.write(tabulate(df, headers='keys', tablefmt='html'))

        if fmt == 'str':
            return tabulate(df, headers='keys', tablefmt='psql')
        elif fmt == 'df':
            return df
        elif fmt == 'html':
            return tabulate(df, headers='keys', tablefmt='html')
        else:
            raise ValueError(f"Format {fmt} is invalid. Supported formats: (str, df, html)")
