# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

from overrides import overrides
from typing import List, Tuple
from testers.TimeStats import BuildTimeStats, QueryTimeStats


class Report:
    def __init__(self, r_id: int):
        self.r_id = r_id

    def summary(self, summary_entry):
        pass


def process_recall(recall):
    arr = [f"({pair[0]} - {pair[1]:.2f})" for pair in recall]
    return ' '.join(arr)


class SingleIndexTestReport(Report):
    def __init__(self, r_id: int, index_name: str, dataset_name: str):
        super().__init__(r_id)
        self.index_name = index_name
        self.dataset_name = dataset_name
        self.recall = []
        self.partition_time = 0
        self.build_time = 0
        self.query_time = 0
        self.query_merge_time = 0
        self.avg_ind = 0.0
        self.data_dims = 0
        self.data_size = 0

    def set_recall(self, recall: List[Tuple[int, int]]):
        self.recall = recall

    def set_all_stats(self, recall: List[Tuple[int, float]] = None, build_time: BuildTimeStats = None,
                      query_time: QueryTimeStats = None, avg_ind: float = None, data_dims: int = None,
                      data_size: int = None):
        if recall is not None:
            self.recall = recall

        if build_time is not None:
            self.build_time = build_time.build_time()
            self.partition_time = build_time.partition_time()

        if query_time is not None:
            self.query_time = query_time.query_time()
            self.query_merge_time = query_time.merge_time()

        if avg_ind is not None:
            self.avg_ind = avg_ind

        if data_dims is not None:
            self.data_dims = data_dims

        if data_size is not None:
            self.data_size = data_size

    def get_all_stats(self):
        return self.recall, self.partition_time, self.build_time, self.query_time, \
               self.query_merge_time, self.avg_ind, self.data_dims, self.data_size

    @overrides
    def summary(self, summary_entry):
        return summary_entry(self.r_id, self.index_name, self.dataset_name, process_recall(self.recall),
                             self.partition_time, self.build_time, self.query_time, self.query_merge_time, self.avg_ind,
                             self.data_dims, self.data_size)
