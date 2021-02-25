# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

from overrides import overrides
from typing import List, Tuple


class Report:
    def __init__(self, r_id: int):
        self.r_id = r_id

    def summary(self, summary_entry):
        pass


class SingleIndexTestReport(Report):
    def __init__(self, r_id: int, index_id: int, dataset_id: int):
        super().__init__(r_id)
        self.index_id = index_id
        self.dataset_id = dataset_id
        self.recall = []
        self.build_time = 0
        self.query_time = 0
        self.data_dims = 0
        self.data_size = 0

    def set_recall(self, recall: List[Tuple[int, int]]):
        self.recall = recall

    def set_all_stats(self, recall: List[Tuple[int, float]] = None, build_time: float = None,
                      query_time: float = None, data_dims: int = None, data_size: int = None):
        if recall is not None:
            self.recall = recall

        if build_time is not None:
            self.build_time = build_time

        if query_time is not None:
            self.query_time = query_time

        if data_dims is not None:
            self.data_dims = data_dims

        if data_size is not None:
            self.data_size = data_size

    def get_all_stats(self):
        return self.recall, self.build_time, self.query_time, self.data_dims, self.data_size

    @overrides
    def summary(self, summary_entry):
        return summary_entry(self.r_id, self.recall, self.build_time, self.query_time, self.data_dims, self.data_size)
