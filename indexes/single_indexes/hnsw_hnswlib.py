# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

import hnswlib
import Index as Index
from overrides import overrides
from typing  import List, Tuple


class HnswHnswlib(Index.Index):
    def __init__(self, name, max_elements, dimensions, metric='l2', distance_function=None, num_threads=-1,
                 ef_construction=1000, ef=1000, m=64):
        super().__init__(name, max_elements, dimensions, metric, distance_function, num_threads)
        self.ef_construction = ef_construction
        self.ef = ef
        self.m = m
        self.index = hnswlib.Index(space=self.metric, dim=self.dimensions)
        self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.m)
        self.index.set_ef(self.ef)

    @overrides
    def build(self, dataset):
        self.index.add_items(data=dataset.get_data(), num_threads=self.num_threads)

    @overrides
    def search(self, query, k=5) -> List[List[Tuple[float, int]]]:
        results = self.index.knn_query(data=query, k=k, num_threads=self.num_threads)
        return [[(results[1][res_id][idx], results[0][res_id][idx]) for idx in range(len(results[0][0]))] for res_id in
                range(len(query))]
