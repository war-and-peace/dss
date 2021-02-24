# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

import hnswlib
import indexes.single_indexes.Index as Index
from overrides import overrides


class HnswHnswlib(Index.Index):
    def __init__(self, name, max_elements, dimensions, metric='l2', distance_function=None, num_threads=-1):
        super().__init__(name, max_elements, dimensions, metric, distance_function, num_threads)
        self.index = hnswlib.Index(space=self.metric, dim=self.dimensions)
        self.index.init_index(max_elements=self.max_elements, ef_construction=1000, M=64)
        self.index.set_ef(1000)

    @overrides
    def build(self, dataset):
        self.index.add_items(data=dataset, num_threads=self.num_threads)

    @overrides
    def search(self, query, k=5):
        return self.index.knn_query(data=query, k=k, num_threads=self.num_threads)
