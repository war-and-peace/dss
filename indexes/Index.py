# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 25.2.2021.
# ------------------------------------------------------------------------------
from typing import List, Tuple


class Index:
    def __init__(self, name: str, max_elements: int, dimensions: int, metric: str = 'l2', distance_function=None,
                 num_threads: int = -1):
        self.name = name
        self.max_elements = max_elements
        self.dimensions = dimensions
        self.metric = metric
        self.distance_function = distance_function
        self.num_threads = num_threads
        self.index = None

    def build(self, dataset):
        pass

    def search(self, query, k=5) -> List[List[Tuple[float, int]]]:
        pass

    def empty(self):
        del self.index
