# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

class Index:
    def __init__(self, name, max_elements, dimensions, metric='l2', distance_function=None, num_threads=-1):
        self.name = name
        self.max_elements = max_elements
        self.dimensions = dimensions
        self.metric = metric
        self.distance_function = distance_function
        self.num_threads = -1

    def build(self, dataset):
        pass

    def search(self, query, k=5):
        pass
