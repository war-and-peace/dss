# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

import numpy as np
from overrides import overrides


class Dataset:
    def __init__(self, name, dataset_path):
        self.name = name
        self.dataset_path = dataset_path
        self.dimensions = 0
        self.size = 0
        self.data = None
        self.loaded = False

    def get_data(self):
        return self.data

    def get_size(self):
        return self.size

    def get_dimensions(self):
        return self.dimensions

    def set_size(self, size):
        self.size = size

    def set_dimensions(self, dimensions):
        self.dimensions = dimensions

    def load_dataset(self, amount=-1):
        if self.loaded:
            return self.data
        data = np.array(np.load(self.dataset_path), dtype=np.float32)
        if amount == -1:
            amount = data.shape[0]
        self.data = data[:max(0, min(data.shape[0], amount))]
        self.dimensions = self.data.shape[1]
        self.size = self.data.shape[0]
        self.loaded = True
        return self.data

    def load_dataset_from_numpy(self, numpy_array):
        self.data = numpy_array
        self.dimensions = self.data.shape[1]
        self.size = self.data.shape[0]
        self.loaded = True
        return self.data

    def unload_dataset(self):
        self.data = None
        self.loaded = False

    def get_exact_query_results(self, queries, k, distance_function):
        results = [[(distance_function(self.data[idx], queries[q_id]), idx) for idx in range(self.data.shape[0])] for
                   q_id in range(len(queries))]

        return [[elem[1] for elem in list(sorted(res))[:k]] for res in results]


class BasicDataset(Dataset):
    def __init__(self, name, dataset_path):
        super().__init__(name, dataset_path)
