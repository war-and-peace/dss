# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

from indexes.single_indexes.Index import Index
from indexes.single_indexes import hnsw_hnswlib
from indexes.utils.dataset import Dataset


def get_index(index_name: str, dataset: Dataset) -> Index:
    if index_name == 'hnsw_hnswlib':
        return hnsw_hnswlib.HnswHnswlib(index_name, dataset.get_size(), dataset.get_dimensions())
    else:
        return None
