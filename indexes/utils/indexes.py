# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

from Index import Index
from indexes.single_indexes import hnsw_hnswlib
from indexes.utils.dataset import Dataset
from typing import Union


def get_index(index_name: str, dataset: Dataset) -> Union[Index, None]:
    if index_name == 'hnsw_hnswlib':
        return hnsw_hnswlib.HnswHnswlib(index_name, dataset.get_size(), dataset.get_dimensions())
    else:
        return None
