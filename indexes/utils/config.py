# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------
import json
from indexes.utils.indexes import get_index
from Index import Index
from indexes.utils.dataset import Dataset


class Config:
    def __init__(self, path='config.json'):
        self.config = {}
        with open(path, 'r') as fd:
            self.config = json.load(fd)

    def get_index_path(self, index_name: str):
        if index_name not in self.config['indexes']:
            raise ValueError(f"Could not find the index with name: '{index_name}'")
        else:
            return self.config['indexes'][index_name]

    def get_dataset_path(self, dataset_name: str) -> str:
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Could not find the dataset with name: '{dataset_name}'")
        else:
            return self.config['datasets'][dataset_name]

    def get_index(self, index_name: str, dataset: Dataset) -> Index:
        if index_name not in self.config['indexes']:
            raise ValueError(f"Could not find the index with name: '{index_name}'")
        else:
            return get_index(index_name, dataset)
