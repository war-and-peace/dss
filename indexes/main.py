# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 9.2.2021.
# ------------------------------------------------------------------------------

import numpy as np
from single_indexes.hnsw_hnswlib import HnswHnswlib
from indexes.utils.dataset import ListedDataset
from indexes.utils.config import Config
from indexes.testers.SingleTester import SingleTester

CONFIG_PATH = 'config.json'
DATASET_PATH = '../datasets/embeddings.txt'
DATASET_NAME = 'Sample'
DIMENSIONS = 768
MAX_ELEMENTS = 10000

if __name__ == '__main__':
    config = Config(path=CONFIG_PATH)

    dataset = ListedDataset(DATASET_NAME, DATASET_PATH)
    data = dataset.load_dataset(amount=10000)
    index = HnswHnswlib(name='hnsw_hnswlib', max_elements=data.shape[0], dimensions=DIMENSIONS, num_threads=-1)

    tester = SingleTester(config, ready_indexes=[(index, dataset)])
    tester.test()
    tester.report()


    # print('index created')
    # index.build(data)
    # print('index built')
    # res = index.search(data[:5])
    # print(res)
