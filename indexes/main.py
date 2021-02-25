# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 9.2.2021.
# ------------------------------------------------------------------------------

from indexes.distributed_indexes.RandomIndex import RandomIndex
from single_indexes.hnsw_hnswlib import HnswHnswlib
from indexes.utils.dataset import BasicDataset
from indexes.utils.config import Config
from indexes.testers.SingleTester import SingleTester

CONFIG_PATH = 'config.json'
DATASET_PATH = '../datasets/embeddings.txt'
DATASET_NAME = 'Sample'
DIMENSIONS = 768
MAX_ELEMENTS = 10000

if __name__ == '__main__':
    config = Config(path=CONFIG_PATH)

    dataset = BasicDataset(DATASET_NAME, DATASET_PATH)
    data = dataset.load_dataset(amount=-1)
    ef_construction = 10
    ef = 10
    M = 16

    index = HnswHnswlib(name='hnsw_hnswlib', max_elements=data.shape[0], dimensions=DIMENSIONS, num_threads=-1,
                        ef_construction=ef_construction, ef=ef, m=M)
    index2 = RandomIndex('Random index', data.shape[0], data.shape[1], num_threads=-1, ef_construction=ef_construction,
                         ef=ef, m=M, num_partitions=4)

    tester = SingleTester(config, ready_indexes=[(index, dataset), (index2, dataset)])
    # tester = SingleTester(config, ready_indexes=[(index2, dataset)])
    tester.test()
    report = tester.report(fmt='str')
    print(report)

    # index = RandomIndex('Random index', data.shape[0], data.shape[1])
    # index.build(dataset)
    # res = index.search([data[0]])
    # print(res)
