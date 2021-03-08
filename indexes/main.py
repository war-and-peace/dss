# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 9.2.2021.
# ------------------------------------------------------------------------------

from indexes.distributed_indexes.KMeansIndex import KMeansIndex
from indexes.distributed_indexes.RandomIndex import RandomIndex
from indexes.distributed_indexes.PivotMappingIndex import PivotMappingIndex
from single_indexes.hnsw_hnswlib import HnswHnswlib
from indexes.utils.dataset import BasicDataset
from indexes.utils.config import Config
from indexes.testers.SingleTester import SingleTester
from copy import deepcopy

CONFIG_PATH = 'config.json'
DATASET_NAME = 'sample'
DATASET_PATH = '../datasets/embeddings.txt'

DATASET2_NAME = 'Glove25-400'
DATASET2_PATH = '../datasets/glove_embed.npy'

DATASET3_NAME = 'sample25-1m'
DATASET3_PATH = '../datasets/sample25-1m.npy'

DIMENSIONS = 768
MAX_ELEMENTS = 10000
AMOUNT = -1


if __name__ == '__main__':
    config = Config(path=CONFIG_PATH)

    # dataset3 = BasicDataset(DATASET3_NAME, DATASET3_PATH)
    # data3 = dataset3.load_dataset(amount=AMOUNT)
    dataset1 = BasicDataset(DATASET_NAME, DATASET_PATH)
    data1 = dataset1.load_dataset(amount=AMOUNT)
    dataset2 = BasicDataset(DATASET2_NAME, DATASET2_PATH)
    data2 = dataset2.load_dataset(amount=AMOUNT)

    ef_construction = 100
    ef = 100
    M = 16

    index1 = PivotMappingIndex(name='pivmap', max_elements=data1.shape[0], dimensions=data1.shape[1], num_threads=-1,
                        ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)
    index2 = RandomIndex('random', data1.shape[0], data1.shape[1], num_threads=-1,
                         ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)
    index3 = KMeansIndex(name='kmeans', max_elements=data1.shape[0], dimensions=data1.shape[1], num_threads=-1,
                         ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)

    index11 = PivotMappingIndex(name='pivmap', max_elements=data2.shape[0], dimensions=data2.shape[1], num_threads=-1,
                        ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)
    index12 = RandomIndex('random', data2.shape[0], data2.shape[1], num_threads=-1,
                          ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)
    index13 = KMeansIndex(name='kmeans', max_elements=data2.shape[0], dimensions=data2.shape[1], num_threads=-1,
                          ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)

    # index21 = PivotMappingIndex(name='pivmap', max_elements=data3.shape[0], dimensions=data3.shape[1], num_threads=-1,
    #                     ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)
    # index22 = RandomIndex('random', data3.shape[0], data3.shape[1], num_threads=-1,
    #                       ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)
    # index23 = KMeansIndex(name='kmeans', max_elements=data3.shape[0], dimensions=data3.shape[1], num_threads=-1,
    #                       ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)

    # index = PivotMappingIndex(name='pivmap', max_elements=data1.shape[0], dimensions=data1.shape[1], num_threads=-1,
    #                           ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)

    # index = RandomIndex('Random index', data1.shape[0], data1.shape[1], num_threads=-1,
    #                       ef_construction=ef_construction, ef=ef, m=M, num_partitions=4, num_partitions_to_search=2)
    # index = KMeansIndex(name='kmeans', max_elements=data1.shape[0], dimensions=data1.shape[1], num_threads=-1,
    #                       ef_construction=ef_construction, ef=ef, m=M, num_partitions=4)

    tester = SingleTester(config, ready_indexes=[(index1, dataset1), (index2, dataset1), (index3, dataset1),
                                                 (index11, dataset2), (index12, dataset2), (index13, dataset2)
                                                 # (index21, dataset3), (index22, datasetS3), (index23, dataset3)
                          ])
    # tester = SingleTester(config, ready_indexes=[(index3, dataset)])
    # tester = SingleTester(config, ready_indexes=[(index2, dataset1)])
    # tester = SingleTester(config, ready_indexes=[(index, dataset1)])
    tester.test()
    report = tester.report(fmt='str', save_path='report.html')
    print(report)

    # index = RandomIndex('Random index', data.shape[0], data.shape[1])
    # index.build(dataset)
    # res = index.search([data[0]])
    # print(res)
