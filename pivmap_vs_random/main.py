# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 13.10.2020.
# ------------------------------------------------------------------------------
import numpy as np
import tqdm

from pivmap_vs_random.random_part import RandomPartitioner
from pivmap_vs_random.kmeans_part import KMeansPartitioner
from pivmap_vs_random.pivmap import PivotMapping

DATASET_PATH = '/home/abdurasul/Code/pai/facts-search/embeddings.txt'
NUM_OF_PARTITIONS = 4


def load_embeddings(path_to_dataset):
    return np.load(path_to_dataset)


def evaluate(partitions):
    res = []
    total_sum = 0
    for partition in tqdm.tqdm(partitions):
        acc = 0
        cnt = 0
        for i in range(partition.shape[0] - 1):
            for j in range(i + 1, partition.shape[0]):
                acc += np.linalg.norm(partition[i] - partition[j])
                cnt += 1
        acc = acc / cnt
        res.append(acc)
        total_sum += acc
    return res, total_sum / len(partitions)


def evaluate_partitioner(partitioner, name='unkonwn'):
    print(f'\n\nPartitioning using {name} partitioner')
    partitions = partitioner.partition(NUM_OF_PARTITIONS)
    print(f'Number of elements in first partition: {partitions[0].shape}')
    res, mean = evaluate(partitions)
    print(res)
    print(f'mean: {mean}')


if __name__ == '__main__':

    embeddings = load_embeddings(DATASET_PATH)
    embeddings = embeddings[:4000]

    # Random partitioner
    randomPartitioner = RandomPartitioner(embeddings=embeddings, path_to_dataset='')
    evaluate_partitioner(randomPartitioner, name='random')

    # KMeans partitioner
    kmeansPartitioner = KMeansPartitioner(embeddings=embeddings, path_to_dataset='')
    evaluate_partitioner(kmeansPartitioner, name='kmeans')

    # Pivot mapping
    pivmapPartitioner = PivotMapping(embeddings=embeddings, path_to_dataset='')
    evaluate_partitioner(pivmapPartitioner, name='pivot mapping')
