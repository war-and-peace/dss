# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 24.2.2021.
# ------------------------------------------------------------------------------

import numpy as np


def l2distance(a, b):
    return np.linalg.norm(a - b) ** 2


def cosine_distance(a, b):
    return 2.0 - 2.0 * np.linalg.norm(np.linalg.norm(a) - np.linalg.norm(b))


def get_distance_function(metric='l2'):
    if metric == 'l2':
        return l2distance
    elif metric == 'cosine':
        return cosine_distance
    else:
        return None
