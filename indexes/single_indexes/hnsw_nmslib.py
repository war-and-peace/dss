# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 5.1.2021.
# ------------------------------------------------------------------------------
import nmslib


class HnswNmslib:
    def __init__(self, dimensions, metric='l2', distance_function=None):
        self.dimensions = dimensions
        self.metric = metric
        self.distance_function = distance_function

        index = nmslib.Index(space=self.metric, dim=self.dimensions)

    def build(self):
        pass

    def search(self, vector, k=5):
        pass
