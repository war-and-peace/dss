# ------------------------------------------------------------------------------
#  Copyright (c) Abdurasul Rakhimov 3.3.2021.
# ------------------------------------------------------------------------------

class TimeStats:
    def __init__(self):
        self.total = 0

    def total_time(self) -> int:
        return self.total


class QueryTimeStats(TimeStats):
    def __init__(self, query_time, merge_time, n):
        super().__init__()
        self.n = n
        self.query = query_time / self.n
        self.merge = merge_time / self.n
        self.total = self.query + self.merge

    def query_time(self):
        return self.query

    def merge_time(self):
        return self.merge


class BuildTimeStats(TimeStats):
    def __init__(self, partition_time, build_time):
        super().__init__()
        self.partition = partition_time
        self.build = build_time

    def partition_time(self):
        return self.partition

    def build_time(self):
        return self.build
