from __future__ import annotations
from enum import IntEnum, verify, UNIQUE


@verify(UNIQUE)
class CLUSTER_STRAT(IntEnum):
    """
    Class to keep track of used cluster methods.
    """

    KMEANS = 1
    SPECTRAL = 2
    DBSCAN = 3
    OPTICS = 4
    GAUSSIAN_MIXTURE = 5

@verify(UNIQUE)
class DATASETS(IntEnum):
    # TODO: implement this class for the new datasets
    pass