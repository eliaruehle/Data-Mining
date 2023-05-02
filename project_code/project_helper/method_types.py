from __future__ import annotations
from enum import IntEnum, unique


@unique
class CLUSTER_STRAT(IntEnum):
    KMEANS = 1
    SPECTRAL = 2
    OPTICS = 3
