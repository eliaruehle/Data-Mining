from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as np
from abc import ABC, abstractmethod
from clustering.similarity_matrix import SimilarityMatrix


class BaseClustering(ABC):
    name: str
    similarity_matrix: SimilarityMatrix
    labels: List[str]

    def __init__(self, labels: List[str], cluster_name: str) -> None:
        self.name = cluster_name
        self.labels = labels
        self.similarity_matrix = SimilarityMatrix(labels, cluster_name)

    @abstractmethod
    def cluster(self) -> None:
        ...

    def get_name(self) -> str:
        return self.name

    def get_similiarity_matrix(self) -> SimilarityMatrix:
        return self.similarity_matrix
