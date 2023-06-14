from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from clustering.similarity_matrix import SimilarityMatrix


class BaseClustering(ABC):
    """
    An abstract class defining the underlying structure fot the Clustering.
    """

    name: str
    similarity_matrix: SimilarityMatrix | List[SimilarityMatrix]
    labels: List[str]

    def __init__(self, labels: List[str], cluster_name: str) -> None:
        """
        Init function.

        Parameters:
        -----------
        labels : List[str]
            the labels for the data
        cluster_name : str
            the name of the applied cluster method - only an identifier
        -----------

        Returns:
        --------
        None
            only the initialized object
        """
        self.name = cluster_name
        self.labels = labels
        self.similarity_matrix = SimilarityMatrix(labels, cluster_name)

    @abstractmethod
    def cluster(self, data_vecs: List[np.ndarray]) -> None:
        """
        An abstract method every cluster-method should implement.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        ...

    def get_name(self) -> str:
        """
        Function to get the name of the cluster-method.

        Parameters:
        ----------
        None

        Returns:
        --------
        name : str
            the name of the cluster method
        """
        return self.name

    def get_similarity_matrix(self) -> SimilarityMatrix | List[SimilarityMatrix]:
        """
        Function to get the underlying similarity matrix of the cluster method.

        Parameters:
        -----------
        None

        Returns:
        --------
        similarity_matrix : SimilarityMatrix
            the underlying similarity matrix
        """
        return self.similarity_matrix

    @abstractmethod
    def write_cluster_results(self) -> None:
        """
        Function to write out cluster results into csv files.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        ...
