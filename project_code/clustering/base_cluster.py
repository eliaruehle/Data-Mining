from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as np
from abc import ABC, abstractmethod
from project_code.clustering.similarity_matrix import SimilarityMatrix


class BaseClustering(ABC):
    """
    An abstract class defining the underlying structure fot the Clustering.
    """

    name: str
    similarity_matrix: SimilarityMatrix
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
    def cluster(self) -> None:
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

        Paramters:
        ----------
        None

        Returns:
        --------
        name : str
            the name of the cluster method
        """
        return self.name

    def get_similiarity_matrix(self) -> SimilarityMatrix:
        """
        Function to get the underlying similarity matrix of the cluster method.

        Parameters:
        -----------
        None

        Returns:
        --------
        similiarity_matrix : SimilarityMatrix
            the underlying similiarity matrix
        """
        return self.similarity_matrix
