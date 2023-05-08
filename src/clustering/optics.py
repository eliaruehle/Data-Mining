from typing import List

import numpy as np
from project_helper import Logger
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA

from clustering.base_cluster import BaseClustering


class OPTICSClustering(BaseClustering):

    """
    Class for setup a OPTICS clustering.
    """

    def __init__(self, cluster_name: str, labels: List[str]) -> None:
        """
        Iniitalizes the class object.

        Paramters:
        ----------
        cluster_name : str
            the name for the cluster object
        labels : List[str]
            the labels for data vectors that are passed into the clustering

        Returns:
        --------
        None
        """
        super().__init__(labels, cluster_name)

    def cluster(self, data_vecs: List[np.ndarray]) -> None:
        """
        Implementation of abstract method from BaseClustering class.

        Paramters:
        ----------
        data_vecs : np.ndarray
            the data vectors for the clustering algorithm

        Returns:
        --------
        None
        """
        pca: PCA = PCA(n_components=2)
        reduced_data: np.ndaarray = pca.fit_transform(data_vecs)
        # create sklearn OPTICS object
        optics: OPTICS = OPTICS(
            min_samples=2, cluster_method="xi", xi=0.6, metric="euclidean"
        ).fit(reduced_data)
        # update the similarity matrix with retrieved labels
        Logger.debug("OPTICS labels:", optics.labels_)
        self.similarity_matrix.update(self.labels, optics.labels_)

    def write_cluster_results(self) -> None:
        """
        Function to write cluster results in .csv file.

        Parameters:
        -----------
        cluster_size : int
            the choosen cluster size for experiment

        Returns:
        --------
        None
        """
        self.similarity_matrix.write_to_csv("centers")
