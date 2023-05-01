from typing import List
from sklearn.cluster import SpectralClustering as SpecCluster
import numpy as np
from project_code.clustering.base_cluster import BaseClustering


class SpectralClustering(BaseClustering):
    """
    Class for setup a Spectral clustering.
    """

    def __init__(self, cluster_name: str, strategies: List[str]) -> None:
        """
        Iniitalizes the class object.

        Paramters:
        ----------
        cluster_name : str
            the name for the cluster object
        strategies : List[str]
            the labels for data vectors that are passed into the clustering

        Returns:
        --------
        None
        """
        super().__init__(strategies, cluster_name)

    def cluster(self, data_vecs: List[np.ndarray], cluster_size: int) -> None:
        """
        Implementation of abstract method from BaseClustering class.

        Paramters:
        ----------
        data_vecs : np.ndarray
            the data vectors for the clustering algorithm
        cluster_size : int
            the number of different clusters in the algorithm

        Returns:
        --------
        None
        """
        # create sklearn KMeans object
        spec_cluster: SpecCluster = SpecCluster(n_clusters=cluster_size, n_init=10).fit(data_vecs)
        # update the similarity matrix with retrieved labels
        self.similarity_matrix.update(self.labels, spec_cluster.labels_)

    def write_cluster_results(self, cluster_size: int) -> None:
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
        self.similarity_matrix.write_to_csv(str(cluster_size) + "centers")
