from typing import List
from sklearn.cluster import KMeans
import numpy as np
from clustering.base_cluster import BaseClustering


class KMeansClustering(BaseClustering):
    """
    Class for setup a KMeans clustering.
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
        kmeans: KMeans = KMeans(n_clusters=cluster_size, n_init=10).fit(data_vecs)
        # update the similarity matrix with retrieved labels
        self.similarity_matrix.update(self.labels, kmeans.labels_)

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
