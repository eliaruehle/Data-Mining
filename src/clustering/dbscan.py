from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from typing import List
import numpy as np
from clustering.base_cluster import BaseClustering


class DBSCANClustering(BaseClustering):

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

        Explaination:
            - a larger eps value will result in larger clusters
            - a larger min_sample value will require more points to be present in the
              eps radius for a point to be considered a core point, resulting in smaller clusters
            - if min_sample is small there are more noise points
        """
        #### attention: the hyperparamter are only configured for the small dataset ####

        # first perform PCA to reduce the data-dimension, that will boost DBSCAN performance
        pca: PCA = PCA(n_components=2)
        reduced_data: np.ndarray = pca.fit_transform(data_vecs)
        # create sklearn OPTICS object
        dbscan: DBSCAN = DBSCAN(eps=0.1, min_samples=1).fit(reduced_data)
        # update the similarity matrix with retrieved labels‚
        self.similarity_matrix.update(self.labels, dbscan.labels_)

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
