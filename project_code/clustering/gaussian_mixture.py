from typing import List
import numpy as np
from base_cluster import BaseClustering
import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianMixtureClustering(BaseClustering):

    """
    Class for setup a GaussianMixture clustering.
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
        # create sklearn OPTICS object
        gm: GaussianMixture = GaussianMixture(n_components=2, random_state=0).fit(data_vecs)
        # update the similarity matrix with retrieved labels
        self.similarity_matrix.update(self.labels, gm.predict(data_vecs))

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