import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List
import os
import traceback
from .matrix import Matrix

class KmeansCPU:

    def __init__(self, config, labels:List[str], result_path:str) -> None:
        """
        The init Function of KmeansCPU.

        Parameters:
        -----------
        config: DictConfig
            The config for the CPU kmeans.
        labels: List[str]
            The names of the different strategies.
        result_path: str
            The path where all results are saved.

        Returns:
        --------
        None
        """
        self.config = config
        self.labels = labels
        self.pca = PCA(n_components=self.config["num_components"], svd_solver=self.config["svd_solver"])
        self.kmeans = KMeans(n_clusters=self.config["num_clusters"], init="k-means++", n_init="auto", tol=1e-4, max_iter=50)
        self.matrix = Matrix(labels, result_path)

    def step(self, data: np.ndarray, pca: bool, wb:bool, metric:str) -> None:
        """
        Performs one clustering step with PCA if reduction of dimensions is requested.
        Parameters:
        -----------
        data: np.ndarray
            The data for clustering.
        pca: bool
            Specifies if PCA is necessary.
        wb: bool
            Specifies if we would like to write back temporary results.
        metric: str
            The metric we calculate on.

        Returns:
        --------
        None
        """
        if pca:
            try:
                data = self.pca.fit_transform(data)
            except Exception as e:
                code_snippet = traceback.format_exc()
                print(f"Error in PCA! : {code_snippet}")
        try:
            k_labels: np.ndarray = self.kmeans.fit_predict(data)
        except Exception as e:
            code_snippet = traceback.format_exc()
            print(f"Error in PCA! : {code_snippet}")
        self.matrix.update(self.labels, k_labels)
        if wb:
            self.matrix.write_numeric_to_csv(metric)
            self.matrix.write_numeric_normalized_to_csv(metric)

    @property
    def get_matrix(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        matrix : Matrix
            The matrix for tracking cluster results.
        """
        return self.matrix

    @property
    def get_labels(self):
        """
       Parameters:
       -----------
       None

       Returns:
       --------
       labels : List[str]
           The labels used for matrix description.
       """
        return self.labels
