import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List
import os
import traceback
from .matrix import Matrix

class KmeansCPU:

    def __init__(self, config, labels:List[str], result_path:str):
        self.config = config
        self.labels = labels
        self.pca = PCA(n_components=self.config["num_components"], svd_solver=self.config["svd_solver"])
        self.kmeans = KMeans(n_clusters=self.config["num_clusters"], init="k-means++", n_init="auto", tol=1e-5, max_iter=100)
        self.matrix = Matrix(labels, result_path)

    def step(self, data: np.ndarray, pca: bool, wb:bool, metric:str):
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
        return self.matrix

    @property
    def get_labels(self):
        return self.labels
