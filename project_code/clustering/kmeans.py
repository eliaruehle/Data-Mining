from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict
from clustering.base_cluster import BaseClustering


class KMeansClustering(BaseClustering):
    def __init__(self, cluster_name: str, labels: List[str]) -> None:
        super().__init__(labels, cluster_name)

    def cluster(self, data_vecs: np.ndarray, cluster_size: int) -> None:
        kmeans: KMeans = KMeans(n_clusters=cluster_size).fit(data_vecs)
        print(kmeans.labels_)
        self.similarity_matrix.update(self.labels, kmeans.labels_)

    def write_cluster_results(self, cluster_size: int) -> None:
        self.similarity_matrix.write_to_csv(str(cluster_size) + "centers")
