from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
from data import DataLoader
from model import KMeansTorch
import torch
from time import time

global track


def check_dimensions(array: np.ndarray) -> bool:
    return all(
        np.array_equal(array[i].shape, array[0].shape) for i in range(1, array.shape[0])
    )


def process_results(labels):
    value_map = dict()
    for i, num in enumerate(labels):
        if num not in value_map:
            value_map[num] = [i]
        else:
            value_map[num].append(i)

    result = []
    for indices in value_map.values():
        if len(indices) > 1:
            result.append(indices)

    for list in result:
        for entries in list:
            for others in list:
                track[entries][others] += 1


if __name__ == "__main__":
    track = np.zeros((8, 8))
    loader = DataLoader()
    kmeans = KMeansTorch(2, 1e-6)
    pca = PCA(n_components=4)

    pca2 = TruncatedSVD(n_components=4)

    for metric in loader.get_metrices():
        for dataset in loader.get_datasets():
            start = time()
            all_data = loader.load_files_per_metric_and_dataset(metric, dataset)
            end = time()
            print("Loading time: ", end - start)
            start = time()
            print("Shape ", all_data[0][1].to_numpy().shape)
            files = torch.tensor(
                [
                    pca2.fit_transform(pca2.fit_transform(pair[1].to_numpy()).T)
                    for pair in all_data
                ],
                dtype=torch.float32,
            ).to("mps")
            print("Nur Tensor ", time() - start)
            _, labels = kmeans.fit(files)
            end = time()
            print("Clustering: ", end - start)
            print(labels)
