import numpy as np
import torch
from pandarallel import pandarallel
from data import DataLoader, TensorMatrix
from model import KMeansTorch, Autoencoder, train


def main():
    data = DataLoader()
    kmeans = KMeansTorch(2, 1e-6)
    mat = TensorMatrix("new/results", len(data.get_strategies()), "mps")

    for metric in data.get_metrices()[:20]:
        for dataset in data.get_datasets():
            all_data = data.retrieve_tensor(metric, dataset)
            _, labels = kmeans.fit(all_data[1])
            mat.update(labels)
        mat.write_back()

if __name__ == "__main__":
    main()
