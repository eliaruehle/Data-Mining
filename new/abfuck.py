import numpy as np
import torch
from pandarallel import pandarallel
from data import DataLoader
from model import KMeansTorch, Autoencoder, train

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
    # pandarallel.initialize(progress_bar=True)
    track = np.zeros((8, 8))
    loader = DataLoader()
    kmeans = KMeansTorch(3, 1e-6)
    for metric in loader.get_metrices()[:20]:
        for dataset in loader.get_datasets():
            all_data = loader.load_files_per_metric_and_dataset(metric, dataset)
            files = np.array([pair[1].to_numpy() for pair in all_data])
            files_tensor = torch.tensor(files, dtype=torch.float32, device="mps")
            label_names = [pair[0] for pair in all_data]
            if check_dimensions(files):
                dim0, dim1, dim2 = files_tensor.size()
                autoencoder = Autoencoder(dim1 * dim2, 2)
                reduced = autoencoder(files_tensor)
                reduced = autoencoder.encoder(reduced).unsqueeze(2)
                print(reduced.shape)
                _, labels = kmeans.fit(reduced)
            else:
                continue
            process_results([val for val in labels.tolist()])
    print(track)
