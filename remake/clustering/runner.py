from enum import Enum
from omegaconf import OmegaConf, DictConfig
from .gpu.kmeans_torch import KMeansTorch
from .gpu.tensor_matrix import TensorMatrix
from data.loader import DataLoader
from .cpu.kmeans_cpu import KmeansCPU
import multiprocessing as mp
import numpy as np
from time import time
import warnings
from sklearn.exceptions import ConvergenceWarning


class MODE(Enum):
    """
    Specifies if we run the GPU accelerated version or simply on the CPU.
    """
    CPU = 0
    GPU = 1


class STACKED(Enum):
    """
    Specifies if we use stackes data or not.
    """
    SINGLE = 0
    DATASET = 1


class ClusterRunner:

    def __init__(self, mode: MODE, config: DictConfig, data: DataLoader, dim_reduce: bool = True,
                 stacked: STACKED = STACKED.SINGLE) -> None:

        self.mode = mode
        if mode == MODE.CPU:
            # should not affect anything, just for safety
            self.device = "cpu"
        #else:
        #    self.device = self._check_available_device()
        # the config file for the clustering
        self.config = config
        self.gpu_config = self.config["clustering"]["gpu"]
        self.cpu_config = self.config["clustering"]["cpu"]
        # specifies which kind of data we use
        self.stacked = stacked
        # the data object
        self.data = data
        # specifies whether we will use a dimension reduction
        self.dim_reduce = dim_reduce

    """
    def _check_available_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("No GPU available, running on CPU.")
            return torch.device("cpu")
    """

    def run(self, index:int):
        """
        Function to call if you want to run the clustering.
        """
        if self.mode == MODE.CPU:
            self._run_cpu(index)
        elif self.mode == MODE.GPU:
            self._run_gpu2()
        else:
            raise ValueError("The mode is not specified correctly.")

    def _run_cpu(self, index:int):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        cluster = KmeansCPU(self.cpu_config, self.data.get_strategies(), self.config["save_dir"])
        counter = 1
        metric = self.data.get_metrices()[index]
        start_glob = time()
        for dataset in self.data.get_datasets():
            start_lok = time()
            print(f"Start experiment on {dataset} and metric {metric}")
            runned_hypers = self.data.get_hyperparameter_for_dataset(dataset)
            print(f"Number of experiments sampled: {len(runned_hypers)}")
            labels, frames = self.data.load_data_for_metric_dataset(metric, dataset)
            if labels is None and frames is None:
                print("Nicht aufgenommen")
                continue
            for hyper in runned_hypers:
                to_process = list(zip(labels, frames, [hyper for _ in range(len(frames))]))
                with mp.Pool(mp.cpu_count()) as pool:
                    results = pool.map(self.data.get_row, to_process)
                pool.close()
                results = sorted(results, key=lambda x: x[0])
                rows = list(map(lambda x: x[1], results))
                if all((row.shape == rows[0].shape) and (row.shape[0] != 0) for row in rows):
                    to_cluster = np.array(rows)
                    print(f"Clustering for the {counter}-th time.")
                    cluster.step(to_cluster, self.dim_reduce, False, metric)
                    counter += 1
            cluster.get_matrix.write_numeric_to_csv(metric)
            print(f"Time used for {dataset}: {time()-start_lok} sec")
        cluster.get_matrix.write_numeric_normalized_to_csv(metric)
        print(f"Terminated normally for every dataset and metric {metric} in {(time()-start_glob)/3600} hours")

    def _run_gpu2(self):
        return

    def _run_gpu(self):
        # create a kmeans object for pytorch_clustering
        kmeans = KMeansTorch(self.gpu_config["num_clusters"], self.gpu_config["error"], device=self.device)
        matrix = TensorMatrix(self.config["save_dir"], len(self.data.get_strategies()), self.device)

        if self.stacked == STACKED.SINGLE:

            for metric in self.data.get_metrices():
                for dataset in self.data.get_datasets():
                    cluster_data = self.data.retrieve_tensor(metric, dataset)
                    if cluster_data is None:
                        print(f"Shape Error, skip clustering for metric {metric} on dataset: {dataset}")
                        continue
                    else:
                        cluster_data = cluster_data[1]
                    if self.dim_reduce:
                        try:
                            cluster_data = self.reduce_gpu_data(cluster_data)
                        except AssertionError:
                            print(f"Shape Error, skip clustering for Dataset: {dataset}")
                            continue
                    _, labels = kmeans.fit(cluster_data)
                    print("labels", labels)
                    matrix.update(labels)
            matrix.write_back()