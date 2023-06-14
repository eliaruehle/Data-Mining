import os
import sys
print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
import torch
from enum import Enum
from typing import List, Dict, Tuple, Union
from omegaconf import OmegaConf, DictConfig
from multiprocessing import Pool
from gpu.kmeans_torch import KMeansTorch
from gpu.tensor_matrix import TensorMatrix
from data.loader import DataLoader

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

    def __init__(self, mode:MODE, config:DictConfig, stacked:STACKED, data:DataLoader, dim_reduce:bool=False) -> None:
        
        self.mode = mode
        if mode == MODE.CPU:
            # should not affect anything, just for safety
            self.device = torch.device("cpu")
        else: 
            self.device = self._check_available_device()
        # the config file for the clustering
        self.config = config
        # specifies which kind of data we use
        self.stacked = stacked
        # the data object
        self.data = data
        # specifies wheter we will use 
        self.dim_reduce = dim_reduce
    
    def _check_available_device():
        """
        Function to check if a GPU is available and if yes, which one.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("No GPU available, running on CPU.")
            return torch.device("cpu")
    
    def run(self):
        """
        Function to call if you want to run the clustering.
        """
        if self.mode == MODE.CPU:
            self._run_cpu()
        elif self.mode == MODE.GPU:
            self._run_gpu()
        else:
            raise ValueError("The mode is not specified correctly.")
    
    def _run_cpu(self):
        pass

    def _run_gpu(self):
        # create a kmeans object for pytorch_clustering
        kmeans = KMeansTorch(self.config["num_clusters"], self.config["error"], device=self.device)
        matrix = TensorMatrix(self.config["save_dir"], len(self.data.get_strategies()), self.device)
        
        if self.stacked == STACKED.SINGLE:

            for metric in self.data.get_metrices():
                for dataset in self.data.get_datasets():
                    cluster_data = self.data.retrieve_tensor(metric, dataset)[1]
                    if cluster_data is None:
                        print(f"Shape Error, skip clustering for Dataset: {dataset}")
                        continue
                    if self.dim_reduce:
                        try:
                            cluster_data = self.reduce_gpu_data(cluster_data)
                        except AssertionError:
                            print(f"Shape Error, skip clustering for Dataset: {dataset}")
                            continue
                    _, labels = kmeans.fit(cluster_data)
                    matrix.update(labels)
            matrix.write_back()
                         
        
        elif self.stacked == STACKED.DATASET:
            pass
    
    def reduce_gpu_data(self, data:torch.Tensor) -> torch.Tensor:
        return data