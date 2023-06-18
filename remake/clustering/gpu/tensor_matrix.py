import numpy as np 
import torch 
from typing import List, Dict, Set
from collections import defaultdict
import os

class TensorMatrix:

    def __init__(self, directory:str , num_labels:int, device:str) -> None:
        """
        The init function of the tensor matrix.

        Parameters:
        -----------
        directory : str
            The directory to save the matrix.
        num_labels : int
            The number of labels.
        device : str
            The device to use for the tensor matrix.

        Returns:
        --------
        None
            Only the initialized object
        """
        self.directory = directory
        self.num_labels = num_labels
        self.values = torch.zeros((num_labels, num_labels), dtype=torch.int32, device=device)

    def update(self, labels:torch.Tensor) -> None:
        """
        Function to update the matrix values after clustering.

        Parameters:
        -----------
        labels : torch.Tensor
            The labels of the data vectors.

        Returns:
        --------
        None
        """
        label_dict = defaultdict(list)
        for i, val in enumerate(labels):
            label_dict[val.item()].append(i)
        for eq in list(dict(label_dict).values()):
            for i in eq:
                for j in eq:
                    self.values[i, j] += 1
    
    
    def write_back(self) -> None:
        """
        Function to write the matrix back to the disk.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        np.savetxt(f"{self.directory}/kmeans.csv", self.values.cpu().numpy())
