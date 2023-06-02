import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class Autoencoder(nn.Module):
    """
    The autoencoder class for dimension reduction.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        The init function of the autoencoder.

        Paramters:
        ----------
        input_size : int
            the size of the input data
        output_size : int
            the size of the output
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, output_size).to("mps")
        self.decoder = nn.Linear(output_size, input_size).to("mps")

    def forward(self, x: Tensor):
        """
        Function for the forward step.

        Paramters:
        ----------
        x : Tensor
            the input data

        Returns:
        --------
        None
        """
        s, _, _ = x.size()
        x = x.view(s, -1)
        print(x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(s, -1)
        print(x.shape)
        return x


if __name__ == "__main__":
    matrix = torch.randn(2, 100, 50).to("mps")

    autoencoder = Autoencoder(100 * 50, 10)
    reduced_vec = autoencoder(matrix)
    reduced_vec = autoencoder.encoder(reduced_vec)
    print(reduced_vec.shape)
