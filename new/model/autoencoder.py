import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.optim import Adam, lr_scheduler
from typing import List, Tuple
from tqdm import tqdm


class Autoencoder(nn.Module):
    """
    The autoencoder class for dimension reduction.
    """

    # sets the device for running accelerated on graphic cards
    device: str = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

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
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16384),
            nn.ReLU(),
            nn.Linear(16384, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 16384),
            nn.ReLU(),
            nn.Linear(16384, input_size),
        ).to(self.device)
        self.input_size = input_size
        self.output_size = output_size

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
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(s, -1)
        return x

    def get_device(self) -> str:
        """
        Returns the used device for the model.

        Parameters:
        -----------
        None

        Returns:
        --------
        device : str
            The used device.
        """
        return self.device

    def get_input_output_sizes(self) -> Tuple[int, int]:
        """
        Returns input and output sizes as tuple.

        Paramters:
        ----------
        None

        Returns:
        --------
        input_size, output_size : Tuple[int, int]
            The input and output size.
        """

        return self.input_size, self.output_size


def train(
    model: Autoencoder,
    train_samples: List[Tensor],
    learning_rate: float,
    num_epochs: int,
):
    """
    Training function for the autoencoder model.

    Paramters:
    ----------
    model : Autoencoder
        The model that needs to be trained.
    train_samples : List[Tensor]
        The training samples.
    learning_rate : float
        The learning rate for training.
    num_epochs : int
        The number of training iterations.

    Returns:
    --------
    None
    """
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    for epoch in range(num_epochs):
        sum_loss: float = 0.0
        for data in tqdm(train_samples):
            # set the gradients to zero
            optimizer.zero_grad()
            # get the result from model
            output = model(data)
            # calculate loss
            loss = criterion(output, data.view(data.size()[0], -1))
            # backpropagate the error
            loss.backward()
            # make an optimizer step
            optimizer.step()
            # save the loss
            sum_loss += loss.item()
        print(f"Everage loss in epoch {epoch}: {sum_loss/len(train_samples)}")
    # make a scheduler step
    scheduler.step()


if __name__ == "__main__":
    matrix = torch.rand(8, 450, 48).to("mps")
    trainings_data = [torch.rand(40, 100, 50).to("mps") for _ in range(20)]
    autoencoder = Autoencoder(450 * 48, 8)
    # print("Start Training")
    # train(autoencoder, trainings_data, learning_rate=0.001, num_epochs=50)
    reduced_vec = autoencoder(matrix)
    print(reduced_vec.shape)
    reduced_vec = autoencoder.encoder(reduced_vec)
    print(reduced_vec.shape)
    reduced_vec = reduced_vec.unsqueeze(2)
    print(reduced_vec.shape)
