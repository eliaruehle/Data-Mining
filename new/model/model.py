import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans


class MatrixFactorizationClustering(nn.Module):
    # setup the device:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    def __init__(
        self, num_matrices: int, matrix_dim: int, latent_dim: int, num_clusters: int
    ):
        super(MatrixFactorizationClustering, self).__init__()
        self.to(self.device)
        self.num_matrices = num_matrices
        self.matrix_dim = matrix_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters

        # define layers for matrix factorization
        self.encoder = nn.Linear(matrix_dim, latent_dim).to(self.device)
        self.decoder = nn.Linear(latent_dim, matrix_dim).to(self.device)

        # define the layer for clustering
        self.cluster_layer = nn.Linear(latent_dim, num_clusters).to(self.device)

    def forward(self, input_matrices):
        input_matrices.to(self.device)

        latent_factors = self.encoder(input_matrices)

        reconstructed_matrices = self.decoder(latent_factors)

        cluster_outputs = self.cluster_layer(latent_factors)

        predicted_labels = torch.argmax(cluster_outputs, dim=1)

        return reconstructed_matrices, cluster_outputs, predicted_labels

    def train(
        self,
        data: torch.Tensor,
        target_clusters: torch.Tensor,
        num_epochs: int,
    ):
        data = data.to(self.device)
        target_clusters = target_clusters.to(self.device)
        # adjust the loss to data
        loss = nn.KLDivLoss().to(device)
        # maybe change the optimizer and adjust the way the gradient is calculated
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            reconstructed_matrices, cluster_outputs, predicted_labels = self.forward(
                data
            )

            loss_val = loss(reconstructed_matrices, data) + loss(
                cluster_outputs, target_clusters
            )
            loss_val2 = loss(cluster_outputs, target_clusters)

            loss_val.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val.item()}")
        return predicted_labels


if __name__ == "__main__":
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    num_matrices = 40
    matrix_dim = 81000
    latent_dim = 50
    num_clusters = 8

    model = MatrixFactorizationClustering(
        num_matrices, matrix_dim, latent_dim, num_clusters
    )
    input_indices = torch.arange(num_matrices).to(device)
    target_clusters = torch.randint(num_clusters, (num_matrices,)).to(device)
    data = torch.randn(len(input_indices), matrix_dim).to(device)

    num_epochs = 1000
    print("start training")
    labels = model.train(data, target_clusters, num_epochs)
    print("end training")
    print("labels", labels)
