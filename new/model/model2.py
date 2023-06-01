import torch
import torch.nn as nn
import torch.optim as optim
from kmeans_pytorch import kmeans

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)


class NMF(nn.Module):
    def __init__(self, num_features: int):
        super(NMF, self).__init__()
        self.W = nn.Parameter(torch.randn(num_features, 1))
        self.H = nn.Parameter(torch.randn(1, num_features))

    def forward(self, X):
        return torch.matmul(self.W, self.H).to(device)


def nmf_factorization(data, num_features, num_iterations):
    model = NMF(num_features)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(num_iterations):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, data)
        loss.backward()
        optimizer.step()

    return model.W, model.H


def clustering(data, num_features, num_clusters):
    W, H = nmf_factorization(data, num_features, num_iterations=100)

    cluster_ids, cluster_centers = kmeans(
        X=W, num_clusters=num_clusters, distance="euclidean", device=device
    )

    return cluster_ids


if __name__ == "__main__":
    print("Start main")
    matrices = [
        torch.randn(10000, 10000).to(device),
        torch.randn(10000, 10000).to(device),
        torch.randn(10000, 10000).to(device),
    ]
    num_features = 10000
    num_clusters = 3

    for matrix in matrices:
        cluster_labels = clustering(matrix, num_features, num_clusters)
        # Access the cluster labels for each input matrix
        print("Clustering result for matrix:")
        print(cluster_labels)
        print()
