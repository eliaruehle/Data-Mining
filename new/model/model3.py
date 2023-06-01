import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans


class GNN(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Assuming matrices is a list of PyTorch tensors
matrices = [
    torch.randn(10, 8),
    torch.randn(10, 9),
    torch.randn(10, 7),
    torch.randn(10, 7),
    torch.randn(20, 24),
]
num_features = 10
hidden_dim = 16
num_clusters = 3

# Step 1: Convert matrices to graphs
graphs = []
for matrix in matrices:
    # Construct the adjacency matrix representation
    adj_matrix = torch.Tensor(matrix)
    edge_index = adj_matrix.nonzero(as_tuple=False).t()  # Construct edge indices
    x = torch.eye(matrix.size(0))  # Node features (identity matrix in this example)
    graph = Data(x=x, edge_index=edge_index)
    graphs.append(graph)

# Step 2: Create GNN model and train
model = GNN(num_features, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100

for epoch in range(num_epochs):
    for graph in graphs:
        optimizer.zero_grad()
        embeddings = model(graph.x, graph.edge_index)
        loss = torch.norm(embeddings)  # Define a suitable loss function
        loss.backward()
        optimizer.step()

# Step 3: Apply clustering on flattened GNN embeddings
flattened_embeddings = []
for graph in graphs:
    embeddings = model(graph.x, graph.edge_index).detach().numpy()
    flattened_embeddings.append(embeddings.flatten())

kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(flattened_embeddings)

# Print cluster labels for each matrix
for i, label in enumerate(cluster_labels):
    print("Matrix", i + 1, "Cluster Label:", label)
