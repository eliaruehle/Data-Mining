import torch
from torch import Tensor
from typing import List, Tuple
import time


class KMeansTorch:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    def __init__(self, num_cluster: int, error: float):
        self.num_clusters = num_cluster
        self.error = error

    def fit(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        starting_time = time.time()
        # if the are more requested cluster_centers than entries in the tensor update the count of centers
        if data.size()[0] < self.num_clusters:
            self.num_clusters = data.size()[0]

        # converts collection of tensors into a single 3 dimensional Tensor
        input: Tensor = torch.Tensor(data).to(self.device)

        # initialize ceentroids with the first matrices
        centroids: Tensor = input[: self.num_clusters, :].to(self.device)

        # set a counter for needed iterations:
        counter: int = 0
        # run kmeans iterations
        while True:
            # update the counter in every step
            counter += 1
            # calculates the distance between all matrices in the 2nd and 3rd dimension
            # attention: to match the dimensions we extend the input to a dummy 4th dimension
            distances: Tensor = torch.norm(
                input[:, None] - centroids, p="fro", dim=(2, 3)
            )

            # assign the matrix to the centroid with the shortest distance
            assignment: Tensor = torch.argmin(distances, dim=1)
            # save the old centroids
            centroids_old = torch.clone(centroids).to(self.device)
            # update cluster centroids:
            for i in range(self.num_clusters):
                # extraxt matrices that are assigned to the i'th cluster
                cluster_matrices = input[assignment == i]
                # recalculate the centroids
                centroids[i] = torch.mean(cluster_matrices, dim=0)

            # if the variance only changes little than brak the calculation
            if torch.var(centroids_old) - torch.var(centroids) < self.error:
                print(
                    f"Used {counter} iterations in {time.time()-starting_time} seconds on {self.device}."
                )
                break

        return centroids, assignment

    def adjust_error(self, error: float) -> None:
        self.error = error

    def adjust_num_clusters(self, num_clusters: int) -> None:
        self.num_clusters = num_clusters


if __name__ == "__main__":
    kmeans = KMeansTorch(2, 0.001)

    # Tensor of zeros
    zeros_tensor = torch.zeros((100, 100))
    # Tensor of ones
    ones = torch.ones((100, 100))
    # Tensor of twelfs
    twelfs = torch.full((100, 100), 12)
    # Tensor of eights
    eights_tensor = torch.full((100, 100), 8)
    # Tensor of nines
    nines_tensor = torch.full((100, 100), 9)
    # stack matrices together
    matrices = torch.stack(
        (zeros_tensor, twelfs, ones, eights_tensor, nines_tensor), dim=0
    ).to(device="mps")
    _, labels = kmeans.fit(matrices)
    print(labels)
