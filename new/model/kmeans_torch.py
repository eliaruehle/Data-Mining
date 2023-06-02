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
        self.error = torch.tensor(error).to(self.device)

    def fit(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        starting_time = time.time()
        # if the are more requested cluster_centers than entries in the tensor update the count of centers
        if data.size()[0] < self.num_clusters:
            self.num_clusters = data.size()[0]

        # converts collection of tensors into a single 3 dimensional Tensor
        input: Tensor = torch.Tensor(data).to(self.device)

        # initialize ceentroids with the first matrices
        centroids: Tensor = input[: self.num_clusters, :].to(self.device)

        # initialize the distorsion
        distorsion: Tensor = torch.tensor([0.0]).to(self.device)

        # set a counter for needed iterations:
        counter: int = 0
        # run kmeans iterations
        while True:
            # update the counter in every step
            counter += 1
            # save the old distorsion value
            distorsion_old: Tensor = torch.clone(distorsion)
            # calculates the distance between all matrices in the 2nd and 3rd dimension
            # attention: to match the dimensions we extend the input to a dummy 4th dimension
            distances: Tensor = torch.norm(
                input[:, None] - centroids, p="fro", dim=(2, 3)
            )

            # assign the matrix to the centroid with the shortest distance
            assignment: Tensor = torch.argmin(distances, dim=1)
            # update cluster centroids:
            for i in range(self.num_clusters):
                # extraxt matrices that are assigned to the i'th cluster
                cluster_matrices = input[assignment == i]
                # recalculate the centroids
                centroids[i] = torch.mean(cluster_matrices, dim=0)
                # calculate distortion: distorsion = sum(||x-centroid||_F^2)
                distorsion = torch.sum(
                    torch.square(
                        torch.norm(cluster_matrices - centroids[i], p="fro", dim=(1, 2))
                    )
                )

            # if the distortion change is less than our formal formal requirement stop
            if torch.lt(
                abs(distorsion - distorsion_old),
                self.error,
            ):
                # TODO: write message into logger instead in print out
                print(
                    f"Used {counter} iterations in {time.time()-starting_time} seconds on {self.device}."
                )
                break

        return centroids, assignment

    def adjust_error(self, error: float) -> None:
        self.error = error

    def adjust_num_clusters(self, num_clusters: int) -> None:
        self.num_clusters = num_clusters

    def get_device(self):
        return self.device


if __name__ == "__main__":
    kmeans = KMeansTorch(10, 1e-3)

    start = time.time()
    for i in range(10):
        exp = torch.randn(40, 400, 100).to("mps")
        _, labels = kmeans.fit(exp)
        print(f"Labels: {labels}")
    print(f"Time used: {time.time()-start} sec")
