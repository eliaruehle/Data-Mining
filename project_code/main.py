from datasets import Loader
from side_handler.errors import ClusterFormatError
from clustering.kmeans import KMeansClustering
from clustering.spectral_clustering import SpectralClustering
from clustering.optics import OPTICSClustering
from typing import List
import numpy as np
from project_helper.Logger import Logger
from pprint import pprint
from project_helper.method_types import CLUSTER_STRAT
from experiment_runner import ClusterRunner


def main():
    # initialize one data object
    PROJECT_DATA: Loader = Loader("kp_test")

    # initialize experiment runners
    CLUSTER_RUNNER = ClusterRunner(
        PROJECT_DATA,
        "clusterings",
        [
            CLUSTER_STRAT.KMEANS,
            CLUSTER_STRAT.SPECTRAL,
            CLUSTER_STRAT.OPTICS,
            CLUSTER_STRAT.DBSCAN,
        ],
        [4],
    )

    Logger.info(f"Start Running {str(CLUSTER_RUNNER)}")
    CLUSTER_RUNNER.run()


if __name__ == "__main__":
    main()
