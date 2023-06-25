import sys
import warnings

from sklearn.exceptions import ConvergenceWarning

from datasets.loader import Loader
from experiment_runner import ClusterRunner
from project_helper.method_types import CLUSTER_STRAT


def main(cluster_method: int):
    hpc: bool = False

    # Ignore convergence warnings in sklearn clusterings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Initialize loader depending on whether we are on HPC
    print("DEBUG: Start reading in data")

    if hpc:
        project_data: Loader = Loader("...")
    else:
        project_data: Loader = Loader("/home/ature/University/6th-Semester/Data-Mining/kp_test/strategies")

    print("DEBUG: Ready to load data")
    print("DEBUG: Initialize cluster-runner")

    # Initialize experiment runners
    cluster_runner = ClusterRunner(
        project_data,
        "clusterings",
        [
            CLUSTER_STRAT.KMEANS,
            CLUSTER_STRAT.SPECTRAL,
            CLUSTER_STRAT.OPTICS,
            CLUSTER_STRAT.DBSCAN,
            CLUSTER_STRAT.GAUSSIAN_MIXTURE,
        ],
        [3, 4, 5],
    )

    # Start clustering
    cluster_runner.run(cluster_method)


# Start main
print("Starting main ...")

if len(sys.argv) > 1:
    index = int(sys.argv[1])
    main(index)
