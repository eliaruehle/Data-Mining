import warnings

from sklearn.exceptions import ConvergenceWarning

from datasets import Loader
from experiment_runner import ClusterRunner
from project_helper.method_types import CLUSTER_STRAT


def main() -> None:
    """
    The main function of the project.

    Parameters:
    ----------
    None

    Returns:
    --------
    None
    """
    # ignore convergence warnings in sklearn clusterings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # the next one could be resolved by adjust OPTICS parameter
    # TODO: adjust OPTICS Parameters to get rid of RuntimeWarnings instead of ignoring them
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # initialize loader
    print("DEBUG: start reading in data")
    PROJECT_DATA: Loader = Loader("/home/wilhelm/Uni/data_mining/Data-Mining/kp_test")
    print("DEBUG: ready to load data")

    print("DEBUG: initialize cluster-runner")
    # initialize experiment runners
    CLUSTER_RUNNER = ClusterRunner(
        PROJECT_DATA,
        "clusterings",
        [
            CLUSTER_STRAT.KMEANS,
            CLUSTER_STRAT.SPECTRAL,
            CLUSTER_STRAT.OPTICS,
            CLUSTER_STRAT.DBSCAN,
            CLUSTER_STRAT.GAUSSIAN_MIXTURE,
        ],
        [4, 5],
    )


# start main
print("now start main")
main()
