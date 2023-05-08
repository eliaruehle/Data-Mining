import warnings

from sklearn.exceptions import ConvergenceWarning

from datasets import Loader
from experiment_runner import ClusterRunner
from project_helper.Logger import Logger
from project_helper.method_types import CLUSTER_STRAT


def main() -> None:
    """
    The main function of the project.

    Paramters:
    ----------
    None

    Returns:
    --------
    None
    """
    # ignore convergence warnings in sklearns clusterings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # the next one could be resolved by adjust OPTICS parameter
    # TODO: adjust OPTICS Parameters to get rid of RuntimeWarnings instead of ignoring them
    warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        [3, 4, 5],
    )

    Logger.info(f"Start Running {str(CLUSTER_RUNNER)}")
    CLUSTER_RUNNER.run()


if __name__ == "__main__":
    main()
