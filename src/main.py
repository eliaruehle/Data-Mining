import warnings

from sklearn.exceptions import ConvergenceWarning

from datasets import Loader
from experiment_runner import ClusterRunner
from project_helper.Logger import Logger
from project_helper.method_types import CLUSTER_STRAT
from src.clustering.evaluation.top_k import TopK


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

    print("DEBUG: start reading in data")
    # initialize one data object
    PROJECT_DATA: Loader = Loader("/home/wilhelm/Uni/data_mining/Data-Mining/kp_test")
    print("DEBUG: ready to load data")

    print("DEBUG: intialize cluster-runner")
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
    print("DEBUG: ready to load cluster-runner")

    """# Logger.info(f"Start Running {str(CLUSTER_RUNNER)}")
    print("DEBUG: start cluster runner")
    CLUSTER_RUNNER.run()"""

    base_directory = "/home/wilhelm/Uni/data_mining/Data-Mining/strategies"

    top_k = TopK(PROJECT_DATA)
    top_k.collect_top_k(directory=base_directory, threshold=0.0, epsilon=0.05)
    top_k.collect_best_strategy_for_dataset(base_directory)
# start main
print("now start main")
main()
