from datasets import Loader
from project_helper.Logger import Logger
from project_helper.method_types import CLUSTER_STRAT
from experiment_runner import ClusterRunner
import warnings
from sklearn.exceptions import ConvergenceWarning


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
def gaussian_mixture_clustering():
    data = Loader("kp_test")
    
    gm_clustering = GaussianMixtureClustering("gaussian_mixture", data.get_strategy_names())

    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
            if tracker % 10000 == 0:
                gm_clustering.write_cluster_results()
            for hyper_tuple in data.get_hyperparameter_for_metric_filtering():
                for strategy in data.get_strategy_names():
                    frame = data.get_single_dataframe(strategy, dataset, metric)
                    to_append = frame.loc[
                        (frame["EXP_RANDOM_SEED"] == hyper_tuple[0])
                        & (frame["EXP_START_POINT"] == hyper_tuple[1])
                        & (frame["EXP_BATCH_SIZE"] == hyper_tuple[2])
                        & (frame["EXP_LEARNER_MODEL"] == hyper_tuple[3])
                        & (frame["EXP_TRAIN_TEST_BUCKET_SIZE"] == hyper_tuple[4])
                    ]
                    to_append = to_append.iloc[:, :-9].dropna(axis=1)
                    if to_append.empty:
                        pass
                    else:
                        test.extend(to_append.to_numpy())
                if len(test) == len(data.get_strategy_names()):
                    try:

                        gm_clustering.cluster(test)
                    except:
                        Logger.info(ClusterFormatError("multiple values per cycle"))
                        test.clear()
                        pass
                    tracker += 1
                    test.clear()
                else:
                    test.clear()
            break
        break

    gm_clustering.write_cluster_results()
    pprint(gm_clustering.get_similiarity_matrix().get_orderd_similarities())


if __name__ == "__main__":
    gaussian_mixture_clustering()
    
   