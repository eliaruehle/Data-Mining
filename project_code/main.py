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


def main():
    data = Loader("../kp_test")

    # kmeans_clustering = KMeansClustering("kmeans", data.get_strategy_names())
    spectral_clustering = SpectralClustering(
        "spectral_clustering", data.get_strategy_names()
    )

    # fest: Dataset, Metric, -> search for Hyperparameters that are similar
    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
            if tracker % 10000 == 0:
                # kmeans_clustering.write_cluster_results(4)
                spectral_clustering.write_cluster_results(4)
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
                    # print(to_append.to_numpy())
                    # print(to_append.shape)
                    # wenn to_append leer ist dann nicht hinzufügen
                    if to_append.empty:
                        pass
                    else:
                        test.extend(to_append.to_numpy())
                # print(len(test))
                if len(test) == len(data.get_strategy_names()):
                    try:
                        # kmeans_clustering.cluster(test, 4)
                        spectral_clustering.cluster(test, 4)
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

    # kmeans_clustering.write_cluster_results(4)
    # print(kmeans_clustering.get_similiarity_matrix().get_orderd_similarities())

    # optics_clustering.write_cluster_results()
    # print(optics_clustering.get_similiarity_matrix().get_orderd_similarities())


if __name__ == "__main__":
    main()
