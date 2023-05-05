from multiprocessing import Pool
from pprint import pprint
from typing import List

import numpy as np

from clustering.kmeans import KMeansClustering
from clustering.optics import OPTICSClustering
from clustering.spectral_clustering import SpectralClustering
from datasets import Loader
from experiment_runner import ClusterRunner
from project_helper.Logger import Logger
from project_helper.method_types import CLUSTER_STRAT
from side_handler.errors import ClusterFormatError


def cluster(strategy, dataset, metric, hyper_tuple):
    data = Loader("kp_test")
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
        return None
    else:
        return to_append.to_numpy()

def main():
    data = Loader("kp_test")

    spectral_clustering = SpectralClustering(
        "spectral_clustering", data.get_strategy_names()
    )

    tracker: int = 0
    test: List[np.ndarray] = list()
    
    with Pool() as pool:
        results = []
        for dataset in data.get_dataset_names():
            for metric in data.get_metric_names():
                for hyper_tuple in data.get_hyperparameter_for_metric_filtering():
                    for strategy in data.get_strategy_names():
                        results.append(pool.apply_async(cluster, (strategy, dataset, metric, hyper_tuple)))
        for r in results:
            result = r.get()
            if result is not None:
                test.extend(result)
            if len(test) == len(data.get_strategy_names()):
                try:
                    spectral_clustering.cluster(test, 4)
                except:
                    Logger.info(ClusterFormatError("multiple values per cycle"))
                tracker += 1
                test.clear()
            else:
                test.clear()

    spectral_clustering.write_cluster_results(4)

if __name__ == "__main__":
    main()
