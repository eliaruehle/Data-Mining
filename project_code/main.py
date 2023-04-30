from datasets.loader import Loader
from side_handler.errors import NoSuchPathOrCSV
from clustering.kmeans import KMeansClustering
from typing import List
import numpy as np


def main():
    data = Loader("kp_test")
    kmeans_clustering = KMeansClustering("kmeans", data.get_strategy_names())

    # fest: Dataset, Metric, -> search for Hyperparameters that are similar
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
            for strategy in data.get_strategy_names():
                frame = data.get_single_dataframe(strategy, dataset, metric)
                to_append = frame.loc[
                    (frame["EXP_RANDOM_SEED"] == 0)
                    & (frame["EXP_START_POINT"] == 9)
                    & (frame["EXP_BATCH_SIZE"] == 5)
                    & (frame["EXP_LEARNER_MODEL"] == 5)
                    & (frame["EXP_TRAIN_TEST_BUCKET_SIZE"] == 0)
                ]
                to_append = to_append.iloc[:, :-9].dropna(axis=1)
                # print(to_append.to_numpy())
                # print(to_append.shape)
                test.extend(to_append.to_numpy())
            break
        break

    kmeans_clustering.cluster(test, 4)
    kmeans_clustering.write_cluster_results(4)
    print(kmeans_clustering.get_similiarity_matrix().get_orderd_similarities())


if __name__ == "__main__":
    main()
