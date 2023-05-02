from datasets.loader import Loader
from side_handler.errors import NoSuchPathOrCSV
from clustering.kmeans import KMeansClustering
from clustering.optics import OPTICSClustering
from clustering.dbscan import DBSCANClustering
from typing import List
import numpy as np
from project_helper.Logger import Logger


def kmeans():
    data = Loader("kp_test")

    kmeans_clustering = KMeansClustering("kmeans", data.get_strategy_names())
    

    # fest: Dataset, Metric, -> search for Hyperparameters that are similar
    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
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
                    Logger.info("Start clustering iteration: " + str(tracker))
                    kmeans_clustering.cluster(test, 4)
                    
                    tracker += 1
                    test.clear()
                else:
                    test.clear()
            break
        break

    kmeans_clustering.write_cluster_results(4)
    print(kmeans_clustering.get_similiarity_matrix().get_orderd_similarities())


def optics():

    data = Loader("kp_test")

    optics_clustering = OPTICSClustering("dbscan", data.get_strategy_names())

    # fest: Dataset, Metric, -> search for Hyperparameters that are similar
    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
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
                    Logger.info("Start clustering iteration: " + str(tracker))
                    optics_clustering.cluster(test)
                    tracker += 1
                    test.clear()
                else:
                    test.clear()
            break
        break

    optics_clustering.write_cluster_results()
    print(optics_clustering.get_similiarity_matrix().get_orderd_similarities())


def dbscan():

    data = Loader("kp_test")

    dbscan_clustering = DBSCANClustering("dbscan", data.get_strategy_names())

    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
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
                    Logger.info("Start clustering iteration: " + str(tracker))
                    dbscan_clustering.cluster(test)
                    tracker += 1
                    test.clear()
                else:
                    test.clear()
            break
        break

    dbscan_clustering.write_cluster_results()
    print(dbscan_clustering.get_similiarity_matrix().get_orderd_similarities())

if __name__ == "__main__":
    #main()
    #optics()
    dbscan()
