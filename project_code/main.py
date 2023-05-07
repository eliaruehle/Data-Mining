from datasets import Loader
from side_handler.errors import ClusterFormatError
from clustering.kmeans import KMeansClustering
from clustering.spectral_clustering import SpectralClustering
from clustering.gaussian_mixture import GaussianMixtureClustering
from clustering.optics import OPTICSClustering
from typing import List
import numpy as np
from project_helper.Logger import Logger
from pprint import pprint
from project_helper.method_types import CLUSTER_STRAT


def kmeans_clustering():
    data = Loader("kp_test")

    kmeans_clustering = KMeansClustering("kmeans_test", data.get_strategy_names())

    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
            if tracker % 10000 == 0:
                kmeans_clustering.write_cluster_results(4)
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
                        kmeans_clustering.cluster(test, 4)
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


    kmeans_clustering.write_cluster_results(4)

    pprint(kmeans_clustering.get_similiarity_matrix().get_orderd_similarities())



def spectral_clustering():
    data = Loader("kp_test")

    spectral_clustering = SpectralClustering("spectral_clustering", data.get_strategy_names())

    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
            if tracker % 10000 == 0:
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
                    
                    if to_append.empty:
                        pass
                    else:
                        test.extend(to_append.to_numpy())
                
                if len(test) == len(data.get_strategy_names()):
                    try:
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


    spectral_clustering.write_cluster_results(4)

    pprint(spectral_clustering.get_similiarity_matrix().get_orderd_similarities())


def optics_clustering():
    data = Loader("kp_test")
    
    optics_clustering = OPTICSClustering("optics_clustering", data.get_strategy_names())

    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
            if tracker % 10000 == 0:
                optics_clustering.write_cluster_results()
                pass
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
                        optics_clustering.cluster(test)
                        
                        #print(optics_clustering.cluster(test))
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

    print("all labels: ", optics_clustering.labels)
    optics_clustering.write_cluster_results()

    pprint(optics_clustering.get_similiarity_matrix().get_orderd_similarities())
    #print(optics_clustering.cluster)


def gaussian_mixture_clustering():
    data = Loader("kp_test")
    
    gm_clustering = GaussianMixtureClustering("gaussian_mixture", data.get_strategy_names())


    tracker: int = 0
    test: List[np.ndarray] = list()
    for dataset in data.get_dataset_names():
        for metric in data.get_metric_names():
            if tracker % 10000 == 0:
                pass
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


def gm_test():
    data = Loader("kp_test")

    gm_clustering = GaussianMixtureClustering("gaussian_mixture_test", data.get_strategy_names())

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
                print(len(test))
                if len(test) == len(data.get_strategy_names()):
                    Logger.info("Start clustering iteration: " + str(tracker))
                    gm_clustering.cluster(test)
                    tracker += 1
                    test.clear()
                else:
                    test.clear()
            break
        break
    
    gm_clustering.write_cluster_results()
    print(gm_clustering.get_similiarity_matrix().get_orderd_similarities())


if __name__ == "__main__":
    #gaussian_mixture_clustering()
    optics_clustering()
    #kmeans_clustering()
   