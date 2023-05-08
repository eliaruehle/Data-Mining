from datasets.loader import Loader
from project_helper.method_types import CLUSTER_STRAT
from project_helper.Logger import Logger
from experiment_runner import BaseRunner
from typing import List
from clustering import (
    KMeansClustering,
    SpectralClustering,
    DBSCANClustering,
    OPTICSClustering,
    BaseClustering,
)
from side_handler import NoSuchClusterMethodError, ClusterFormatError
import numpy as np
import pandas as pd


class ClusterRunner(BaseRunner):
    """
    The class for running the clustering experiments.
    """

    labels: List[str]
    cluster_obj: List[BaseClustering]
    num_clusters: List[int]

    def __init__(
        self,
        data: Loader,
        name: str,
        components: List[CLUSTER_STRAT],
        num_clusters: List[int],
    ) -> None:
        """
        Initializes the ClusterRunner class.

        Paramters:
        ----------
        data : Loader
            the loaded data
        name : str
            the name of the runner object
        components : List[CLUSTER_STRAT]
            the cluster components that need to be created

        Returns:
        --------
        None
        """
        # Logger.info(f"Create ClusterRunner named {name}!")
        super().__init__(data, name, components)
        self.num_clusters = num_clusters
        self.labels = self.data.get_strategy_names()
        self.cluster_obj = self.create_cluster_objects()

    def create_cluster_objects(self) -> List[BaseClustering]:
        """
        Function that create the clustering objects.

        Paramters:
        ----------
        None

        Returns:
        --------
        cluster_obj : List[BaseClustering]
            the created objects for clustering
        """
        cluster_obj: List[BaseClustering] = list()

        for strat in self.components:
            match strat:
                case 1:
                    Strat: KMeansClustering = KMeansClustering(
                        "kmeans", self.labels, self.num_clusters
                    )
                case 2:
                    Strat: SpectralClustering = SpectralClustering(
                        "spec", self.labels, self.num_clusters
                    )
                case 3:
                    Strat: DBSCANClustering = DBSCANClustering("dbscan", self.labels)
                case 4:
                    Strat: OPTICSClustering = OPTICSClustering("optics", self.labels)
                case _:
                    raise NoSuchClusterMethodError(
                        "Requested clustering method not registered"
                    )
            cluster_obj.append(Strat)
        return cluster_obj

    def run(self) -> None:
        """
        Function to run the clustering on the entire dataset.

        Paramters:
        ----------
        None

        Returns:
        --------
        None
        """
        for dataset in self.data.get_dataset_names():
            for metric in self.data.get_metric_names():
                for hyper_tuple in self.data.get_hyperparameter_for_metric_filtering():
                    data_vectors: List[np.ndarray] = list()
                    for strategy in self.data.get_strategy_names():
                        frame: pd.DataFrame = self.data.get_single_dataframe(
                            strategy, dataset, metric
                        )
                        single_vec: pd.DataFrame = frame.loc[
                            (frame["EXP_START_POINT"] == hyper_tuple[0])
                            & (frame["EXP_BATCH_SIZE"] == hyper_tuple[1])
                            & (frame["EXP_LEARNER_MODEL"] == hyper_tuple[2])
                            & (frame["EXP_TRAIN_TEST_BUCKET_SIZE"] == hyper_tuple[3])
                        ]
                        single_vec = single_vec.iloc[:, :-9].dropna(axis=1)
                        # proof if we collected some data, or data didn't exist
                        if single_vec.empty:
                            # continue loop
                            # Logger.info(
                            #    f"Single vector was empty. Strategy {strategy} skipped."
                            # )
                            pass
                        else:
                            data_vectors.extend(single_vec.to_numpy())
                    if len(data_vectors) == self.data.NUM_STRATS:
                        try:
                            for clustering in self.cluster_obj:
                                clustering.cluster(data_vectors)
                            data_vectors.clear()
                        except:
                            # Logger.info(
                            #    ClusterFormatError("Error in clustering attempt!")
                            # )
                            data_vectors.clear()
                    else:
                        data_vectors.clear()
                break
            break

        # saves the results from the clusterings
        for cluster_strat in self.cluster_obj:
            cluster_strat.write_cluster_results()

    def get_components(self) -> List[str]:
        """
        Function to get the actice cluster (components) of the Experiment.

        Parameters:
        -----------
        None

        Returns:
        --------
        cluster_strategy_names : List[str]
            the names of the experiments cluster strategies
        """
        return [strat.name for strat in self.components]
