from datasets.loader import Loader
from project_helper.method_types import CLUSTER_STRAT
from experiment_runner import BaseRunner
from typing import Any, List
from clustering import (
    KMeansClustering,
    SpectralClustering,
    DBSCANClustering,
    OPTICSClustering,
    BaseClustering,
)
from side_handler import NoSuchClusterMethodError


class ClusterRunner(BaseRunner):
    """
    The class for running the clustering experiments.
    """

    labels: List[str]
    cluster_obj: List[BaseClustering]

    def __init__(
        self, data: Loader, name: str, components: List[CLUSTER_STRAT]
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
        super().__init__(data, name, components)
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
                    Strat: KMeansClustering = KMeansClustering("k_means", self.labels)
                case 2:
                    Strat: SpectralClustering = SpectralClustering(
                        "spectral", self.labels
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

    def run(self):
        pass

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
