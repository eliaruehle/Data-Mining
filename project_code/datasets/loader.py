from typing import List, Dict, Set, Tuple
from project_code.datasets.base_loader import Base_Loader
import pandas as pd
import numpy as np
from project_code.project_helper.Logger import Logger


class Loader(Base_Loader):
    """
    The Loader class provides methodes for data initialization and should
    be used in the main function to unpack all data files at once.
    """

    def __init__(self, base_dir: str) -> None:
        """
        Init function.

        Parameters:
        -----------
        base : str
            should contain the string of the base directory.
        -----------

        Returns:
        --------
        None
            only the initialized object
        """
        Logger.info("Start read in all data.")
        super().__init__(base_dir)
        self.load_all_csv()
        Logger.info("Finished read in all data.")

    def get_all_datafiles(self) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Returns all datafiles.

        Parameters:
        -----------
        None

        Returns:
        --------
        data_dict : Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
            the container with all data
        """
        return self.data_dict

    def get_single_dataframe(
        self, strategy: str, dataset: str, metric: str
    ) -> pd.DataFrame:
        """
        Returns a single dataframe.

        Parameters:
        -----------
        strategy : str
            the name of the strategy you want to have
        dataset : str
            the dataset you search in for the metric
        metric : str
            the metric name you want to have

        Returns:
        --------
        dataframe : pd.DataFrame
            the dataframe for the requested parameters
        """
        return self.data_dict[strategy][dataset][metric]

    def get_strategy_names(self) -> List[str]:
        """
        Function to get the names of all aplied strategies.

        Parameters:
        -----------
        None

        Returns:
        --------
        names : List[str]
            a list of all strategy names
        """
        return self.strategies

    def get_dataset_names(self) -> List[str]:
        """
        Function to get the names of the datasets we operate on.

        Parameters:
        -----------
        None

        Returns:
        --------
        names : List[str]
            a list of all dataset names
        """
        return self.datasets

    def get_metric_names(self) -> List[str]:
        """
        Function to get the names of existing metrics.

        Parameters:
        -----------
        None

        Returns:
        --------
        names : List[str]
            a list of all metric names
        """
        return self.metrices

    def get_hyperparameter_for_metric_filtering(
        self,
    ) -> Set[Tuple[int, int, int, int, int]]:
        """
        Function to get all important Hyperparamterconfigs to retrieve clustering data.

        Parameters:
        -----------
        None

        Returns:
        --------
        hyperparam_set : Set[Tuple[int, int, int, int, int]]
            a set of tuples containing the hyperparameter values in tuple
        """
        frame = self.hyperparamters.copy()
        # get the frame with the important columns
        frame = frame[
            [
                "EXP_RANDOM_SEED",
                "EXP_START_POINT",
                "EXP_BATCH_SIZE",
                "EXP_LEARNER_MODEL",
                "EXP_TRAIN_TEST_BUCKET_SIZE",
            ]
        ]
        hyperparam_set = set(frame.apply(tuple, axis=1))
        return hyperparam_set
