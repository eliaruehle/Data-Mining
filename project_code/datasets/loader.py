from typing import List, Dict
from datasets.base_loader import Base_Loader
from side_handler.errors import NoSuchPathOrCSV
import pandas as pd
import numpy as np


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
        super().__init__(base_dir)
        self.load_all_csv()

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
