from __future__ import annotations
from abc import ABC
from typing import Dict, List
import os
import pandas as pd
from side_handler.errors import NoSuchPathOrCSV
from project_helper.Logger import Logger


class Base_Loader(ABC):
    """
    The Base_loader class is the underlying class for the Loader, who prepares the data for usage.
    """

    base_dir: str = ""
    hyperparamters: pd.DataFrame
    strategies: List[str] = list()
    datasets: List[str] = list()
    metrices: List[str] = list()
    data_dict: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = dict()

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
        self.base_dir = base_dir
        self.hyperparamters = pd.read_csv(
            base_dir
            + "/"
            + list(filter(lambda x: "done_workload" in x, os.listdir(base_dir)))[0]
        )
        self.strategies = sorted(
            [strat for strat in os.listdir(base_dir + "/") if strat[0].isupper()],
            key=str.lower,
        )
        self.datasets = sorted(
            [dset for dset in os.listdir(base_dir + "/" + self.strategies[0] + "/")],
            key=str.lower,
        )
        self.metrices = sorted(
            [
                metric[:-7]
                for metric in os.listdir(
                    base_dir + "/" + self.strategies[0] + "/" + self.datasets[0] + "/"
                )
            ],
            key=str.lower,
        )

    def load_single_csv(self, strategy: str, dataset: str, metric: str) -> pd.DataFrame:
        """
        Returna a single csv file corresponding to the provided parameters.

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
            the container with all data

        Raises:
        -------
        NoSuchPathOrCSV Error
            if requestes path or csv doesn't exist
        """
        try:
            Logger.debug(
                "Read in "
                + self.base_dir
                + "/"
                + strategy
                + "/"
                + dataset
                + "/"
                + metric
                + ".csv.xz"
            )
            return pd.merge(
                self.remove_nan_rows(
                    pd.read_csv(
                        self.base_dir
                        + "/"
                        + strategy
                        + "/"
                        + dataset
                        + "/"
                        + metric
                        + ".csv.xz"
                    )
                ),
                self.hyperparamters,
                on="EXP_UNIQUE_ID",
            )
        except:
            # Logger.except()
            raise NoSuchPathOrCSV("Path or requestes CSV does not exist!")

    def load_all_csv(self) -> None:
        """
        Fuction to read in all data files at once.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        for strategy in self.strategies:
            dataset_metric: Dict[str, Dict[str, pd.DataFrame]] = dict()
            for dataset in self.datasets:
                metric_file: Dict[str, pd.DataFrame] = dict()
                for metric in self.metrices:
                    metric_file[metric] = self.load_single_csv(
                        strategy, dataset, metric
                    )
                dataset_metric[dataset] = metric_file.copy()
            self.data_dict[strategy] = dataset_metric.copy()

    def remove_nan_rows(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Function for preprocessing. Removes all rows who only have np.nan values.

        Parameters:
        -----------
        data_frame : pd.DataFrame
            the dataframe for preprocessing

        Returns:
        --------
        data_frame : pd.DataFrame
            the cleared dataframe
        """
        data_frame = data_frame.dropna(subset=data_frame.columns[:-1], how="all")
        return data_frame
