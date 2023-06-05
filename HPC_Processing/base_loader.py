from __future__ import annotations

import datetime
import logging
import os
from abc import ABC
from typing import Dict, List

import pandas as pd
from errors import NoSuchPathOrCSV

log_file = f"my_app_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # logs are printed to console
    ],
)


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
            os.path.join(
                base_dir,
                list(filter(lambda x: "done_workload" in x, os.listdir(base_dir)))[0],
            )
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
        Return a single csv file corresponding to the provided parameters.

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
            if requested path or csv doesn't exist
        """
        file_path = os.path.join(self.base_dir, strategy, dataset, metric + ".csv.xz")
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} was not found.")
            return None

        try:
            logger.info(f"Loading: {file_path}")
            return pd.merge(
                self.remove_nan_rows(pd.read_csv(file_path)),
                self.hyperparamters,
                on="EXP_UNIQUE_ID",
            )
        except EOFError:
            logger.error(f"File {file_path} appears to be corrupted. Skipping...")
            return None
        except Exception as e:
            logger.error(f"An error occurred while loading file {file_path}: {str(e)}")
            raise NoSuchPathOrCSV("An error occurred while loading the requested CSV.")

    def load_all_csv(self) -> None:
        """
        Function to read in all data files at once.

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
                for metric in os.listdir(
                    os.path.join(self.base_dir, strategy, dataset)
                ):
                    metric = metric[:-7]  # Remove ".csv.xz" from the file name
                    dataframe = self.load_single_csv(strategy, dataset, metric)
                    if dataframe is not None:
                        metric_file[metric] = dataframe
                    else:
                        logging.error(
                            f"Skipping file {strategy}/{dataset}/{metric}.csv.xz due to loading error or corruption"
                        )
                        continue
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
