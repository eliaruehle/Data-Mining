from typing import List, Dict
import os
import pandas as pd


class Data:
    """
    The data class prepares the experimental data to make it easy to analyze.
    """

    def __init__(self, base_dir: str, hyperparamters_path: str) -> None:
        """
        Init function to initialize the data object.

        Parameters:
        -----------
        base : str
            should contain the string of the base directory.
        hyperparamter_path : str
            should contain the complete path to the Hyperparamter file.
        -----------

        Returns:
        --------
        None
            only the initialized object is initialized
        """
        self.base_dir: str = base_dir
        self.hyperparameters: pd.DataFrame = pd.read_csv(base_dir + hyperparamters_path)
        self.strategies: List[str] = sorted(
            [(strat + "/") for strat in os.listdir(base_dir) if strat[0].isupper()],
            key=str.lower,
        )
        self.datasets: List[str] = sorted(
            [(dset + "/") for dset in os.listdir(base_dir + self.strategies[0])],
            key=str.lower,
        )
        self.metrices: List[str] = sorted(
            list(os.listdir(base_dir + self.strategies[0] + self.datasets[0])),
            key=str.lower,
        )
        self.all_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = self.unpack_all()

    def get_all_strategy_names(self) -> List[str]:
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
        return [name[:-1] for name in self.strategies]

    def get_all_dataset_names(self) -> List[str]:
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
        return [name[:-1] for name in self.datasets]

    def get_all_metric_names(self) -> List[str]:
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
        return [metric[:-7] for metric in self.metrices]

    def unpack_all(self) -> None:
        """
        Fuction to read in all data files at once.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # define the main data container
        all_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = dict()

        for strategy_name in self.get_all_strategy_names():
            # initialize dictionary for mapping drom dataset to metric
            dataset_metric: Dict[str, Dict[str, pd.DataFrame]] = dict()
            for dataset_name in self.get_all_dataset_names():
                # initialize dictionary for mapping from metric to csv-file
                metric_file: Dict[str, pd.DataFrame] = dict()
                for metric_name in self.get_all_metric_names():
                    # read in the file and merge the EXP_UNIQUE_ID to the hyperparameters
                    metric_file[metric_name] = pd.merge(
                        pd.read_csv(
                            self.base_dir
                            + strategy_name
                            + "/"
                            + dataset_name
                            + "/"
                            + metric_name
                            + ".csv.xz"
                        ),
                        self.hyperparameters,
                        on="EXP_UNIQUE_ID",
                    )
                # use copy method for a save input of the data into the underlying dictionary
                dataset_metric[dataset_name] = metric_file.copy()
            all_data[strategy_name] = dataset_metric.copy()
        # returns the data structure
        return all_data

    def get_all_data(self) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Returns all datafiles.

        Parameters:
        -----------
        None

        Returns:
        --------
        all_data : Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
            the container with all data
        """
        return self.all_data

    def get_dataframe(self, strategy: str, dataset: str, metric: str) -> pd.DataFrame:
        """
        Returns all datafiles.

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
        """
        return self.all_data[strategy][dataset][metric]
