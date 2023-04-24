from typing import List
import os
import pandas as pd
import numpy as np


class Data:
    """
    The data class prepares the experimental data to make it easy to analyze.
    """

    hyperparameters: pd.DataFrame

    def __init__(self, base: str, hyerparameter_path: str) -> None:
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

        self.base: str = base
        self.strategies: List[str] = sorted(
            [(strat + "/") for strat in os.listdir(self.base) if strat[0].isupper()],
            key=str.lower,
        )
        self.datasets: List[str] = sorted(
            [(dset + "/") for dset in os.listdir(base + self.strategies[0])],
            key=str.lower,
        )
        self.metrices: List[str] = sorted(
            list(os.listdir(base + self.strategies[0] + self.datasets[0])),
            key=str.lower,
        )
        self.hyperparameters = pd.read_csv(self.base + hyerparameter_path)

    def __str__(self) -> str:
        """
        Function to print the used base directory if the str() method is called
        on data object.

        Paramters:
        ----------
        None

        Returns:
        --------
        base_name : str
            name of the currently used base directory
        """
        return "Base Directory: " + self.base

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
        return self.metrices

    def get_dataframe_by_metric(
        self, strategy: str, dataset: str, metric: str
    ) -> pd.DataFrame:
        """
        Function to get all data from a specific metric on a dataset where a specific strategy
        was applied.

        Parameters:
        -----------
        strategy : str
            the name of the strategy you want to have
        dataset : str
            the dataset you search in for the metric
        metrix : str
            the metric name you want to have

        Returns:
        --------
        pd.DataFrame
            the DataFrame which contains all data from the corresponding csv file
        """
        return pd.read_csv(
            self.base + strategy + "/" + dataset + "/" + metric + ".csv.xz"
        )

    def get_dataframe_by_metric_batchsize(
        self, strategy: str, dataset: str, metric: str, batch_size: int
    ) -> pd.DataFrame:
        """
        Function to get all data which belong to a specific batch size from a specific
        metric on a dataset where a specific strategy was applied.

        Parameters:
        -----------
        strategy : str
            the name of the strategy you want to have
        dataset : str
            the dataset you search in for the metric
        metric : str
            the metric name you want to have
        batch_size : int
            the batch size you want to have

        Returns:
        --------
        pd.DataFrame
            the DataFrame which contains all data from the filtered csv file

        Raises:
        -------
        ValueError
            if requested batch-size is not valid
        """

        if batch_size not in [1, 5, 10]:
            raise ValueError("Requested batch-size is not valid!")

        data_frame: pd.DataFrame = pd.read_csv(
            self.base + strategy + "/" + dataset + "/" + metric + ".csv.xz"
        )
        to_drop: List[int] = list()
        for index, row in data_frame.iterrows():
            # get the EXP_UNIQUE_ID as entry in the last row
            exp_unique_id: int = int(row[-1])
            # if row does not correspond to batch size mark index
            if (
                int(
                    self.hyperparameters[
                        self.hyperparameters["EXP_UNIQUE_ID"] == exp_unique_id
                    ]["EXP_BATCH_SIZE"].iloc[0]
                )
                != batch_size
            ):
                to_drop.append(index)
        # drop all rows with wrong batch sizes
        data_frame = data_frame.drop(index=to_drop)
        data_frame = data_frame.dropna(axis=1, how="any")
        # drop all nan columns
        return data_frame

    def get_numpy_vectors_batchsize(
        self, strategy: str, dataset: str, metric: str, batch_size: int
    ) -> List[np.ndarray]:
        """
        Function to get all data vectors which belong to a specific batch size from a specific
        metric on a dataset where a specific strategy was applied.

        Parameters:
        -----------
        strategy : str
            the name of the strategy you want to have
        dataset : str
            the dataset you search in for the metric
        metric : str
            the metric name you want to have
        batch_size : int
            the batch size you want to have

        Returns:
        --------
        numpy_vecs : List[np.ndarray]
            collection of the data vectors as numpy array
        """
        data_frame: pd.DataFrame = self.get_dataframe_by_metric_batchsize(
            strategy, dataset, metric, batch_size
        )
        numpy_vecs: List[np.ndarray] = [
            row[:-1].to_numpy() for _, row in data_frame.iterrows()
        ]

        return numpy_vecs


def main() -> None:
    """
    The main function of the file.
    """
    data: Data = Data("kp_test/", "05_done_workload.csv")
    print(str(data))


if __name__ == "__main__":
    main()
