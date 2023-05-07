import os
from abc import ABC

import pandas as pd


class Metrics(ABC):
    """
    A class to represent a Metrics object that can load CSV datasets.

    Attributes
    ----------
    file_path : str
        The path to the directory containing the CSV datasets.
    data_sets : list[str]
        A sorted list of CSV file names in the specified directory.
    data_sets_dict : dict[str, pd.DataFrame]
        A dictionary to store loaded CSV datasets, keyed by their file names.

    """

    def __init__(self, file_path: str) -> None:
        """
        Constructs all the necessary attributes for the Metrics object.

        Parameters
        ----------
            file_path : str
                The path to the directory containing the CSV datasets.
        """
        self.file_path = file_path
        self.data_sets = sorted(
            [
                data_set
                for data_set in os.listdir(self.file_path)
                if not os.path.isdir(os.path.join(self.file_path, data_set))
            ]
        )
        self.data_sets_list: list[pd.DataFrame] = list()

    def load_single_csv_dataset(self, data_set: str) -> pd.DataFrame:
        """
        Load a single CSV dataset into a DataFrame.

        Parameters
        ----------
            data_set : str
                The file name of the CSV dataset to load.

        Returns
        -------
            pd.DataFrame
                A DataFrame containing the loaded CSV dataset.
        """
        try:
            return pd.read_csv(
                self.file_path + "/" + data_set,
                usecols=lambda coloumn: coloumn != "LABEL_TARGET",
            )
        except:
            # TODO: Fix the Error to be from side_handler package
            raise ImportError(
                f"The given path: {self.file_path}/{data_set} or requested CSV: {data_set} does not exist."
            )

    def load_all_csv_datasets(self) -> None:
        """
        Load all CSV datasets in the specified directory into a list of DataFrames.

        This method iterates through the `data_sets` attribute, calling the `load_single_csv_dataset()` method for each file,
        and storing the resulting DataFrame in the `data_sets_list` attribute.
        """
        for data_item in self.data_sets:
            tmp = self.load_single_csv_dataset(data_item)
            tmp.name = data_item
            self.data_sets_list.append(tmp)


if __name__ == "__main__":
    metric = Metrics("kp_test/datasets")
    metric.load_all_csv_datasets()
    print(metric.data_sets_list)
