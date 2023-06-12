import pandas as pd
import numpy as np
import os
from typing import List, Dict


class SimilarityMatrix:
    """
    This class represents a matrix which keep track of similarities between the different
    AL-strategies.
    """

    labels: List[str]
    values: pd.DataFrame
    filename: str
    directory: str = "cl_res"

    def __init__(self, labels: List[str], cluster_strat_name: str) -> None:
        """
        Init function.

        Parameters:
        -----------
        labels : List[str]
            the labels which indicate the rows and columns of the matrix
        cluster_strat_name : str
            the name of the cluster strategy the matrix keeps track of
        -----------

        Returns:
        --------
        None
            only the initialized object
        """
        self.labels = labels
        self.values = pd.DataFrame(index=labels, columns=labels, data=0)
        self.filename = cluster_strat_name

    def __str__(self) -> str:
        """
        Prettyprint of the matrix.

        Parameters:
        -----------
        None

        Returns:
        --------
        values : str
            the values of the matrix
        """
        return self.values.to_string()

    def update(self, strategies: List[str], labels: np.ndarray) -> None:
        """
        Function to update the matrix values after clustering.

        Parameters:
        -----------
        strategies : List[str]
            the strategies whose data vectors are labeled in clustering
        labels : np.ndarray
            the retrieved labels

        Returns:
        --------
        None
        """

        strat_label: Dict[str, int] = {
            strat: label for strat, label in zip(strategies, labels)
        }
        for strat, label in strat_label.items():
            same_cluster: List[str] = [
                key
                for key, value in strat_label.items()
                if value == label and label >= 0
            ]
            for sim_label_strat in same_cluster:
                self.values.loc[strat, sim_label_strat] += 1

    def get_orderd_similarities(self) -> Dict[str, List[str]]:
        """
        Function to retrieve the orderd similiarities.

        Paramters:
        ----------
        None

        Returns:
        --------
        similarities : Dict[str, List[str]]
            the dictionary with all similiar strategies
        """
        similarities: Dict[str, List[str]] = dict()
        for index, row in self.values.iterrows():
            sorted_indexes = row.sort_values(ascending=False).index.to_list()
            sorted_indexes.remove(index)
            similarities[index] = sorted_indexes.copy()
        return similarities

    def write_to_csv(self, additional_tag: str) -> None:
        """
        Function to write the matrix data into an .csv file.

        Paramters:
        ----------
        additional_tag : str
            an additional tag for specifying the name of the .csv file

        Returns:
        --------
        None
        """
        filepath: str = os.path.join(
            self.directory, self.filename + "_" + additional_tag + ".csv"
        )

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if os.path.exists(filepath):
            mode = "w"
        else:
            mode = "x"

        self.values.to_csv(filepath, index=True, mode=mode)
