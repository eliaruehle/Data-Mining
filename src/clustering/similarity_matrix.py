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

    def __init__(self, labels: List[str], cluster_strategy_name: str) -> None:
        """
        Init function.

        Parameters:
        -----------
        labels : List[str]
            the labels which indicate the rows and columns of the matrix
        cluster_strategy_name : str
            the name of the cluster strategy the matrix keeps track of
        -----------

        Returns:
        --------
        None
            only the initialized object
        """
        self.labels = labels
        self.values = pd.DataFrame(index=labels, columns=labels, data=0)
        self.filename = cluster_strategy_name

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

        strategy_label: Dict[str, int] = {
            strategy: label for strategy, label in zip(strategies, labels)
        }
        for strategy, label in strategy_label.items():
            same_cluster: List[str] = [
                key
                for key, value in strategy_label.items()
                if value == label and label >= 0
            ]
            for sim_label_strategy in same_cluster:
                self.values.loc[strategy, sim_label_strategy] += 1

    def get_ordered_similarities(self) -> Dict[str, List[str]]:
        """
        Function to retrieve the ordered similarities.

        Parameters:
        ----------
        None

        Returns:
        --------
        similarities : Dict[str, List[str]]
            the dictionary with all similar strategies
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

        Parameters:
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

    @classmethod
    def from_csv(cls, filepath: str) -> 'SimilarityMatrix':
        """
        Class method to create a SimilarityMatrix object from a .csv file.

        Parameters:
        -----------
        filepath : str
            The path to the .csv file containing the similarity matrix data.

        Returns:
        --------
        SimilarityMatrix
            The created SimilarityMatrix object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("The specified file does not exist.")

        # Extract the filename and cluster strategy name from the filepath
        filename = os.path.splitext(os.path.basename(filepath))[0]
        cluster_strategy_name = filename.split("_")[0]

        # Read the data from the .csv file into a DataFrame
        values = pd.read_csv(filepath, index_col=0)

        # Extract the labels from the DataFrame
        labels = values.index.to_list()

        # Create a new SimilarityMatrix object and initialize its attributes
        similarity_matrix = cls(labels, cluster_strategy_name)
        similarity_matrix.values = values
        similarity_matrix.filename = filename

        return similarity_matrix

    def normalize(self) -> 'SimilarityMatrix':
        """
        Normalize the similarity matrix by dividing the values by the sum of the upper half.

        Parameters:
        -----------
        similarity_matrix : SimilarityMatrix
            The SimilarityMatrix object to be normalized.

        Returns:
        --------
        None
        """
        # Get the upper triangle of the similarity matrix
        upper_triangle = np.triu(self.values)

        # Calculate the sum of the upper triangle values
        upper_sum = np.sum(upper_triangle)

        # Normalize the similarity matrix values by dividing by the sum
        self.values /= upper_sum

        return self

    def as_2d_list(self):
        # Remove the strategy names (index and column names) from the DataFrame
        matrix = self.values.iloc[:, :].values

        # Convert any non-numeric values to floats
        matrix = matrix.astype(float)

        # Convert the matrix to a 2D list
        matrix_list = matrix.tolist()

        return matrix_list
