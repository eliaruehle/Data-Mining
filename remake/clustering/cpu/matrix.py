import pandas as pd
import numpy as np
import os
from typing import List, Dict
import os
from collections import defaultdict
import json
import csv

class Matrix:

    def __init__(self, labels: List[str], result_path:str) -> None:
        """
        The init function of the matrix.

        Parameters:
        -----------
        labels: List[str]
            The labels used for row and column index.
        result_path: str
            The path where the results are saved.

        Returns:
        --------
        None
        """
        if not os.path.exists(result_path):
            raise FileNotFoundError("Result path does not exist!")
        self.labels = labels
        self.values = pd.DataFrame(index=self.labels, columns=self.labels, data=0)
        self.result_path = result_path

    def update(self, strategies: List[str], labels: np.ndarray) -> None:
        """

        Parameters:
        -----------
        strategies: List[str]
            The strategies as List.
        labels: np.ndarray
            The calculated cluster labels per strategy.

        Returns:
        --------
        None
        """
        strategy_label_dict: Dict[str, int] = {
            strategy: label for strategy, label in zip(strategies, labels)
        }
        for strategy, label in strategy_label_dict.items():
            same_cluster: List[str] = [
                key for key, value in strategy_label_dict.items()
                if value == label and label >= 0
            ]
            for label_to_strategy in same_cluster:
                self.values.loc[strategy, label_to_strategy] += 1
        return

    def get_results_as_dict(self) -> Dict[str, List[str]]:
        """
        Function to get cluster results as dict.

        Parameters:
        -----------
        None

        Returns:
        --------
        results : Dict[str, List[str]]
            The results as list.
        """
        similarities = defaultdict(list)
        for index, row in self.values.iterrows():
            sorted_indices = row.sort_values(ascending=False).index.to_list()
            sorted_indices.remove(index)
            similarities[index] = sorted_indices.cops()
        return dict(similarities)

    def get_results_as_json(self) -> None:
        """
        Function to get cluster results as json.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        result_dict = self.get_results_as_dict()
        with open(os.path.join(self.result_path, "cpu_cluster_result.json"), "w") as json_file:
            json.dump(result_dict, json_file)

    def get_result_as_csv(self):
        """
        Function to get cluster results as csv.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        result_dict = self.get_results_as_dict()
        with open(os.path.join(self.result_path, "cpu_cluster_result.csv"), "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=result_dict.keys())
            writer.writeheader()
            writer.writerow(result_dict)

    def write_numeric_to_csv(self, metric:str):
        """
        Function to write numeric results to csv.

        Parameters:
        -----------
        metric : str
            The metric the file should correspond to.

        Returns:
        --------
        None
        """
        if not os.path.exists(os.path.join(self.result_path, f"numeric_cluster_results_{metric}.csv")):
            mode = "x"
        else:
            mode = "w"

        self.values.to_csv(os.path.join(self.result_path, f"numeric_cluster_results_{metric}.csv"), index=True, mode=mode)

    def write_final_numeric_csv(self):
        """
        Function to write the final numeric results to csv.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        if not os.path.exists(os.path.join(self.result_path, "final_cluster.csv")):
            mode = "x"
        else:
            mode = "w"

        self.values.to_csv(os.path.join(self.result_path, "final_cluster.csv"), index=True, mode=mode)

    def write_final_numeric_normalized_csv(self):
        """
        Function to write the final normalized numeric results to csv.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        if not os.path.exists(os.path.join(self.result_path, "final_cluster_normalized.csv")):
            mode = "x"
        else:
            mode = "w"

        normalized_matrix = self.normalize()
        normalized_matrix.to_csv(os.path.join(self.result_path, "final_cluster_normalized.csv"), index=True, mode=mode)

    def write_numeric_normalized_to_csv(self, metric: str):
        """
        Function to write normalized numeric results to csv.

        Parameters:
        -----------
        metric : str
            The metric the file should correspond to.

        Returns:
        --------
        None
        """
        if not os.path.exists(os.path.join(self.result_path, f"numeric_cluster_results_normalized_{metric}.csv")):
            mode = "x"
        else:
            mode = "w"

        normalized_matrix: pd.DataFrame = self.normalize()
        normalized_matrix.to_csv(os.path.join(self.result_path, f"numeric_cluster_results_normalized_{metric}.csv"), index=True, mode=mode)

    def normalize(self):
        """
        Function to normalize the current data.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        diagonal: List[int] = [self.values.iloc[i][i] for i in range(len(self.labels))]
        try:
            assert all(x == diagonal[0] for x in diagonal)
        except:
            print("An error in the diagonal count occurs!")
        upper_limit: int = diagonal[0]
        matrix_normalized = self.values / upper_limit
        return matrix_normalized

    def matrix_to_list(self):
        """
        Function to transform the matrix structure into a 2-d list.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # create an 2-dimensional array
        return self.values.values.tolist()

    def set_values(self, df: pd.DataFrame) -> None:
        """
        Function to set the values to a new dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            The new dataframe.

        Returns:
        --------
        None
        """
        if df.isnull().values.any():
            print("Warning: There are null values in the frame!")
        self.values = df
