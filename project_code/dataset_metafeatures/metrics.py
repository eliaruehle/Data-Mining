import os
from abc import ABC

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew, zscore
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity



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
        self.metafeatures_dict: dict[str, list[int]] = dict()

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
        full_file_path = os.path.join(self.file_path, data_set)
        try:
            df = pd.read_csv(
                full_file_path,
                usecols=lambda coloumn: coloumn != "LABEL_TARGET",
            )
            df.name = data_set
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError) as exception:
            raise FileNotFoundError(
                f"The given path: '{full_file_path}' or requested CSV: '{data_set}' does not exist."
            ) from exception

    def load_all_csv_datasets(self) -> None:
        """
        Load all CSV datasets in the specified directory into a list of DataFrames.

        This method iterates through the `data_sets` attribute, calling the `load_single_csv_dataset()` method for each file,
        and storing the resulting DataFrame in the `data_sets_list` attribute.
        """
        for data_item in self.data_sets:
            tmp = self.load_single_csv_dataset(data_item)
            self.data_sets_list.append(tmp)

    def number_of_features(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the number of features in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the number of features (columns) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of features needs to be calculated.

        """
        features_n = len(data_set.columns)

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(features_n)

    def number_of_examples(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the number of examples (rows) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the number of examples (rows) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """
        examples_n = len(data_set)

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(examples_n)

    def proportion_of_missing_values(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the proportion of missing value per example (row) in a given DataFrame and store it in the
        metafeatures_dict.

        This method calculates the proportion of missing value per example (row) in the input DataFrame and appends the
        value to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """
        examples_n = len(data_set)
        prop_miss_values = data_set.isna().sum() / examples_n
        prop_miss_values = prop_miss_values.to_dict()

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(prop_miss_values)

    def skewness_mean(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the mean skewness of all features (columns) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the mean skewness of all features (columns) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """

        skew_features = skew(data_set.to_numpy())
        skew_mean = np.mean(skew_features)

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(skew_mean)

    def skewness_of_features(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the skewness for each feature (column) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the number of examples (rows) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """

        skew_features = {
            k: v for k, v in zip(data_set.columns, skew(data_set.to_numpy()))
        }

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(skew_features)

    def kurtosis_mean(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the mean kurtosis of all features (columns) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the mean kurtosis of all features (columns) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """

        kurtosis_features = kurtosis(data_set.to_numpy())
        kurtosis_mean = np.mean(kurtosis_features)

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(kurtosis_mean)

    def number_of_feature_correlations(
        self, data_set: pd.DataFrame, correlation_threshold=0.75
    ) -> None:
        """
        Count the number of feature pairs with a correlation greater than the
        specified threshold and store it in the metafeatures dictionary.

        Args:
            data_set (pd.DataFrame): Input pandas DataFrame.
            correlation_threshold (float, optional): Correlation threshold for
                counting feature pairs. Defaults to 0.75.
        """
        correlation_matrix = data_set.corr().abs()

        # Select the upper triangle of the correlation matrix
        upper_half = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # Find all features with a correlation > 0.75
        feature_correlation_n = len(
            [
                coloumn
                for coloumn in upper_half.columns
                if any(upper_half[coloumn] > correlation_threshold)
            ]
        )

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(feature_correlation_n)

    def kurtosis_of_features(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the kurtosis for each feature (column) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the kurtosis for each feature (column) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """

        kurtosis_features = {
            k: v for k, v in zip(data_set.columns, kurtosis(data_set.to_numpy()))
        }

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(kurtosis_features)

    def covariance(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the covariance matrix of the input DataFrame, and store the
        counts of positive, exact zero, and negative covariance values in the
        metafeatures dictionary.

        Args:
            data_set (pd.DataFrame): Input pandas DataFrame.
        """

        covariance_matrix = data_set.cov()
        upper_half = covariance_matrix.where(
            np.triu(np.ones(covariance_matrix.shape), k=1).astype(bool)
        )
        rounded_covariance = np.round(np.array(upper_half), decimals=3)

        number_of_positive_covariance = int(
            np.sum(np.sum(np.array(rounded_covariance) > 0, axis=0))
        )
        number_of_exact_zero_covariance = int(
            np.sum(np.sum(np.array(rounded_covariance) == 0, axis=0))
        )
        number_of_negative_covariance = int(
            np.sum(np.sum(np.array(rounded_covariance) < 0, axis=0))
        )

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(number_of_positive_covariance)
        self.metafeatures_dict[data_name].append(number_of_exact_zero_covariance)
        self.metafeatures_dict[data_name].append(number_of_negative_covariance)

    def entropy_mean(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the mean entropy of all features (columns) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the mean entropy of all features (columns) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """

        feature_entropies = np.array([])
        for feature in data_set.columns:
            useless_value, counts = np.unique(data_set[feature], return_counts=True)
            feature_entropy = entropy(counts)
            feature_entropies = np.append(feature_entropies, feature_entropy)
        entropy_mean = np.mean(feature_entropies)

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(entropy_mean)

    def entropies_of_features(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the entropy for each feature (column) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the entropy for each feature (column) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """

        feature_entropies = {}
        for feature in data_set.columns:
            useless_value, counts = np.unique(data_set[feature], return_counts=True)
            feature_entropy = entropy(counts)
            feature_entropies[feature] = feature_entropy

        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(feature_entropies)


def calculate_all_metics(path):
    metric = Metrics(path)
    metric.load_all_csv_datasets()
    for data in metric.data_sets_list:
        metric.number_of_features(data)
        metric.number_of_examples(data)
        #  This seems very pointless! All the data sets in use have no nan --> no information gain
        #  metric.proportion_of_missing_values(data)
        metric.skewness_mean(data)
        #  This produces a dict, we can not handle as parameter --> might still come in handy
        #  same with kurtosis and entropy
        #  metric.skewness_of_features(data)
        metric.kurtosis_mean(data)
        #  metric.entropies_of_features(data)
        metric.entropy_mean(data)
        metric.covariance(data)
        metric.number_of_feature_correlations(data)
    return metric


def cosine_sim_scipy():
    metric = calculate_all_metics("kp_test/datasets")
    x = np.array(metric.metafeatures_dict['Iris.csv'])
    y = np.array(metric.metafeatures_dict['ThinCross.csv'])
    #  [[0.99905276]]
    #  y = np.array(metric.metafeatures_dict['appendicitis.csv'])   [[0.98639771]]
    #  y = np.array(metric.metafeatures_dict['banana.csv'])         [[0.99897118]]
    #  y = np.array(metric.metafeatures_dict['wine_origin.csv'])    [[0.96245431]]
    #  y = np.array(metric.metafeatures_dict['tic_tac_toe.csv'])    [[0.99934275]]
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    print(f"Iris.csv is this one {x}")
    print(f"ThinCross.csv is this one {y}")
    print(f"This is there cosine-similarity: {1. - cdist(x, y, 'cosine')}")


def cosine_sim_sklearn():
    """
    I am sorry. I know this does not belong here.
    """
    metric = calculate_all_metics("kp_test/datasets")
    x = np.array(metric.metafeatures_dict['Iris.csv'])
    y = np.array(metric.metafeatures_dict['ThinCross.csv'])
    #  [[0.99905276]]
    #  y = np.array(metric.metafeatures_dict['appendicitis.csv'])  [[0.98639771]]
    #  y = np.array(metric.metafeatures_dict['banana.csv']) [[0.99897118]]
    #  y = np.array(metric.metafeatures_dict['wine_origin.csv']) [[0.96245431]]
    #  y = np.array(metric.metafeatures_dict['tic_tac_toe.csv'])  [[0.99934275]]
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    print(f"Iris.csv is this one {x}")
    print(f"ThinCross.csv is this one {y}")
    print(f"This is there cosine-similarity: {cosine_similarity(x, y)}")



if __name__ == "__main__":
    metric = calculate_all_metics("kp_test/datasets")
    print(metric.metafeatures_dict)
    cosine_sim_scipy()
