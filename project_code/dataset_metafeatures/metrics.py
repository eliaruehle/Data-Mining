import itertools
import os
import re
from abc import ABC
from math import log2, log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import entropy, kurtosis, skew


class Metrics(ABC):
    """
    A class to represent a Metrics object that can load CSV datasets.

    Attributes
    ----------
    data_sets : list[str]
        A sorted list of CSV file names in the specified directory.
    data_sets_list : list[pd.DataFrame]
        A list to store loaded CSV datasets.
    metafeatures_dict: dict[str, list[float]]
        A dictionary which stores all the existing datasets with their corresponding metafeatures.
        The key being the name of the dataset (ending with .csv), while the value is a list containing
        all metafeatures.

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
        self.metafeatures_dict: dict[str, list[float]] = dict()

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

    def add_to_meatafeatures_dict(
        self, data_set: pd.DataFrame, metafeature: float
    ) -> None:
        """
        Add the calculated metafeature to the metafeatures_dict.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the metafeature has been calculated.
            metafeature (float): The calculated metafeature value.
        """
        data_name = data_set.name
        if data_name not in self.metafeatures_dict:
            self.metafeatures_dict[data_name] = []
        self.metafeatures_dict[data_name].append(metafeature)

    def number_of_features(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the number of features in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the number of features (columns) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of features needs to be calculated.

        """
        features_n = log2(len(data_set.columns))

        self.add_to_meatafeatures_dict(data_set, features_n)

    def number_of_examples(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the number of examples (rows) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the number of examples (rows) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """
        examples_n = log10(len(data_set))

        self.add_to_meatafeatures_dict(data_set, examples_n)

    def examples_feature_ratio(self, data_set: pd.DataFrame) -> None:
        features = len(data_set.columns)
        examples = len(data_set)

        feature_examples_ratio = features / examples

        self.add_to_meatafeatures_dict(data_set, feature_examples_ratio)

    def overall_mean(self, data_set: pd.DataFrame) -> None:
        overall_mean = data_set.mean().mean()

        self.add_to_meatafeatures_dict(data_set, overall_mean)

    def overall_median(self, data_set: pd.DataFrame) -> None:
        overall_median = data_set.median().median()

        self.add_to_meatafeatures_dict(data_set, overall_median)

    def average_min(self, data_set: pd.DataFrame) -> None:
        average_min = data_set.min().mean()

        self.add_to_meatafeatures_dict(data_set, average_min)

    def median_min(self, data_set: pd.DataFrame) -> None:
        median_min = data_set.min().median()

        self.add_to_meatafeatures_dict(data_set, median_min)

    def average_max(self, data_set: pd.DataFrame) -> None:
        average_max = data_set.max().mean()

        self.add_to_meatafeatures_dict(data_set, average_max)

    def median_max(self, data_set: pd.DataFrame) -> None:
        median_max = data_set.max().median()

        self.add_to_meatafeatures_dict(data_set, median_max)

    def standard_deviation_mean(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the mean of the standard deviation for each column in the data_set and add it to the metafeatures dictionary.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the mean of the standard deviation is calculated.

        """
        standard_deviation_mean = data_set.std().mean()

        self.add_to_meatafeatures_dict(data_set, standard_deviation_mean)

    def standard_deviation_median(self, data_set: pd.DataFrame) -> None:
        standard_deviation_median = data_set.std().median()

        self.add_to_meatafeatures_dict(data_set, standard_deviation_median)

    def variance_mean(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the mean of the variance for each column in the data_set and add it to the metafeatures dictionary.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the mean of the variance is calculated.

        """
        variance_mean = data_set.var().mean()

        self.add_to_meatafeatures_dict(data_set, variance_mean)

    def variance_median(self, data_set: pd.DataFrame) -> None:
        variance_median = data_set.var().median()

        self.add_to_meatafeatures_dict(data_set, variance_median)

    def quantile_mean(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the mean of the quantiles for each column in the data_set and add it to the metafeatures dictionary.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the mean of the quantiles is calculated.

        """
        quantile_mean = data_set.quantile().mean()

        self.add_to_meatafeatures_dict(data_set, quantile_mean)

    def quantile_median(self, data_set: pd.DataFrame) -> None:
        quantile_median = data_set.quantile().median()

        self.add_to_meatafeatures_dict(data_set, quantile_median)

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
        examples_n = log2(len(data_set))
        prop_miss_values = data_set.isna().sum() / examples_n
        prop_miss_values = prop_miss_values.to_dict()

        self.add_to_meatafeatures_dict(data_set, prop_miss_values)

    def skewness_mean(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the mean skewness of all features (columns) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the mean skewness of all features (columns) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """

        skew_mean = data_set.skew().mean()

        self.add_to_meatafeatures_dict(data_set, skew_mean)

    def skewness_median(self, data_set: pd.DataFrame) -> None:
        skew_median = data_set.skew().median()

        self.add_to_meatafeatures_dict(data_set, skew_median)

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

        self.add_to_meatafeatures_dict(data_set, skew_features)

    def kurtosis_mean(self, data_set: pd.DataFrame) -> None:
        """
        Calculate the mean kurtosis of all features (columns) in a given DataFrame and store it in the metafeatures_dict.

        This method calculates the mean kurtosis of all features (columns) in the input DataFrame and appends the value
        to the list associated with the DataFrame's name in the metafeatures_dict. If the DataFrame's name is not
        present in the metafeatures_dict, a new list is initialized for the key.

        Args:
            data_set (pd.DataFrame): The input DataFrame for which the number of examples needs to be calculated.

        """

        kurtosis_mean = data_set.kurt().mean()

        self.add_to_meatafeatures_dict(data_set, kurtosis_mean)

    def kurtosis_median(self, data_set: pd.DataFrame) -> None:
        kurtosis_median = data_set.kurt().median()

        self.add_to_meatafeatures_dict(data_set, kurtosis_median)

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

        self.add_to_meatafeatures_dict(data_set, kurtosis_features)

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

        self.add_to_meatafeatures_dict(data_set, feature_correlation_n)

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

        self.add_to_meatafeatures_dict(data_set, number_of_positive_covariance)
        self.add_to_meatafeatures_dict(data_set, number_of_exact_zero_covariance)
        self.add_to_meatafeatures_dict(data_set, number_of_negative_covariance)

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
            _, counts = np.unique(data_set[feature], return_counts=True)
            feature_entropy = entropy(counts)
            feature_entropies = np.append(feature_entropies, feature_entropy)
        entropy_mean = np.mean(feature_entropies)

        self.add_to_meatafeatures_dict(data_set, entropy_mean)

    def entropy_median(self, data_set: pd.DataFrame) -> None:
        feature_entropies = np.array([])
        for feature in data_set.columns:
            _, counts = np.unique(data_set[feature], return_counts=True)
            feature_entropy = entropy(counts)
            feature_entropies = np.append(feature_entropies, feature_entropy)

        entropy_median = np.mean(feature_entropies)

        self.add_to_meatafeatures_dict(data_set, entropy_median)

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

        self.add_to_meatafeatures_dict(data_set, feature_entropies)


def calculate_all_metrics(path) -> Metrics:
    metric = Metrics(path)
    metric.load_all_csv_datasets()
    for data in metric.data_sets_list:
        metric.number_of_features(data)
        metric.number_of_examples(data)
        metric.average_min(data)
        metric.overall_mean(data)
        metric.average_max(data)
        metric.standard_deviation_mean(data)
        metric.variance_mean(data)
        metric.quantile_mean(data)
        metric.skewness_mean(data)
        metric.kurtosis_mean(data)
        metric.number_of_feature_correlations(data)
        metric.covariance(data)
        metric.entropy_mean(data)
        #  This seems very pointless! All the data sets in use have no nan --> no information gain
        # metric.proportion_of_missing_values(data)
        #  This produces a dict, we can not handle as parameter --> might still come in handy
        #  same with kurtosis and entropy
        #  metric.skewness_of_features(data)
        #  metric.entropies_of_features(data)
    return metric


def cosine_sim_scipy(data_set_a, data_set_b):
    x = np.array(metric.metafeatures_dict[data_set_a])
    y = np.array(metric.metafeatures_dict[data_set_b])

    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    # print(f"{data_set_a} has this this normalized Vector: {x}")
    # print(f"{data_set_b}: {y}")

    return 1.0 - cdist(x, y, "cosine")


def extract_cosine_similarity(f_string: str) -> float:
    numbers = re.findall(r"[-+]?[\d]+(?:\.\d+)?(?:[eE][-+]?\d+)?", f_string)
    return float(numbers[-1]) if numbers else 0


def calculate_all_cosine_similarities(data_sets: list[pd.DataFrame]) -> list[str]:
    results = []

    for data_set_a, data_set_b in itertools.combinations(metric.data_sets_list, 2):
        # Calculate cosine similarity only for unique pairs
        cos_sim = cosine_sim_scipy(data_set_a.name, data_set_b.name)
        results.append((f"{data_set_a.name} - {data_set_b.name}", cos_sim[0][0]))

    return results


def sort_results(results: list[str]) -> list[str]:
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results


def write_results_into_file(sorted_f_strings: list[str]) -> None:
    with open("metrics.txt", "w") as file:
        for item in sorted_f_strings:
            file.write(f"{item[0]}: {item[1]}\n")


def plot_cosine_distribution_graph(sorted_f_strings: list[str]) -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 5))

    # Create a scatter plot with single points
    dataset_pairs, cos_sim_values = zip(*sorted_f_strings)
    sns.scatterplot(x=dataset_pairs, y=cos_sim_values, color="blue", alpha=0.5)

    plt.xlabel("Dataset Pairs")
    plt.xticks(rotation=90)
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity between Datasets")
    plt.ylim(-0.1, 1)

    plt.yticks(np.arange(-0.1, 1.05, 0.1))
    plt.show()


if __name__ == "__main__":
    metric = calculate_all_metrics("kp_test/datasets")
    datasets_cosine_similarities = calculate_all_cosine_similarities(
        metric.data_sets_list
    )
    sorted_f_strings = sort_results(datasets_cosine_similarities)
    # print(sorted_f_strings)
    write_results_into_file(sorted_f_strings)
    plot_cosine_distribution_graph(sorted_f_strings)
