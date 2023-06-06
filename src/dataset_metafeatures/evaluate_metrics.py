import itertools
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from metrics import Metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, RobustScaler


class Evaluate_Metrics:
    """Evaluate_Metrics class for evaluating various metrics for datasets.

    Attributes:
        metric (Metrics): Metrics object for calculating various metrics.
    """

    def __init__(self, file_path) -> Metrics:
        """Initializes an Evaluate_Metrics object with Metrics object created with the given file_path.

        Args:
            file_path (str): Path to the datasets file.
        """
        self.metric = Metrics(file_path)

    def calculate_all_metrics(self) -> None:
        """Calculates all metrics for all datasets in data_frames_list of the Metrics object."""
        self.metric.load_all_csv_datasets()
        for data in self.metric.data_frames_list:
            self.metric.number_of_features(data)
            self.metric.number_of_examples(data)
            self.metric.examples_feature_ratio(data)
            self.metric.average_min(data)
            self.metric.median_min(data)
            self.metric.overall_mean(data)
            # metric.overall_median(data)
            self.metric.average_max(data)
            # metric.median_max(data)
            self.metric.standard_deviation_mean(data)
            # metric.standard_deviation_median(data)
            self.metric.variance_mean(data)
            # metric.variance_median(data)
            self.metric.quantile_mean(data)
            # metric.quantile_median(data)
            self.metric.skewness_mean(data)
            # metric.skewness_median(data)
            self.metric.kurtosis_mean(data)
            # metric.kurtosis_median(data)
            # metric.number_of_feature_correlations(data)
            self.metric.percentile(data)
            self.metric.coloumn_cosine_similarity_mean(data)
            self.metric.range_mean(data)
            self.metric.coefficient_variation_mean(data)
            self.metric.covariance(data)
            self.metric.entropy_mean(data)
            # metric.entropy_median(data)
            #  This seems very pointless! All the data sets in use have no nan --> no information gain
            # metric.proportion_of_missing_values(data)
            #  This produces a dict, we can not handle as parameter --> might still come in handy
            #  same with kurtosis and entropy
            #  metric.skewness_of_features(data)
            #  metric.entropies_of_features(data)

    def normalise_metrics_weights_robust_scaler(
        self, metafeatures: np.array
    ) -> np.array:
        """Normalizes the given metafeatures using a robust scaler.

        Args:
            metafeatures (np.array): The metafeatures to be normalized.

        Returns:
            np.array: The normalized metafeatures.
        """

        robust_scaler = RobustScaler()

        metafeatures_scaled = robust_scaler.fit_transform(metafeatures.reshape(-1, 1))

        return metafeatures_scaled.flatten()

    def normalise_metrics_weights_min_max_scaler(
        self, metafeatures: np.array
    ) -> np.array:
        """Normalizes the given metafeatures using a min-max scaler.

        Args:
            metafeatures (np.array): The metafeatures to be normalized.

        Returns:
            np.array: The normalized metafeatures.
        """

        min_max_scaler = MinMaxScaler()

        metafeatures_scaled = min_max_scaler.fit_transform(metafeatures.reshape(-1, 1))

        return metafeatures_scaled.flatten()

    def cosine_sim_scipy(self, data_set_a, data_set_b):
        """Calculates the cosine similarity between the two given datasets.

        Args:
            data_set_a: The first dataset.
            data_set_b: The second dataset.

        Returns:
            float: The cosine similarity between the two datasets.
        """

        x = self.metric.metafeatures_dict[data_set_a]
        y = self.metric.metafeatures_dict[data_set_b]

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        # print(f"{data_set_a} has this this normalized Vector: {x}")
        # print(f"{data_set_b}: {y}")

        return 1.0 - cdist(x, y, "cosine")

    def calculate_all_cosine_similarities(self) -> pd.DataFrame:
        """Calculates cosine similarities for all unique pairs of datasets in the given list.

        Args:
            data_sets_list (list[pd.DataFrame]): List of datasets for which cosine similarities are to be calculated.

        Returns:
            pd.DataFrame: DataFrame containing pairs of dataset names and their cosine similarity.
        """

        results = []

        for data_set_a, data_set_b in itertools.combinations(
            self.metric.data_frames_list, 2
        ):
            # Calculate cosine similarity only for unique pairs
            cos_sim = self.cosine_sim_scipy(data_set_a.name, data_set_b.name)

            # Explicit Tuple Notation [(data_set_a, data_set_b, cosine_similarity)]
            results.append((data_set_a.name, data_set_b.name, cos_sim[0][0]))

        df = pd.DataFrame(
            results, columns=["dataset_name_a", "dataset_name_b", "cosine_similarity"]
        )

        return df

    def difference_between_results(
        self, our_results: pd.DataFrame, other_results: pd.DataFrame
    ) -> pd.DataFrame:
        # Assume 'df1' and 'df2' are your DataFrames
        file1_data = dict(
            zip(
                zip(our_results["dataset_name_a"], our_results["dataset_name_b"]),
                our_results["cosine_similarity"],
            )
        )
        file2_data = dict(
            zip(
                zip(other_results["dataset_name_a"], other_results["dataset_name_b"]),
                other_results["cosine_similarity"],
            )
        )

        common_keys = set(file1_data.keys()).intersection(file2_data.keys())

        differences = {
            key: abs(file1_data[key] - file2_data[key]) for key in common_keys
        }

        # Convert the dictionary to a pandas DataFrame for better visualization
        df_diff = pd.DataFrame(
            list(differences.items()), columns=["Datasets", "cosine_similarity"]
        )

        # Split the tuple into two separate columns for 'dataset_name_a' and 'dataset_name_b'
        df_diff[["dataset_name_a", "dataset_name_b"]] = pd.DataFrame(
            df_diff["Datasets"].tolist(), index=df_diff.index
        )
        df_diff = df_diff.drop(columns="Datasets")

        # Reorder the columns
        df_diff = df_diff[["dataset_name_a", "dataset_name_b", "cosine_similarity"]]

        return df_diff

    def sort_dataframe(
        self, df: pd.DataFrame, columns: list[str], ascending=True
    ) -> pd.DataFrame:
        """Sorts the DataFrame by the given columns.

        Args:
            df (pd.DataFrame): The DataFrame to be sorted.
            columns (list): The list of columns to sort by. These can only be a combination of: ["dataset_name_a", "cosine_similarity"]
            ascending (bool): If True, sort in ascending order. If False, sort in descending order.

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """

        valid_columns = ["dataset_name_a", "cosine_similarity"]

        # Check if the provided columns are valid
        for column in columns:
            if column not in valid_columns:
                raise ValueError(
                    f"Invalid column: {column}. Valid options are {valid_columns}"
                )

        df_sorted = df.sort_values(by=columns, ascending=ascending)
        return df_sorted


def plot_cosine_distribution_graph(
    dataframes: list[pd.DataFrame], colors: list[str]
) -> None:
    """Plots a graph showing the distribution of cosine similarities for the given sorted results.

    Args:
        dataframes (List[pd.DataFrame]): The sorted results for which the graph is to be plotted.
    """
    if len(dataframes) != len(colors):
        raise ValueError(
            f"Number of dataframes and colors must be equal. Provided: dataframes = {len(dataframes)} & colors = {len(colors)}"
        )

    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 5))

    # Merge all dataframes into one for the purpose of getting unique dataset pairs
    merged_df = pd.concat(dataframes)

    # Create a new column combining 'dataset_name_a' and 'dataset_name_b'
    merged_df["dataset_pair"] = (
        merged_df["dataset_name_a"] + " - " + merged_df["dataset_name_b"]
    )

    # Get unique dataset pairs
    dataset_pairs = merged_df["dataset_pair"].unique()

    # Sort the merged dataframe by 'cosine_similarity' in descending order
    merged_df = merged_df.sort_values(by="cosine_similarity", ascending=False)

    # Reorder dataset_pairs according to 'cosine_similarity' in the sorted merged_df
    dataset_pairs = merged_df["dataset_pair"].unique()

    for i, df in enumerate(dataframes):
        df["dataset_pair"] = df["dataset_name_a"] + " - " + df["dataset_name_b"]

        # Set dataset pairs as categories for x-axis
        df["dataset_pair"] = pd.Categorical(
            df["dataset_pair"], categories=dataset_pairs
        )

        sns.scatterplot(
            x="dataset_pair", y="cosine_similarity", data=df, color=colors[i], alpha=0.5
        )

    plt.xlabel("Dataset Pairs")
    plt.xticks(rotation=90)
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity between Datasets")
    plt.ylim(-0.1, 1)
    plt.yticks(np.arange(-0.1, 1.05, 0.1))
    plt.legend(
        [f"DataFrame {i+1}" for i in range(len(dataframes))]
    )  # Add a legend to differentiate the points
    plt.show()


def filter_for_matching_pairs(
    df1: pd.DataFrame, df_to_be_filtered: pd.DataFrame
) -> pd.DataFrame:
    df_filtered = pd.merge(
        df1[["dataset_name_a", "dataset_name_b"]],
        df_to_be_filtered,
        on=["dataset_name_a", "dataset_name_b"],
    )

    return df_filtered


def main():
    evaluate_metrics = Evaluate_Metrics("kp_test/datasets")
    evaluate_metrics.calculate_all_metrics()

    df_cosine_similarities = evaluate_metrics.calculate_all_cosine_similarities()

    # Load the Results from the Master Thesis
    other_results_df = pd.read_csv(
        "./src/dataset_metafeatures/results/cos_sim_mastergroup.csv"
    )

    other_results_filtered = filter_for_matching_pairs(
        df1=df_cosine_similarities, df_to_be_filtered=other_results_df
    )

    # Calculate the Absolute Difference between our and their cosine_similarity
    df_diff = evaluate_metrics.difference_between_results(
        df_cosine_similarities, other_results_filtered
    )

    # Sort the Difference by name and cosine_similarity
    df_diff = evaluate_metrics.sort_dataframe(
        df_diff, ["dataset_name_a", "cosine_similarity"]
    )

    # # Plot the Differences in the Graph
    plot_cosine_distribution_graph(
        [df_cosine_similarities, other_results_filtered], ["blue", "orange"]
    )


if __name__ == "__main__":
    main()
