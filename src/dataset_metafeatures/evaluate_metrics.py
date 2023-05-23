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

    def extract_cosine_similarity(self, f_string: str) -> float:
        """Extracts the cosine similarity from the given formatted string.

        Args:
            f_string (str): The formatted string containing the cosine similarity.

        Returns:
            float: The cosine similarity.
        """

        numbers = re.findall(r"[-+]?[\d]+(?:\.\d+)?(?:[eE][-+]?\d+)?", f_string)
        return float(numbers[-1]) if numbers else 0

    def calculate_all_cosine_similarities(
        self, data_sets_list: list[pd.DataFrame]
    ) -> list[str]:
        """Calculates cosine similarities for all unique pairs of datasets in the given list.

        Args:
            data_sets_list (list[pd.DataFrame]): List of datasets for which cosine similarities are to be calculated.

        Returns:
            list[str]: List of tuples containing pairs of dataset names and their cosine similarity.
        """

        results = []

        for data_set_a, data_set_b in itertools.combinations(
            self.metric.data_frames_list, 2
        ):
            # Calculate cosine similarity only for unique pairs
            cos_sim = self.cosine_sim_scipy(data_set_a.name, data_set_b.name)
            results.append((f"{data_set_a.name},{data_set_b.name}", cos_sim[0][0]))

        return results

    def sort_results(self, results: list[str]) -> list[str]:
        """Sorts the given list of results.

        Args:
            results (list[str]): The results to be sorted.

        Returns:
            list[str]: The sorted results.
        """

        sorted_results = sorted(results, key=lambda x: x[0].lower(), reverse=False)
        return sorted_results

    def write_results_into_file(self, sorted_f_strings: list[str]) -> None:
        """Writes the given sorted results into a file.

        Args:
            sorted_f_strings (list[str]): The sorted results to be written into the file.
        """

        with open(
            "./src/dataset_metafeatures/results/unsere_ergebnisse.txt", "w+"
        ) as file:
            file.write("dataset_name_a,dataset_name_b,cosine_similarity\n")
            for item in sorted_f_strings:
                file.write(f"{item[0]},{item[1]}\n")

    def plot_cosine_distribution_graph(self, sorted_f_strings: list[str]) -> None:
        """Plots a graph showing the distribution of cosine similarities for the given sorted results.

        Args:
            sorted_f_strings (list[str]): The sorted results for which the graph is to be plotted.
        """

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


def main():
    evaluate_metrics = Evaluate_Metrics("kp_test/datasets")
    evaluate_metrics.calculate_all_metrics()

    datasets_cosine_similarities = evaluate_metrics.calculate_all_cosine_similarities(
        evaluate_metrics.metric.data_frames_list
    )

    sorted_f_strings = evaluate_metrics.sort_results(datasets_cosine_similarities)
    print(sorted_f_strings)
    evaluate_metrics.write_results_into_file(sorted_f_strings)
    # plot_cosine_distribution_graph(sorted_f_strings)

    results = pd.read_csv("./src/dataset_metafeatures/cosine_sim_results.csv")
    results = results.dropna(subset=["cosine_similarity"])

    filtered_results = results[
        results["dataset_name_a"].isin(evaluate_metrics.metric.data_sets_list)
        & results["dataset_name_b"].isin(evaluate_metrics.metric.data_sets_list)
    ]

    filtered_results.to_csv(
        "./src/dataset_metafeatures/results/master_ergebnisse.txt",
        sep=",",
        index=False,
    )


if __name__ == "__main__":
    main()
