import itertools
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from matplotlib.lines import Line2D
from strategy_recommendation.dataset_metafeatures.metrics import Metrics
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler


class Evaluate_Metrics:
    """A class for evaluating various metrics for datasets.

    This class uses the RobustScaler and MinMaxScaler preprocessing techniques from the sklearn library to normalize data.
    After normalization, it provides the ability to evaluate different metrics using a Metrics object.

    Attributes:
        robust_scaler (RobustScaler): Scaler object to scale features using statistics that are robust to outliers. This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range).
        min_max_scaler (MinMaxScaler): Transforms features by scaling each feature to a given range. This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g., between zero and one.
        reduced_metafeatures_dict (dict[str, np.array]): Dictionary that contains the metafeatures that have been reduced through Principal Component Analysis. The keys represent the names of the metafeatures and the values are numpy arrays containing the values of the reduced metafeatures.
    """

    def __init__(self, file_path: str):
        """Initializes an Evaluate_Metrics object with Metrics object created with the given file_path.

        Args:
            file_path (str): Path to the datasets file.
        """
        self.metric = Metrics(file_path)
        self.reduced_metafeatures_dict: dict[str, np.array] = {}
        self.robust_scaler = RobustScaler()
        self.min_max_scaler = MinMaxScaler()

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
            self.metric.total_geometric_mean(data)
            self.metric.total_harmonic_mean(data)
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

        metafeatures_scaled = self.robust_scaler.fit_transform(
            metafeatures.reshape(-1, 1)
        )

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

        metafeatures_scaled = self.min_max_scaler.fit_transform(
            metafeatures.reshape(-1, 1)
        )

        return metafeatures_scaled.flatten()

    def calculate_normalisation_for_all_metafeatures(
        self, metafeatures: dict[str, np.array], normalisation_method
    ) -> None:
        """Calculate the normalisation for all metafeatures.

        Args:
            metafeatures (dict[str, np.array]): Dictionary of metafeatures, where the keys
                are the names and the values are NumPy arrays representing the metafeatures.
            normalisation_method: The normalisation method to be applied to each metafeature.

        Returns:
            dict[str, np.array]: Dictionary of normalised metafeatures, where the keys
                are the names and the values are the normalised metafeature arrays.
        """

        normalised_dict = {}

        for key, value in metafeatures.items():
            normalised_dict[key] = normalisation_method(value)

        return normalised_dict

    def principal_component_analysis(
        self, normalised_metafeatures: dict[str, np.array], n_components: int = 8
    ) -> dict[str, np.array]:
        """Perform Principal Component Analysis (PCA) on normalised metafeatures.

        Args:
            normalised_metafeatures (dict[str, np.array]): Dictionary of normalised metafeatures,
                where the keys are the names and the values are NumPy arrays representing
                the normalised metafeatures.
            n_components (int, optional): The number of dimensions for the reduced data. Defaults to 8.

        Returns:
            dict[str, np.array]: Dictionary of reduced metafeatures obtained through PCA,
                where the keys are the names and the values are the reduced metafeature arrays.
        """
        X = np.vstack(list(normalised_metafeatures.values()))

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        pca_dict = {
            name: vec for name, vec in zip(self.metric.metafeatures_dict.keys(), X_pca)
        }

        self.reduced_metafeatures_dict["pca"] = pca_dict

    def t_distributed_stochastic_neighbor_embedding(
        self,
        normalised_metafeatures: dict[str, np.array],
        method: str = "barnes_hut",
        n_components: int = 2,
    ) -> dict[str, np.array]:
        """Perform t-SNE on normalised metafeatures.

        Args:
            normalised_metafeatures (dict[str, np.array]): Dictionary of normalised metafeatures,
                where the keys are the names and the values are NumPy arrays representing
                the normalised metafeatures.
            method (str, optional): The method to use for t-SNE. Options are 'barnes_hut' or 'exact'.
                Defaults to 'barnes_hut'.
            n_components (int, optional): The number of dimensions for the reduced data.
                Defaults to 2 for compatibility with the 'barnes_hut' method.

        Raises:
            ValueError: If a method is not recognised.

        Returns:
            dict[str, np.array]: Dictionary of reduced metafeatures obtained through t-SNE,
                where the keys are the names and the values are the reduced metafeature arrays.
        """
        if method not in ["barnes_hut", "exact"]:
            raise ValueError(
                f"Invalid method: {method}. Valid options are {['barnes_hut', 'exact']}"
            )

        X = np.vstack(list(normalised_metafeatures.values()))

        tsne = TSNE(n_components=n_components, method=method)
        X_tsne = tsne.fit_transform(X)

        tsne_dict = {
            name: vec for name, vec in zip(self.metric.metafeatures_dict.keys(), X_tsne)
        }

        self.reduced_metafeatures_dict["tsne"] = tsne_dict

    def uniform_manifold_approximation_and_projection(
        self, normalised_metafeatures: dict[str, np.array], n_components: int = 8
    ) -> dict[str, np.array]:
        """Perform UMAP on normalised metafeatures.

        Args:
            normalised_metafeatures (dict[str, np.array]): Dictionary of normalised metafeatures,
                where the keys are the names and the values are NumPy arrays representing
                the normalised metafeatures.
            n_components (int, optional): The number of dimensions for the reduced data. Defaults to 8.

        Returns:
            dict[str, np.array]: Dictionary of reduced metafeatures obtained through UMAP,
                where the keys are the names and the values are the reduced metafeature arrays.
        """

        X = np.vstack(list(normalised_metafeatures.values()))

        reducer = umap.UMAP(n_components=8)
        X_umap = reducer.fit_transform(X)

        umap_dict = {
            name: vec for name, vec in zip(self.metric.metafeatures_dict.keys(), X_umap)
        }

        self.reduced_metafeatures_dict["umap"] = umap_dict

    def visualize(self, n_components: int):
        methods = ["tsne", "umap"]
        le = LabelEncoder()

        for i, method in enumerate(methods):
            if method not in self.reduced_metafeatures_dict:
                print(f"No reduced data for method {method}, skipping plot.")
                continue

            labels = list(self.reduced_metafeatures_dict[method].keys())
            encoded_labels = le.fit_transform(
                labels
            )  # Convert labels to integers for coloring

            if n_components == 2:
                data = {
                    "x": [
                        vec[0]
                        for vec in self.reduced_metafeatures_dict[method].values()
                    ],
                    "y": [
                        vec[1]
                        for vec in self.reduced_metafeatures_dict[method].values()
                    ],
                    "label": encoded_labels,
                }
                df = pd.DataFrame(data)
                # Create a dictionary that maps encoded labels to original labels
                label_dict = dict(zip(encoded_labels, labels))
                # Create a new column in the dataframe that contains the original labels
                df["label_name"] = df["label"].map(label_dict)

                plt.figure(figsize=(10, 7))
                # Use the new column for the 'hue' argument
                sns.scatterplot(data=df, x="x", y="y", hue="label_name", palette="husl")
                plt.title(f"{method} 2D scatter plot")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                plt.show()

            elif n_components == 3:
                data = {
                    "x": [
                        vec[0]
                        for vec in self.reduced_metafeatures_dict[method].values()
                    ],
                    "y": [
                        vec[1]
                        for vec in self.reduced_metafeatures_dict[method].values()
                    ],
                    "z": [
                        vec[2]
                        for vec in self.reduced_metafeatures_dict[method].values()
                    ],
                    "label": encoded_labels,
                }
                df = pd.DataFrame(data)

                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection="3d")
                scatter = ax.scatter(
                    df["x"], df["y"], df["z"], c=df["label"], cmap="viridis", alpha=1
                )
                plt.title(f"{method} 3D scatter plot")

                # Create a custom legend
                legend_labels = le.inverse_transform(list(set(encoded_labels)))
                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=label,
                        markerfacecolor=plt.cm.viridis(i / len(legend_labels)),
                        markersize=10,
                    )
                    for i, label in enumerate(legend_labels)
                ]

                ax.legend(
                    handles=legend_elements,
                    bbox_to_anchor=(1.05, 1),
                    loc=2,
                    borderaxespad=0.0,
                )

                plt.show()

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

        return 1.0 - cdist(x, y, "cosine")

    def calculate_all_cosine_similarities(self) -> pd.DataFrame:
        """Calculates cosine similarities for all unique pairs of datasets in the given list.

        Args:
            data_sets_list (list[pd.DataFrame]): List of datasets for which cosine similarities
                are to be calculated.

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
        # Assume 'our_results' and 'other_results' are DataFrames
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

    def generate_evaluations(
        self,
        normalisation: bool,
        dimension_reductions: List[List[Union[str, Dict[str, Union[int, str]]]]],
        visualize: bool = False,
    ):
        """
        Generate evaluations based on various dimension reduction methods.

        Args:
            normalisation (bool): Whether to perform normalisation on the metrics before applying dimension reduction.
            dimension_reductions (List[List[Union[str, Dict[str, Union[int, str]]]]]): List of dimension reduction methods to be applied.
                Each method is represented as a list, where the first element is the method name and the second element is a dictionary of parameters for the method.
            visualize (bool, optional): Whether to visualize the reduced dimension data.

        Raises:
            TypeError: If `normalisation` is not a boolean.
            ValueError: If a dimension reduction method is not recognised.

        """
        if not isinstance(normalisation, bool):
            raise TypeError(
                f"`normalision` must be of type `bool`, not {type(normalisation)}"
            )

        if not isinstance(visualize, bool):
            raise TypeError(
                f"`visualize` must be of type `bool`, not {type(visualize)}"
            )

        self.calculate_all_metrics()

        if normalisation:
            normalised_metafeatures = self.calculate_normalisation_for_all_metafeatures(
                metafeatures=self.metric.metafeatures_dict,
                normalisation_method=self.normalise_metrics_weights_robust_scaler,
            )

        if normalisation and dimension_reductions:
            # Mapping dimension reduction methods to their corresponding functions
            dimension_reduction_methods = {
                "pca": self.principal_component_analysis,
                "tsne": self.t_distributed_stochastic_neighbor_embedding,
                "umap": self.uniform_manifold_approximation_and_projection,
            }

            for tup in dimension_reductions:
                method = tup[0]
                parameters = tup[1]

                if method not in dimension_reduction_methods:
                    raise ValueError(
                        f"Unrecognised dimension reduction method: {method}. Please use one of the following: {list(dimension_reduction_methods.keys())}."
                    )

                # Call the corresponding function
                dimension_reduction_methods[method](
                    normalised_metafeatures, **parameters
                )

        if visualize:
            n_components = min(
                [tup[1].get("n_components", 2) for tup in dimension_reductions]
            )
            if n_components > 3:
                print(
                    f"Visualization not supported for n_components > 3. Skipping visualization."
                )
            else:
                self.visualize(n_components)


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

    # Add a legend to differentiate the points
    plt.legend([f"DataFrame {i+1}" for i in range(len(dataframes))])
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
    # main()
    eval_metrics = Evaluate_Metrics(file_path="kp_test/datasets")

    eval_metrics.generate_evaluations(
        normalisation=True,
        dimension_reductions=[
            ["pca", {"n_components": 8}],
            ["tsne", {"n_components": 2}],
            ["umap", {"n_components": 2}],
        ],
        visualize=True,
    )
