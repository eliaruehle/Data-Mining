from collections import defaultdict

import numpy as np
import pandas as pd
import json
import os as os

from datasets.loader import Loader
from typing import List, Tuple, Dict


class TopK:

    def __init__(self, loader_directory: str):
        self.unwanted = []  # List of weirdly formatted files, e.g. files that contain lists. Only here for debugging
        self.blacklisted_words = [
            # Metrics that are the first derivative
            'lag', 'selected_indices', 'auc',

            # Metrics that contain lists
            'CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS', 'CLOSENESS_TO_CLUSTER_CENTER',
            'CLOSENESS_TO_SAMPLES_OF_SAME_CLASS', 'CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN',
            'IMPROVES_ACCURACY_BY', 'COUNT_WRONG_CLASSIFICATIONS',
            'CLOSENESS_TO_DECISION_BOUNDARY', 'class_distributions_chebyshev_batch',
            'AVERAGE_UNCERTAINTY', 'OUTLIERNESS', 'class_distributions_manhattan_batch',
            'CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN', 'MELTING_POT_REGION', 'REGION_DENSITY',
            'SWITCHES_CLASS_OFTEN', 'y_pred_train', 'y_pred_test',

            # Metrics that contain negative or only non-monotonic values
            'class_distribution_chebyshev_added_up', 'class_distribution_manhattan_added_up',
            'avg_dist_batch', 'avg_dist_labeled']

        self.considered_metric = [string for string in Loader.list_metrics(base_dir=loader_directory) if
                                  not any(word.lower() in string.lower() for word in self.blacklisted_words)]

        self.data = Loader(base_dir=loader_directory, wanted_metrics=self.considered_metric)

    # Calculate for generally the best AL strategy for a given metric and save the result
    def calculate_best_strategy_for_metric(self, directory: str):

        # Create directory 'metric_batch_size' if it doesn't exist
        subdirectory = f"{directory}/metric_batch_size"
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        for metric in self.considered_metric:
            for batch_size in [1, 5, 10]:
                best_al_strats = {}
                for dataset in self.data.get_dataset_names():
                    file_name = f"{directory}/dataset_batch_size/{dataset}_{batch_size}.json"

                    with open(file_name, 'r') as file:
                        data_dict = json.load(file)

                    entries = data_dict.get(metric, [])
                    for index in range(len(entries)):
                        best_al_strats[entries[index]] = best_al_strats.get(entries[index], 0) + 1 / (index + 1)

                sorted_best_al_strats = dict(sorted(best_al_strats.items(), key=lambda x: x[1], reverse=True))

                total_sum = sum(sorted_best_al_strats.values())
                percentages = {key: (value / total_sum) for key, value in sorted_best_al_strats.items()}

                destination = f"{directory}/metric_batch_size/{metric}_{batch_size}.json"
                with open(destination, 'w') as f:
                    json.dump(percentages, f)

    # Calculate which AL strategy gives the best result over all datasets and metrics. The result is normalized to
    # individual_score / sum(all_scores) and saved as a JSON file
    def calculate_generally_best_strategy(self, directory: str):
        dicts = []

        for dataset in self.data.get_dataset_names():
            for batch_size in [1, 5, 10]:
                file_name = f"{directory}/best_strategy_for/best_strategy_for_{dataset}_{batch_size}.json"

                with open(file_name, 'r') as file:
                    best_strategies: dict[str: list[str]] = json.load(file)
                    dicts.append(best_strategies)

        result_dict = {}

        for entry in dicts:
            for index, key in enumerate(entry.keys(), start=1):
                result_dict[key] = result_dict.get(key, 0) + 1 / index

        sorted_result_dict = dict(sorted(result_dict.items(), key=lambda x: x[1], reverse=True))

        total_sum = sum(sorted_result_dict.values())
        percentages = {key: (value / total_sum) for key, value in sorted_result_dict.items()}

        destination = f"{directory}/overall_best.json"
        with open(destination, 'w') as f:
            json.dump(percentages, f)

    # Do calculations of best_al_strategy(...) for all datasets and save the results
    def collect_best_strategy_for_dataset(self, directory: str):

        # Create directory 'best_strategy_for' if it doesn't exist
        subdirectory = f"{directory}/best_strategy_for"
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        for dataset in self.data.get_dataset_names():
            for batch_size in [1, 5, 10]:
                result_dict = self.best_al_strategy(dataset=dataset, batch_size=batch_size, directory=directory)

                file_name = f"{directory}/best_strategy_for/best_strategy_for_{dataset}_{batch_size}.json"
                with open(file_name, 'w') as f:
                    json.dump(result_dict, f)

    # Order the AL strategies by how good they generally apply to all the metrics of a given dataset. The more often
    # an AL strategy performs well for a metric, the higher its score is. At the end, all scores are normalized to
    # individual_score / sum(all_scores)
    @staticmethod
    def best_al_strategy(dataset: str, batch_size: int, directory: str):
        file_name = f"{directory}/dataset_batch_size/{dataset}_{batch_size}.json"
        with open(file_name, 'r') as file:
            top_k_data: dict[str: list[str]] = json.load(file)

        result_dict = {}
        for key, value_list in top_k_data.items():
            for index in range(len(value_list)):
                result_dict[value_list[index]] = result_dict.get(value_list[index], 0) + 1 / (index + 1)

        sorted_result_dict = dict(sorted(result_dict.items(), key=lambda x: x[1], reverse=True))

        total_sum = sum(sorted_result_dict.values())
        percentages = {key: (value / total_sum) for key, value in sorted_result_dict.items()}

        return percentages

    # For a given dataset and metric, return an ordered list of AL strategies, representing its goodness
    def get_top_k(self, dataset: str, metric: str, batch_size: int, k: int = 500, threshold: float = 1,
                  max_iterations: int = 50, epsilon: float = 0):
        # Load data
        time_series_list: List[np.ndarray] = []
        strategy_names: List[str] = []

        for strategy in self.data.get_strategy_names():
            frame: pd.DataFrame = self.data.get_single_dataframe(strategy, dataset, metric)
            vector: pd.DataFrame = frame.loc[frame["EXP_BATCH_SIZE"] == batch_size]
            vector = vector.iloc[:, :-9].dropna(axis=1)

            if not vector.empty:
                time_series_list.extend(vector.to_numpy())
                strategy_names.extend([strategy] * len(vector))

        time_series = [list(arr) for arr in time_series_list]

        def is_monotonic_increasing(series, j):
            for i in range(1, len(series)):
                if series[i] < series[i - 1] - j:
                    return False
            return True

        def check_floats_or_integers(series):
            for element in series:
                if not isinstance(element, (float, int)):
                    return False
            return True

        def reaches_threshold(series):
            if threshold == -1:
                return True
            for i in range(len(series)):
                if i < (max_iterations - 1) and series[i] >= threshold:
                    return True
            return False

        def calculate_score(series):
            score = sum(series)  # Use sum() to calculate the score
            return score

        valid_series = []
        for series, strategy in zip(time_series, strategy_names):
            if not check_floats_or_integers(series):
                self.unwanted.append(metric)

            if check_floats_or_integers(series) and is_monotonic_increasing(series, epsilon) and reaches_threshold(
                    series):
                valid_series.append((strategy, calculate_score(series)))

        valid_series = sorted(valid_series, key=lambda x: x[1], reverse=True)  # Sort by score

        if len(valid_series) <= k:
            top_k_series = valid_series
        else:
            top_k_series = valid_series[:k]

        average = defaultdict(int)
        occurrences = defaultdict(int)

        for dataset, score in top_k_series:
            occurrences[dataset] += 1
            average[dataset] += score

        for dataset in average.keys():
            average[dataset] /= occurrences[dataset]

        average_list = [(dataset, score) for dataset, score in average.items()]

        sorted_average_list = sorted(average_list, key=lambda x: x[1], reverse=True)

        return sorted_average_list

    # Do calculations of get_top_k(...) for all datasets and save the results
    def collect_top_k(self, directory: str, k: int = 500, threshold: float = 1, epsilon: float = 0):

        # Create directory 'dataset_batch_size' if it doesn't exist
        subdirectory = f"{directory}/dataset_batch_size"
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        for dataset in self.data.get_dataset_names():
            for batch_size in [1, 5, 10]:
                result_dict = {}
                for metric in self.considered_metric:
                    result_dict[metric] = self.get_top_k(dataset, metric, batch_size=batch_size, k=k,
                                                         threshold=threshold, epsilon=epsilon)

                file_name = f"{directory}/dataset_batch_size/{dataset}_{batch_size}.json"
                with open(file_name, 'w') as f:
                    json.dump(result_dict, f)

    def collect_average_score(self, directory: str):
        result_dict = {}
        for dataset in self.data.get_dataset_names():
            for batch_size in [1, 5, 10]:
                # Initialize result_dict[batch_size] as a dictionary if it doesn't exist
                if batch_size not in result_dict:
                    result_dict[batch_size] = {}

                file_name = f"{directory}/dataset_batch_size/{dataset}_{batch_size}.json"
                with open(file_name, 'r') as file:
                    top_k_data: Dict[str, List[Tuple[str, int]]] = json.load(file)

                for strategy in top_k_data.keys():
                    # Initialize result_dict[batch_size][dataset] as a dictionary if it doesn't exist
                    if dataset not in result_dict[batch_size]:
                        result_dict[batch_size][dataset] = {}

                    result_dict[batch_size][dataset][strategy] = top_k_data[strategy]

        file_name = f"{directory}/average_performance.json"
        with open(file_name, 'w') as f:
            json.dump(result_dict, f)

    def gen_performance_json(self, directory: str):
        # 1. For each strategy, batch_size and metric, find datasets they perform on the best

        values = {}

        for dataset in self.data.get_dataset_names():
            # Add another dimension
            if dataset not in values:
                values[dataset] = {}

            for batch_size in [1, 5, 10]:
                # Add another dimension
                if batch_size not in values[dataset]:
                    values[dataset][batch_size] = {}

                # Read CSV file
                file_name = f"{directory}/dataset_batch_size/{dataset}_{batch_size}.json"
                with open(file_name, 'r') as file:
                    data_dict: dict[str: list[str]] = json.load(file)

                for metric in list(data_dict.keys()):
                    # Add another dimension
                    if metric not in values[dataset][batch_size]:
                        values[dataset][batch_size][metric] = {}

                    # Iterate over the AL strategies with their scores
                    for index in range(len(data_dict[metric])):
                        # Each dataset receives a score for a metric, batch_size and strategy
                        values[dataset][batch_size][metric][data_dict[metric][index][0]] = \
                            values[dataset][batch_size][metric].get(data_dict[metric][index][0], 0) + \
                            data_dict[metric][index][1]

        # Sort values: For a given batch-size, metric and strategy, what are some good datasets to use it on? (Good is
        # relative as there might be other combinations of batch-size, metric and strategy that perform even better on
        # those datasets)

        sorted_datasets = {}

        for dataset in values:
            for batch_size in values[dataset]:
                for metric in values[dataset][batch_size]:
                    for strategy in values[dataset][batch_size][metric]:
                        score = values[dataset][batch_size][metric][strategy]

                        if batch_size not in sorted_datasets:
                            sorted_datasets[batch_size] = {}
                        if metric not in sorted_datasets[batch_size]:
                            sorted_datasets[batch_size][metric] = {}
                        if strategy not in sorted_datasets[batch_size][metric]:
                            sorted_datasets[batch_size][metric][strategy] = []

                        sorted_datasets[batch_size][metric][strategy].append((dataset, score))

        # Sort the datasets by score in descending order
        for batch_size in sorted_datasets:
            for metric in sorted_datasets[batch_size]:
                for strategy in sorted_datasets[batch_size][metric]:
                    sorted_datasets[batch_size][metric][strategy] = sorted(
                        sorted_datasets[batch_size][metric][strategy], key=lambda x: x[1], reverse=True
                    )

        file_name = f"{directory}/performance.json"
        with open(file_name, 'w') as f:
            json.dump(sorted_datasets, f)

    def filter_json_data(self, directory: str, k: int):
        with open(f"{directory}/performance.json") as file:
            data: dict = json.load(file)

        results = {}
        for batch_size, batch_data in data.items():
            results[batch_size] = {}
            for metric, strategies in batch_data.items():
                results[batch_size][metric] = {}
                for dataset in self.data.get_dataset_names():
                    strategy_scores = []
                    for strategy, values in strategies.items():
                        for item in values:
                            if item[0] == dataset:
                                strategy_scores.append((strategy, item[1]))
                    strategy_scores.sort(key=lambda x: x[1], reverse=True)
                    top_strategies = strategy_scores[:k]
                    for strategy, score in top_strategies:
                        if strategy not in results[batch_size][metric]:
                            results[batch_size][metric][strategy] = []
                        results[batch_size][metric][strategy].append([dataset, score])

        with open(f"{directory}/top_k_performance.json", 'w') as f:
            json.dump(results, f)


# Directory where all the data provided by Julius lies
source_directory = "../../../kp_test"

# Directory where the JSON files are stored to
destination_directory = "/home/ature/Programming/Python/DB-Mining-Data/JSON"

# Initialize TopK
top_k = TopK(loader_directory=source_directory)

# For each dataset and batch size, create a list for all metrics with a ranging of AL strategies
# threshold of -1 disables the threshold
top_k.collect_top_k(directory=destination_directory, threshold=-1.0, epsilon=1)

# top_k.collect_best_strategy_for_dataset(destination_directory)
# top_k.calculate_generally_best_strategy(directory=destination_directory)
# top_k.calculate_best_strategy_for_metric(directory=destination_directory)

# Calculate rankings of datasets for provided AL strategy, metric and batch size
# top_k.gen_performance_json(directory=destination_directory)
# top_k.filter_json_data(directory=destination_directory, k=5)
top_k.collect_average_score(directory=destination_directory)
