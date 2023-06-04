import json
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from loader import Loader, logger


class TopK:
    def __init__(self, data: Loader = None):
        self.data = data
        self.unwanted = set()
        self.blacklisted_words = [
            # Metrics that are the first derivative
            "lag",
            "selected_indices",
            # Metrics that contain lists
            "CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS",
            "CLOSENESS_TO_CLUSTER_CENTER",
            "CLOSENESS_TO_SAMPLES_OF_SAME_CLASS",
            "CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN",
            "IMPROVES_ACCURACY_BY",
            "COUNT_WRONG_CLASSIFICATIONS",
            "CLOSENESS_TO_DECISION_BOUNDARY",
            "class_distributions_chebyshev_batch",
            "AVERAGE_UNCERTAINTY",
            "OUTLIERNESS",
            "class_distributions_manhattan_batch",
            "CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN",
            "MELTING_POT_REGION",
            "REGION_DENSITY",
            "SWITCHES_CLASS_OFTEN",
            "y_pred_train",
            "y_pred_test",
            # Metrics that contain negative or only non-monotonic values
            "class_distribution_chebyshev_added_up",
            "class_distribution_manhattan_added_up",
            "avg_dist_batch",
            "avg_dist_labeled",
        ]

        self.considered_metric = [
            string
            for string in self.data.get_metric_names()
            if not any(
                word.lower() in string.lower() for word in self.blacklisted_words
            )
        ]

    def calculate_best_strategy_for_metric(self, directory: str):
        for metric in self.considered_metric:
            for batch_size in [1, 5, 10]:
                best_al_strats = defaultdict(float)
                for dataset in self.data.get_dataset_names():
                    file_name = f"{directory}/{dataset}_{batch_size}.json"

                    with open(file_name, "r") as file:
                        data_dict = json.load(file)

                    entries = data_dict.get(metric, [])
                    for index, entry in enumerate(entries):
                        best_al_strats[entry] += 1 / (index + 1)

                total_sum = sum(best_al_strats.values())
                percentages = {
                    key: (value / total_sum) for key, value in best_al_strats.items()
                }

                destination = f"{directory}/{metric}_{batch_size}.json"
                with open(destination, "w") as f:
                    json.dump(percentages, f)
                logger.info(f"Written to: {destination}")

    def calculate_generally_best_strategy(self, directory: str):
        result_dict = defaultdict(float)

        for dataset in self.data.get_dataset_names():
            for batch_size in [1, 5, 10]:
                file_name = f"{directory}/best_strategy_for_{dataset}_{batch_size}.json"

                with open(file_name, "r") as file:
                    best_strategies: dict[str, list[str]] = json.load(file)

                for index, key in enumerate(best_strategies.keys(), start=1):
                    result_dict[key] += 1 / index

        total_sum = sum(result_dict.values())
        percentages = {key: (value / total_sum) for key, value in result_dict.items()}

        destination = f"{directory}/overall_best.json"
        with open(destination, "w") as f:
            json.dump(percentages, f)
        logger.info(f"Written to: {destination}")

    def collect_best_strategy_for_dataset(self, directory: str):
        for dataset in self.data.get_dataset_names():
            for batch_size in [1, 5, 10]:
                result_dict = self.best_al_strategy(
                    dataset=dataset, batch_size=batch_size, directory=directory
                )

                file_name = f"{directory}/best_strategy_for_{dataset}_{batch_size}.json"
                with open(file_name, "w") as f:
                    json.dump(result_dict, f)
                logger.info(f"Written to: {file_name}")

    def best_al_strategy(self, dataset: str, batch_size: int, directory: str):
        file_name = f"{directory}/{dataset}_{batch_size}.json"
        with open(file_name, "r") as file:
            top_k_data: dict[str, list[str]] = json.load(file)

        result_dict = defaultdict(float)
        for value_list in top_k_data.values():
            for index, entry in enumerate(value_list):
                result_dict[entry] += 1 / (index + 1)

        total_sum = sum(result_dict.values())
        percentages = {key: (value / total_sum) for key, value in result_dict.items()}

        return percentages

    def collect_top_k(
        self, directory: str, k: int = 50, threshold: float = 1, epsilon: float = 0
    ):
        for dataset in self.data.get_dataset_names():
            for batch_size in [1, 5, 10]:
                result_dict = {}
                for metric in self.considered_metric:
                    result_dict[metric] = self.get_top_k(
                        dataset,
                        metric,
                        batch_size=batch_size,
                        k=k,
                        threshold=threshold,
                        epsilon=epsilon,
                    )

                file_name = f"{directory}/{dataset}_{batch_size}.json"
                with open(file_name, "w") as f:
                    json.dump(result_dict, f)
                logger.info(f"Written to: {file_name}")

    def get_top_k(
        self,
        dataset: str,
        metric: str,
        batch_size: int,
        k: int = 10,
        threshold: float = 1,
        max_iterations: int = 50,
        epsilon: float = 0,
    ):
        strategy_scores = defaultdict(list)

        for strategy in self.data.get_strategy_names():
            frame: pd.DataFrame = self.data.get_single_dataframe(
                strategy, dataset, metric
            )
            vector: pd.DataFrame = frame.loc[frame["EXP_BATCH_SIZE"] == batch_size]
            vector = vector.iloc[:, :-9].dropna(axis=1)

            if not vector.empty:
                time_series = vector.to_numpy(dtype=np.float32)
                strategy_scores[strategy].extend(time_series)

        combined_sorted = sorted(
            (
                (list(arr), strategy)
                for strategy, series in strategy_scores.items()
                for arr in series
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        def is_monotonic_increasing(series, j):
            for i in range(1, len(series)):
                if series[i] < series[i - 1] - j:
                    return False
            return True

        def check_floats_or_integers(series):
            for element in series:
                if not np.issubdtype(type(element), np.number):
                    return False
            return True

        def reaches_threshold(series):
            for i in range(len(series)):
                if i < (max_iterations - 1) and series[i] >= threshold:
                    return True
            return False

        valid_series = []
        for series, strategy in combined_sorted:
            if not check_floats_or_integers(series):
                self.unwanted.add(metric)

            if (
                check_floats_or_integers(series)
                and is_monotonic_increasing(series, epsilon)
                and reaches_threshold(series)
            ):
                valid_series.append((series, strategy))

        if k >= len(valid_series):
            return [strategy for series, strategy in valid_series]
        else:
            return [strategy for series, strategy in valid_series[:k]]

    def data_processing(self, directory: str):
        values = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        )

        for dataset in self.data.get_dataset_names():
            for batch_size in [1, 5, 10]:
                file_name = f"{directory}/{dataset}_{batch_size}.json"
                with open(file_name, "r") as file:
                    data_dict: dict[str, list[str]] = json.load(file)

                for metric, entries in data_dict.items():
                    for index, entry in enumerate(entries):
                        values[dataset][batch_size][metric][entry] += 1 / (index + 1)

        sorted_datasets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for dataset in values:
            for batch_size in values[dataset]:
                for metric in values[dataset][batch_size]:
                    for strategy in values[dataset][batch_size][metric]:
                        score = values[dataset][batch_size][metric][strategy]
                        sorted_datasets[batch_size][metric][strategy].append(
                            (dataset, score)
                        )

        for batch_size in sorted_datasets:
            for metric in sorted_datasets[batch_size]:
                for strategy in sorted_datasets[batch_size][metric]:
                    sorted_datasets[batch_size][metric][strategy] = sorted(
                        sorted_datasets[batch_size][metric][strategy],
                        key=lambda x: x[1],
                        reverse=True,
                    )

        file_name = f"{directory}/performance.json"
        with open(file_name, "w") as f:
            json.dump(sorted_datasets, f)
        logger.info(f"Written to: {file_name}")


PROJECT_DATA: Loader = Loader("/Users/user/GitHub/Data-Mining/kp_test")
base_directory = "/Users/user/GitHub/Data-Mining/HPC_Processing"

top_k = TopK(PROJECT_DATA)
top_k.collect_top_k(directory=base_directory, threshold=0.0, epsilon=0.05)
top_k.data_processing(directory=base_directory)
