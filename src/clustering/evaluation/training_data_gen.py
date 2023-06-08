import json
from collections import defaultdict
from statistics import mean
from typing import List

import numpy as np
import pandas as pd
import os


class TrainingDataGenerator:

    def __init__(self, loader_directory: str, destination: str):
        self.source = loader_directory
        self.destination = destination

        self.hyperparameters = self.get_hyperparameters()
        self.strategies = self.get_strategies()
        self.datasets = self.get_common_datasets()
        self.metrics = self.get_common_metrics()

    def gen_base_data(self):
        subdirectory = f"{self.destination}/dataset_batch_size"
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        for dataset in self.datasets:
            for batch_size in [1, 5, 10]:
                result_dict = {}
                for metric in self.metrics:
                    result_dict[metric] = self.get_top_k(dataset=dataset, metric=metric, batch_size=batch_size)

                file_name = f"{self.destination}/dataset_batch_size/{dataset}_{batch_size}.json"
                with open(file_name, 'w') as f:
                    json.dump(result_dict, f)

    def get_top_k(self, dataset: str, metric: str, batch_size: int):

        # Load data
        time_series_list: List[np.ndarray] = []
        strategy_names: List[str] = []

        for strategy in self.strategies:
            frame: pd.DataFrame = self.load_single_csv(strategy, dataset, metric)
            vector: pd.DataFrame = frame.loc[frame["EXP_BATCH_SIZE"] == batch_size]
            vector = vector.iloc[:, :-9].dropna(axis=1)

            if not vector.empty:
                time_series_list.extend(vector.to_numpy())
                strategy_names.extend([strategy] * len(vector))

        time_series = [list(arr) for arr in time_series_list]

        def check_floats_or_integers(input_series):
            for element in input_series:
                if not isinstance(element, (float, int)):
                    return False
            return True

        def calculate_score(input_series):
            return sum(input_series)  # Use sum() to calculate the score

        # Calculate the integral for all series that are of type float or int
        valid_series = []
        for series, strategy in zip(time_series, strategy_names):
            if check_floats_or_integers(series):
                valid_series.append((strategy, calculate_score(series)))

        valid_series = sorted(valid_series, key=lambda x: x[1], reverse=True)  # Sort by score

        average = defaultdict(int)
        occurrences = defaultdict(int)

        for dataset, score in valid_series:
            occurrences[dataset] += 1
            average[dataset] += score

        for dataset in average.keys():
            average[dataset] /= occurrences[dataset]

        average_list = [(dataset, score) for dataset, score in average.items()]

        sorted_average_list = sorted(average_list, key=lambda x: x[1], reverse=True)

        return sorted_average_list

    @staticmethod
    def get_subdirectories(path: str) -> List[str]:
        return [entry.name for entry in os.scandir(path) if entry.is_dir()]

    @staticmethod
    def get_files(path: str) -> List[str]:
        return [entry.name[:-7] for entry in os.scandir(path) if entry.is_file()]

    def get_strategies(self) -> List[str]:
        return TrainingDataGenerator.get_subdirectories(self.source)

    def get_common_datasets(self):
        common_datasets = set()
        for strategy in self.get_strategies():
            path_to_datasets = f"{self.source}/{strategy}"
            if len(common_datasets) == 0:
                common_datasets = set(self.get_subdirectories(path_to_datasets))
            else:
                common_datasets.intersection_update(set(self.get_subdirectories(path_to_datasets)))
        return list(set(common_datasets))

    def get_common_metrics(self):
        common_metrics = set()
        for strategy in self.get_strategies():
            for dataset in self.get_common_datasets():
                path_to_metric = f"{self.source}/{strategy}/{dataset}"
                if len(common_metrics) == 0:
                    common_metrics = set(self.get_files(path_to_metric))
                else:
                    common_metrics.intersection_update(set(self.get_files(path_to_metric)))
        return list(set(common_metrics))

    def get_hyperparameters(self):
        return pd.read_csv(f"{self.source}/{next((x for x in os.listdir(self.source) if 'done_workload' in x), None)}")

    def load_single_csv(self, strategy: str, dataset: str, metric: str) -> pd.DataFrame:
        path_to_metric = f"{self.source}/{strategy}/{dataset}/{metric}.csv.xz"
        return pd.merge(self.remove_nan_rows(pd.read_csv(path_to_metric)), self.hyperparameters, on="EXP_UNIQUE_ID")

    @staticmethod
    def remove_nan_rows(data_frame: pd.DataFrame) -> pd.DataFrame:
        return data_frame.dropna(subset=data_frame.columns[:-1], how="all")


# Directory where all the data provided by Julius lies
source_directory = "../../../kp_test/strategies"

# Directory where the JSON files are stored to
destination_directory = "/home/ature/Programming/Python/DB-Mining-Data/JSON"

generator = TrainingDataGenerator(loader_directory=source_directory, destination=destination_directory)
generator.gen_base_data()
