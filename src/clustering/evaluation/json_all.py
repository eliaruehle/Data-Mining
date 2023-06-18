import os
import sys

import pandas as pd
import json

import numpy as np

from typing import List, Callable, Tuple, Dict


class JsonAll:

    def __init__(self, loader_directory: str, destination: str):
        self.source = loader_directory
        self.destination = destination

        self.hyperparameters = self.get_hyperparameters()
        self.hyperparameters["EXP_UNIQUE_ID"] = self.hyperparameters["EXP_UNIQUE_ID"].astype(int)

    @staticmethod
    def get_subdirectories(path: str) -> List[str]:
        return sorted([entry.name for entry in os.scandir(path) if entry.is_dir()])

    @staticmethod
    def get_files(path: str) -> List[str]:
        return sorted([entry.name[:-7] for entry in os.scandir(path) if entry.is_file()])

    # Method that returns all possible AL strategies
    def get_all_strategies(self):
        return self.get_subdirectories(self.source)

    # Method that returns all datasets
    def get_all_datasets(self):
        datasets = []
        for strategy in self.get_all_strategies():
            path_to_datasets = f"{self.source}/{strategy}"
            datasets.extend(self.get_subdirectories(path_to_datasets))
        return sorted(list(set(datasets)))

    # Method that return all possible metrics for a dataset
    def get_all_metrics_for(self, dataset: str):
        metrics = []
        for strategy in self.get_all_strategies():
            path_to_metric = f"{self.source}/{strategy}/{dataset}"
            try:
                metrics.extend(self.get_files(path_to_metric))
            except FileNotFoundError:
                pass
        return sorted(list(set(metrics)))

    # Method that return all possible metrics there are
    def get_all_metrics(self):
        metrics = []
        for strategy in self.get_all_strategies():
            for dataset in self.get_all_datasets():
                path_to_metric = f"{self.source}/{strategy}/{dataset}"
                try:
                    metrics.extend(self.get_files(path_to_metric))
                except FileNotFoundError:
                    pass
        return sorted(list(set(metrics)))

    # Read hyperparameters from done workload
    def get_hyperparameters(self):
        return pd.read_csv(f"{self.source}/05_done_workload.csv")

    @staticmethod
    def remove_nan_rows(data_frame: pd.DataFrame) -> pd.DataFrame:
        return data_frame.dropna(subset=data_frame.columns[:-1], how="all")

    # Load a single CSV file
    def load_single_csv(self, strategy: str, dataset: str, metric: str) -> pd.DataFrame:
        path_to_metric = f"{self.source}/{strategy}/{dataset}/{metric}.csv.xz"
        try:
            return pd.merge(self.remove_nan_rows(pd.read_csv(path_to_metric)), self.hyperparameters, on="EXP_UNIQUE_ID")
        except ValueError:
            try:
                df = self.remove_nan_rows(pd.read_csv(path_to_metric))
                df["EXP_UNIQUE_ID"] = df["EXP_UNIQUE_ID"].astype(int)
                return pd.merge(df, self.hyperparameters, on="EXP_UNIQUE_ID")
            except ValueError:
                print(f"Could not merge {path_to_metric} with hyperparameters")
                return pd.DataFrame()
        except FileNotFoundError:
            print(f"Could not find {path_to_metric}. Returning empty dataframe instead")
            return pd.DataFrame()

    def calculate_score_for(self, dataset: str, batch_size: int, metric: str, score: Callable):

        scores = []

        for strategy in self.get_all_strategies():
            try:
                df = self.load_single_csv(strategy=strategy, dataset=dataset, metric=metric)
                df = df.loc[df["EXP_BATCH_SIZE"] == batch_size]
                df = df.iloc[:, :(int(50 / batch_size) - 59)].dropna(axis=1)

                as_numpy = df.to_numpy()
                if len(as_numpy) > 0:
                    try:
                        strategy_average = sum([score(series) for series in as_numpy]) / len(as_numpy)
                        scores.append((strategy, strategy_average))
                    except TypeError:
                        print(f"Datatype of {strategy}, {dataset}, {metric}: {as_numpy.dtype}")
                        scores.append((strategy, 0))
                else:
                    scores.append((strategy, 0))

            except FileNotFoundError:
                print(f"File for {strategy}/{dataset}/{metric} not found. Should not get triggered")
                scores.append((strategy, 0))
            except KeyError:
                print(f"KeyError. File for {strategy}/{dataset}/{metric} not found")

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def write_dataset_batch_size(self, score: Callable):

        # Create directory 'dataset_batch_size' if it doesn't exist
        subdirectory = f"{self.destination}/dataset_batch_size"
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        # For each dataset, create dataset_batch-size.json
        for dataset in self.get_all_datasets():
            self.write_dataset_batch_size_for(dataset=dataset, score=score)

    def write_dataset_batch_size_for(self, dataset: str, score: Callable):
        # Create directory 'dataset_batch_size' if it doesn't exist
        subdirectory = f"{self.destination}/dataset_batch_size"
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        # Create dataset_batch-size-json
        for batch_size in [1, 5, 10]:
            result_dict = {}
            for metric in self.get_all_metrics():
                result_dict[metric] = self.calculate_score_for(dataset, batch_size, metric, score)
            file_name = f"{self.destination}/dataset_batch_size/{dataset}_{batch_size}.json"
            with open(file_name, 'w') as f:
                json.dump(result_dict, f)

    @staticmethod
    def score_integral(time_series: np.ndarray):
        return np.sum(time_series)

    @staticmethod
    def get_dataset_names_from_json(directory):
        dataset_names = []
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                dataset_name = filename.rsplit("_", 1)[0]
                dataset_names.append(dataset_name)
        return dataset_names

    def gen_performance_json(self, directory: str):
        result_dict = {}
        for dataset in json_all.get_dataset_names_from_json(directory=directory):
            for batch_size in [1, 5, 10]:
                if batch_size not in result_dict:
                    result_dict[batch_size] = {}
                file_name = f"{directory}/{dataset}_{batch_size}.json"
                with open(file_name, 'r') as file:
                    try:
                        data: Dict[str, List[Tuple[str, int]]] = json.load(file)
                    except json.decoder.JSONDecodeError:
                        data = {}
                        print(file_name)
                for strategy in data.keys():
                    if dataset not in result_dict[batch_size]:
                        result_dict[batch_size][dataset] = {}
                    result_dict[batch_size][dataset][strategy] = data[strategy]

        file_name = f"{self.destination}/average_performance.json"
        with open(file_name, 'w') as f:
            json.dump(result_dict, f)


# Entry point
hpc: bool = False

if hpc:
    # Directory where all the data provided by Julius lies
    source_directory = "/home/vime121c/Workspaces/scratch/vime121c-db-project/Extrapolation"

    # Directory where the JSON files are stored to
    destination_directory = "/home/vime121c/Workspaces/scratch/vime121c-db-project/JSON"

    json_all = JsonAll(loader_directory=source_directory, destination=destination_directory)

    if len(sys.argv) > 1:
        index = int(sys.argv[1])
        print(f"Datasets: {json_all.get_all_datasets()}")
        print(f"This dataset: {json_all.get_all_datasets()[index]}")
        json_all.write_dataset_batch_size_for(json_all.get_all_datasets()[index], score=json_all.score_integral)

else:
    # Directory where all the data provided by Julius lies
    source_directory = "/home/ature/University/6th-Semester/Data-Mining/kp_test/strategies"

    # Directory where the JSON files are stored to
    destination_directory = "/home/ature/University/6th-Semester/Data-Mining/src/clustering/generated/JSON"

    json_all = JsonAll(loader_directory=source_directory, destination=destination_directory)
    json_all.gen_performance_json(directory="/home/ature/University/6th-Semester/Data-Mining/src/clustering/generated/JSON/dataset_batch_size")
