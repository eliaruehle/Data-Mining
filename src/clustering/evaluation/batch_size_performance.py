from typing import List, Dict

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class BatchSizePerformance:

    def __init__(self, source_directory: str, destination_directory: str):
        self.wanted_metrics = ['accuracy', 'weighted_f1-score', 'weighted_precision', 'weighted_recall']
        self.source = source_directory
        self.destination = destination_directory

        self.hyperparameters = pd.read_csv(f"{self.source}/05_done_workload.csv")

    @staticmethod
    def get_subdirectories(path: str) -> List[str]:
        return sorted([entry.name for entry in os.scandir(path) if entry.is_dir() and not entry.name.startswith('_')])

    def get_all_strategies(self):
        return self.get_subdirectories(self.source)

    @staticmethod
    def remove_nan_rows(data_frame: pd.DataFrame) -> pd.DataFrame:
        return data_frame.dropna(subset=data_frame.columns[:-1], how="all")

    def generate_plot_for(self, dataset: str, metric: str):
        categorized = {}
        for batch_size in [1, 5, 10]:
            categorized[batch_size] = []

        # Calculate average slope for a given dataset and metric
        for strategy in self.get_all_strategies():
            path_to_metric = f'{self.source}/{strategy}/{dataset}/{metric}.csv.xz'

            # Load dataframe
            df = pd.merge(pd.read_csv(path_to_metric), self.hyperparameters, on="EXP_UNIQUE_ID")
            for batch_size in [1, 5, 10]:
                dummy: pd.DataFrame = df.loc[df["EXP_BATCH_SIZE"] == batch_size].iloc[:, :-9]
                dummy.to_csv(f'{self.destination}/{metric}_{strategy}_{batch_size}.csv', index=False)
                time_series = dummy.to_numpy().tolist()
                time_series = time_series if time_series is not None else []
                for series in time_series:
                    categorized[batch_size].append(series)

        to_plot = []
        legend = []
        for key in [1, 5, 10]:
            # Calculate average time series
            time_series = categorized[key]
            to_plot.append([sum(elements) / len(time_series) for elements in zip(*time_series)])
            legend.append(f'Batch size: {key}')

        # Plot the time series
        for series in to_plot:
            plt.plot(series)
        plt.xlabel('Score')
        plt.xlabel('Iteration')
        plt.title(f'Average time series for dataset @{dataset} and metric @{metric}')
        plt.legend(legend)
        plt.show()


source = '/home/ature/University/6th-Semester/Data-Mining/kp_test/strategies'
destination = '/home/ature/Programming/Python/DB-Mining-Data/Plots'

analysis = BatchSizePerformance(source_directory=source, destination_directory=destination)
analysis.generate_plot_for(dataset='Iris', metric='accuracy')
