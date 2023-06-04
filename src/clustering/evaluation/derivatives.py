from datasets.loader import Loader
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Derivation:

    def __init__(self, directory: str):
        self.lag_metrics = [metric for metric in Loader.list_metrics(base_dir=directory) if 'lag' in metric]
        self.data = Loader(base_dir=directory, wanted_metrics=self.lag_metrics)

    # Loads formatted dataframe for a given dataset, strategy, metric and batch_size and returns it as a 2D list
    def load_diff(self, dataset: str, strategy: str, metric: str, batch_size: int) -> List[List[int]]:
        # Load lag data
        time_series_list: List[np.ndarray] = []

        frame: pd.DataFrame = self.data.get_single_dataframe(strategy, dataset, metric)
        vector: pd.DataFrame = frame.loc[frame["EXP_BATCH_SIZE"] == batch_size]
        vector = vector.iloc[:, :-9].dropna(axis=1)

        if not vector.empty:
            time_series_list.extend(vector.to_numpy())

        time_series = [list(arr) for arr in time_series_list]
        return time_series

    # Plots a list of time series data
    @staticmethod
    def plot_time_series(dataset_name: str, time_series: List[List[float]], legend_labels: List[str] = None,
                         save_path: str = None):

        if len(time_series) != len(legend_labels):
            raise ValueError("Number of time series and legend labels must be equal.")

        if legend_labels is not None:
            for i, series in enumerate(time_series):
                plt.plot(series, label=legend_labels[i])
        else:
            for series in time_series:
                plt.plot(series)

        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(dataset_name)

        if legend_labels is not None:
            plt.legend()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    # Creates a separate plot for each dataset. One plot contains all time series of a dataframe specified by a given
    # strategy, metric and batch_size
    def plot_all_datasets(self, strategy: str, metric: str, batch_size: int, directory: str = None):
        for dataset in self.data.get_dataset_names():
            data = self.load_diff(dataset=dataset, strategy=strategy, metric=metric, batch_size=batch_size)
            file_name = None if directory is None else f"{directory}/{dataset}_{strategy}_{metric}_{batch_size}"
            Derivation.plot_time_series(dataset_name=f"{strategy}, {dataset}, {metric}, {batch_size}", time_series=data,
                                        save_path=file_name)

    # Plots an average time series for each dataset in one single plot
    def plot_all_datasets_average(self, strategy: str, metric: str, batch_size: int, directory: str = None):
        average = []
        for dataset in self.data.get_dataset_names():
            data = self.load_diff(dataset=dataset, strategy=strategy, metric=metric, batch_size=batch_size)
            average.append(Derivation.calculate_average_time_series(data))

        file_name = None if directory is None else f"{directory}/average_{strategy}_{metric}_{batch_size}"
        Derivation.plot_time_series(dataset_name=f"Average: {strategy}, {metric}, {batch_size}", time_series=average,
                                    legend_labels=self.data.get_dataset_names(),
                                    save_path=file_name)

    # Calculates the average time series of a list of time series data
    @staticmethod
    def calculate_average_time_series(time_series_list: List[List[float]]):
        if not time_series_list:
            return None

        # Determine the length of the time series
        series_length = len(time_series_list[0])

        # Initialize a list to store the cumulative sum of each data point
        cumulative_sum = [0] * series_length

        # Calculate the cumulative sum of each data point across all time series
        for series in time_series_list:
            cumulative_sum = [cumulative_sum[i] + series[i] for i in range(series_length)]

        # Calculate the average by dividing each data point by the number of time series
        average_time_series = [cumulative_sum[i] / len(time_series_list) for i in range(series_length)]

        return average_time_series

    def save_as_one(self, strategy: str, metric: str, directory: str):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        for index, batch_size in enumerate([1, 5, 10]):
            average = []
            for dataset in self.data.get_dataset_names():
                data = self.load_diff(dataset=dataset, strategy=strategy, metric=metric, batch_size=batch_size)
                average.append(Derivation.calculate_average_time_series(data))
            for avg_data in average:
                axes[index].plot(avg_data)

            axes[index].set_title(f'Batch Size: {batch_size}')  # Set title for the subplot

        # Add a single legend for all subplots
        fig.legend(self.data.get_dataset_names(), loc='lower center',
                   ncol=len(self.data.get_dataset_names()), bbox_to_anchor=(0.5, -0.1))
        fig.suptitle(f"Average {metric} of {strategy}")

        plt.savefig(f'{directory}/{strategy}_{metric}.png', bbox_inches='tight')
        plt.close()

    def save_all_averages(self, directory: str):
        for strategy in self.data.get_strategy_names():
            for metric in self.lag_metrics:
                self.save_as_one(strategy=strategy, metric=metric, directory=directory)


# Directory where all the data provided by Julius lies
source_directory = "../../../kp_test"

# Directory where the plots should be saved to
destination_directory = "/home/ature/Programming/Python/DB-Mining-Data/PLOTS"

# Initialize Derivation
derivation = Derivation(directory=source_directory)

# Generate and save all plots
derivation.save_all_averages(directory=destination_directory)
