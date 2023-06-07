import math
import warnings

import numpy as np
import pandas as pd
import os as os
import ast

from datasets import Loader


class Extrapolation:

    def __init__(self, source_directory: str, destination_directory: str):
        self.source_directory = source_directory
        self.destination_directory = destination_directory
        self.LAST = 50
        self.data = Loader(self.source_directory)

    # Do extrapolation for the entire dataset
    def extrapolate_all(self):
        for strategy in self.data.get_strategy_names():
            for dataset in self.data.get_dataset_names():
                for metric in self.data.get_metric_names():
                    self.extrapolate(strategy, dataset, metric)

    # Extrapolate the dataframe of a given strategy, dataset and metric
    def extrapolate(self, strategy: str, dataset: str, metric: str):

        # Path where changed CSV file is written to
        subdirectories: str = f"{self.destination_directory}/{strategy}/{dataset}"

        # Load dataframe from given 'strategy', 'dataset' and 'metric'
        frame: pd.DataFrame = pd.read_csv(f"{self.source_directory}/{strategy}/{dataset}/{metric}.csv.xz")

        # Check if dataframe is a lag metric
        is_lag_metric: bool = "_lag" in metric

        # 1. Calculate the average of all list elements. Empty lists are replaced with np.nan
        frame = Extrapolation.calculate_list_mean(frame=frame)

        # 2. Extrapolate missing values according to 'is_lag_metric'
        if is_lag_metric:
            frame = self.extrapolate_lag(frame=frame)
        else:
            frame = self.extrapolate_normal(frame=frame)

        # Save changed dataframe
        if not os.path.exists(subdirectories):
            os.makedirs(subdirectories)

        frame.to_csv(f"{subdirectories}/{metric}.csv.xz", index=False)

    # Calculates the mean of all lists and propagates the mean of the valid list
    @staticmethod
    def calculate_list_mean(frame: pd.DataFrame):
        for column in frame.columns:
            for index, value in frame[column].items():
                element = value
                # Check if value is a string
                if isinstance(value, str):
                    element = ast.literal_eval(value)
                if isinstance(element, list):
                    frame.at[index, column] = np.mean(element)
                elif isinstance(element, float):
                    frame.at[index, column] = element
                elif element == '[]':
                    frame.at[index, column] = np.nan
                    break  # Stop checking other values in this row
        return frame

    # Replace all values following a NaN with the last valid value
    def extrapolate_normal(self, frame: pd.DataFrame):
        for index, row in frame.iterrows():
            last_valid = None
            for column_idx, value in enumerate(row):
                if pd.isna(value) or math.isnan(value):
                    if last_valid is not None:
                        frame.iloc[index, column_idx:self.LAST] = last_valid
                        break   # Stop checking other values in this row
                else:
                    last_valid = value
        return frame

    # Replace all values following a NaN with 0
    def extrapolate_lag(self, frame: pd.DataFrame):
        for index, row in frame.iterrows():
            for column_idx, value in enumerate(row):
                if pd.isna(value):
                    frame.iloc[index, column_idx:self.LAST] = 0
                    break  # Stop checking other values in this row
        return frame


extrapolation = Extrapolation(
    source_directory="/home/ature/University/6th-Semester/Data-Mining/kp_test",
    destination_directory="/home/ature/Programming/Python/DB-Mining-Data/EXTRAPOLATION"
)

warnings.filterwarnings('ignore')

extrapolation.extrapolate_all()

# extrapolation.extrapolate("ALIPY_RANDOM", "Iris", "AVERAGE_UNCERTAINTY")
