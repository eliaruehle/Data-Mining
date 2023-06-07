import sys
import warnings

import numpy as np
import pandas as pd
import os as os
import ast


class Extrapolation:

    def __init__(self, source_directory: str, destination_directory: str):
        self.source_directory = source_directory
        self.destination_directory = destination_directory
        self.LAST = 50

    # Extrapolate data of a given strategy
    def extrapolate_strategy(self, strategy_index: int):
        strategy = Extrapolation.get_subdirectories(self.source_directory)[strategy_index]

        dataset_path = f"{self.source_directory}/{strategy}/"
        for dataset in Extrapolation.get_subdirectories(dataset_path):
            metric_path = f"{dataset_path}/{dataset}/"
            for metric in Extrapolation.get_files(metric_path):
                self.extrapolate(strategy, dataset, metric)

    # Extrapolate the dataframe of a given strategy, dataset and metric
    def extrapolate(self, strategy: str, dataset: str, metric: str):

        # Load dataframe from given 'strategy', 'dataset' and 'metric'
        frame: pd.DataFrame = pd.read_csv(f"{self.source_directory}/{strategy}/{dataset}/{metric}.csv.xz")

        # Check if dataframe is a lag metric
        is_lag_metric: bool = "_lag" in metric

        # Do extrapolation
        frame = self.do_everything(frame=frame, is_lag=is_lag_metric)

        # Path where changed CSV file is written to
        subdirectories: str = f"{self.destination_directory}/{strategy}/{dataset}"

        # Save changed dataframe
        if not os.path.exists(subdirectories):
            os.makedirs(subdirectories)

        frame.to_csv(f"{subdirectories}/{metric}.csv.xz", index=False)

    def do_everything(self, frame: pd.DataFrame, is_lag: bool):
        for row_idx, row in frame.iterrows():
            last_valid = None
            for col_idx, column in enumerate(frame.columns):
                # Get the current value
                value = frame.iloc[row_idx, col_idx]

                # If 'value' is NaN, set all of its successors according to is_leg
                if pd.isna(value):
                    frame.iloc[row_idx, col_idx:self.LAST] = 0 if is_lag else last_valid
                    break   # Stop checking other values in this row

                # Check if 'value' is a string
                if isinstance(value, str):
                    value = ast.literal_eval(value)

                    # If 'parsed' == [], set all of its successors according to is_leg
                    if not value:
                        frame.iloc[row_idx, col_idx:self.LAST] = 0 if is_lag else last_valid
                        break   # Stop checking other values in this row

                    # If 'parsed' is a float, save its float version
                    if isinstance(value, float):
                        frame.iloc[row_idx, col_idx] = value

                # If it is a list, calculate the mean
                if isinstance(value, list):
                    frame.iloc[row_idx, col_idx] = np.mean(value)

        # Return the frame
        return frame

    @staticmethod
    def get_subdirectories(path: str):
        return [entry.name for entry in os.scandir(path) if entry.is_dir()]

    @staticmethod
    def get_files(path: str):
        return [entry.name[:-7] for entry in os.scandir(path) if entry.is_file()]


extrapolation = Extrapolation(
    source_directory="/home/ature/University/6th-Semester/Data-Mining/kp_test",
    destination_directory="/home/ature/Programming/Python/DB-Mining-Data/EXTRAPOLATION"
)

warnings.filterwarnings('ignore')

if len(sys.argv) > 1:
    index = int(sys.argv[1])
    extrapolation.extrapolate_strategy(index)
