import math
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
        path_to_metric = f"{self.source_directory}/{strategy}/{dataset}/{metric}.csv.xz"
        frame: pd.DataFrame = pd.read_csv(path_to_metric)

        # Check if dataframe is a lag metric
        is_lag_metric: bool = "_lag" in metric

        # Do extrapolation
        frame = self.do_everything(frame=frame, is_lag=is_lag_metric, file_name=path_to_metric)

        # Path where changed CSV file is written to
        subdirectories: str = f"{self.destination_directory}/{strategy}/{dataset}"

        # Save changed dataframe
        if not os.path.exists(subdirectories):
            os.makedirs(subdirectories)

        frame.to_csv(f"{subdirectories}/{metric}.csv.xz", index=False)

    @staticmethod
    def do_everything(frame: pd.DataFrame, is_lag: bool, file_name: str):
        second_last_columns = len(frame.columns) - 1
        for row_idx, row in frame.iterrows():
            last_valid = 0
            for col_idx, column in enumerate(frame.columns):
                # Get the current value
                value = frame.iloc[row_idx, col_idx]

                # If 'value' is NaN, set all of its successors according to is_leg
                if Extrapolation.is_nan(value):
                    value = last_valid
                    frame.iloc[row_idx, col_idx:second_last_columns] = 0 if is_lag else last_valid

                # Check if 'value' is a string
                if isinstance(value, str) and not isinstance(value, list):
                    try:
                        value = ast.literal_eval(value)

                        # If 'parsed' == [], set all of its successors according to is_leg
                        if not value:
                            frame.iloc[row_idx, col_idx:second_last_columns] = 0 if is_lag else last_valid

                        # If 'parsed' is a float, save its float version
                        if isinstance(value, float):
                            frame.iloc[row_idx, col_idx] = value

                    # If the value is weirdly formatted, pass
                    except ValueError:
                        if not isinstance(value, list):
                            print(f"Tried to cast {value} (Type: {type(value)}) of {file_name} at index:({row_idx},{col_idx})\n")
                            value = frame.iloc[row_idx, col_idx]

                # If it is a list, calculate the mean
                if isinstance(value, list):
                    mean = np.mean(value)
                    value = mean if not Extrapolation.is_nan(mean) else 0
                    frame.iloc[row_idx, col_idx] = value

                # Update last valid value
                last_valid = value

        # Return the frame
        return frame

    @staticmethod
    def is_nan(value) -> bool:
        if pd.isna(value):
            return True

        if isinstance(value, str):
            if "nan" in value.lower():
                return True

            try:
                parsed = ast.literal_eval(value)

                if isinstance(parsed, (float, int, np.float_)) and math.isnan(parsed) or pd.isna(parsed):
                    return True

            except ValueError:
                pass

        return False

    @staticmethod
    def get_subdirectories(path: str):
        return [entry.name for entry in os.scandir(path) if entry.is_dir() and not entry.name.startswith('_')]

    @staticmethod
    def get_files(path: str):
        return [entry.name[:-7] for entry in os.scandir(path) if entry.is_file()]


extrapolation = Extrapolation(
    source_directory="/beegfs/ws/1/s5968580-al_benchmark/exp_results/fuller_exp",
    destination_directory="/home/vime121c/Workspaces/scratch/vime121c-db-project/Pain"
)

warnings.filterwarnings('ignore')

if len(sys.argv) > 1:
    index = int(sys.argv[1])
    extrapolation.extrapolate_strategy(index)
