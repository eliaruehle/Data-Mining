import math
import re

import numpy as np
import pandas as pd

from datasets import Loader


class Extrapolation:

    def __init__(self, source_directory: str):
        self.source_directory = source_directory
        self.data = Loader(self.source_directory)

    def data_interpolation(self, strategy: str, dataset: str, metric: str):

        # Load dataframe from given 'strategy', 'dataset' and 'metric'
        frame: pd.DataFrame = self.data.get_single_dataframe(strategy, dataset, metric)

        # If the number of columns is less than 11, quit
        if frame.shape[1] < 11:
            return

        is_list = False
        pattern = r'\[(?!\s*\])[\d.,\s]+\]'

        for ind, r in frame.iterrows():
            if any(re.match(pattern, str(value)) for value in r):
                is_list = True

        if is_list:
            print("with arrays")
            frame = self.interpolation_with_arrays(frame)
            frame = self.interpolation_without_arrays(frame)
            frame.to_csv(f"with_" + metric + ".csv", index=False, sep=',')

        else:
            if re.search(r'_lag', metric):
                print("lag metrics")
                frame = self.interpolation_lag_metrics(frame)
                frame.to_csv(f"without_lag_" + metric + ".csv", index=False, sep=',')

            else:
                print("without arrays")
                frame = self.interpolation_without_arrays(frame)
                frame.to_csv(f"without_" + metric + ".csv", index=False, sep=',')

    def interpolation_with_arrays(self, frame: pd.DataFrame):
        for index, row in frame.iterrows():
            if any(type(el) == str and el == '[]' for el in row):
                first_empty = int((row == '[]').idxmax())
            elif any(type(el) == float and math.isnan(el) for el in row):
                first_empty = int(row.isnull().idxmax())
            else:
                first_empty = 51

            frame = self.change_arrays_to_values(frame, first_empty)
        return frame

    def change_arrays_to_values(self, frame: pd.DataFrame, first_empty: int):
        for index, row in frame.iterrows():
            for position, value in row.iloc[:first_empty - 1].items():
                if type(value) == str and value != '[]':
                    data_list = value.strip('[]').split(",")
                    d = [float(x) for x in data_list]
                    mean_value = sum(d) / len(d)
                    frame.loc[index, position] = mean_value
                elif type(value) == float and not math.isnan(value):
                    mean_value = np.mean(value)
                    frame.loc[index, position] = mean_value
                else:
                    continue
        return frame

    def interpolation_without_arrays(self, frame: pd.DataFrame):
        for index, row in frame.iterrows():
            if any(type(el) == float and math.isnan(el) for el in row):
                first_empty = int(row.isnull().idxmax())
            elif any(row == '[]'):
                first_empty = int((row == '[]').idxmax())
            else:
                continue

            last_not_empty = row[first_empty - 1]
            frame.iloc[index, first_empty:50] = last_not_empty

        return frame

    def interpolation_lag_metrics(self, frame: pd.DataFrame):
        for index, row in frame.iterrows():
            if any(type(el) == float and math.isnan(el) for el in row):
                first_empty = int(row.isnull().idxmax())
            elif any(row == '[]'):
                first_empty = int((row == '[]').idxmax())
            else:
                continue

            frame.iloc[index, first_empty:50] = 0
        return frame
