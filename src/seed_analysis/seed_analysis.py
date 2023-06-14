import fnmatch
import os
import pprint

import pandas as pd


class Seed_Analysis:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.csv_files: dict[str, list[str]] = self.find_csv_files()
        self.data_frames: dict[
            str, list[pd.DataFrame]
        ] = self.load_csv_into_data_frames()

    def find_csv_files(self):
        csv_files = {}
        metrics = [
            "accuracy",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1-score",
        ]
        # ! IMPORTANT: The "strategies" is the name of the folder where all Active Learning Strategies reside, must be adjusted if names dont match
        strategies_path = os.path.join(self.file_path, "strategies")

        for root, dirs, files in os.walk(strategies_path):
            for metric in metrics:
                for filename in fnmatch.filter(files, f"{metric}.csv.xz"):
                    if metric not in csv_files:
                        csv_files[metric] = []
                    csv_files[metric].append(os.path.join(root, filename))

        return csv_files

    def pretty_print_csv_files(self):
        pprint.pprint(self.csv_files)

    def pretty_print_data_frames(self):
        pprint.pprint(self.data_frames)

    def load_csv_into_data_frames(self):
        data_frames = {}

        for metric, files in self.csv_files.items():
            data_frames[metric] = [
                pd.read_csv(file, compression="xz") for file in files
            ]
        return data_frames


seed = Seed_Analysis("kp_test")
# seed.pretty_print_csv_files()
seed.pretty_print_data_frames()
