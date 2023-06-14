import concurrent.futures
import fnmatch
import os
import pprint

import pandas as pd


class Seed_Analysis:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.csv_files: dict[str, list[str]] = self.find_csv_files()
        self.data_frames: dict[str, list[pd.DataFrame]] = self.load_data_frames()

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

    def load_data_frames(self):
        data_frames = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for metric, files in self.csv_files.items():
                future_to_df = {
                    executor.submit(pd.read_csv, file, compression="xz"): file
                    for file in files
                }
                data_frames[metric] = []
                for future in concurrent.futures.as_completed(future_to_df):
                    df = future.result()
                    df = df.drop(columns="EXP_UNIQUE_ID", errors="ignore")
                    data_frames[metric].append(df)
        return data_frames

    def pretty_print_csv_files(self):
        pprint.pprint(self.csv_files)

    def pretty_print_data_frames(self):
        pprint.pprint(self.data_frames)

    def save_data_frames(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for metric, dfs in self.data_frames.items():
            for i, df in enumerate(dfs):
                output_path = os.path.join(output_dir, f"{metric}_{i}.csv")
                df.to_csv(output_path, index=False)


seed = Seed_Analysis("kp_test")
# seed.pretty_print_csv_files()
# seed.pretty_print_data_frames()
seed.save_data_frames("/Users/user/GitHub/Data-Mining/test_results")
