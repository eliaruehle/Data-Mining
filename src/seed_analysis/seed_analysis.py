import fnmatch
import os
import pprint

import pandas as pd


class Seed_Analysis:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.csv_files = self.find_csv_files()
        self.data_frames = self.load_data_frames()

    def find_csv_files(self):
        csv_files = {}
        metrics = [
            "accuracy",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1-score",
        ]
        # ! IMPORTANT: The "strategies" is the name of the directory where all Active Learning Strategies reside, must be adjusted if names do not match
        strategies_path = os.path.join(self.file_path, "strategies")

        for root, dirs, files in os.walk(strategies_path):
            for metric in metrics:
                for filename in sorted(fnmatch.filter(files, f"{metric}.csv.xz")):
                    if metric not in csv_files:
                        csv_files[metric] = []
                    csv_files[metric].append(os.path.join(root, filename))

        return csv_files

    def load_data_frames(self):
        data_frames = {}
        for metric, files in self.csv_files.items():
            data_frames[metric] = []
            for file in files:
                df = pd.read_csv(file, compression="xz")
                df = df.drop(columns="EXP_UNIQUE_ID", errors="ignore")
                # store a tuple with the filename and the DataFrame
                data_frames[metric].append((file, df))
        return data_frames

    def get_monotonically_increasing_or_same_rows(self):
        mono_increasing_rows = {}
        for metric, dfs in self.data_frames.items():
            mono_increasing_rows[metric] = []
            for file, df in dfs:  # unpack the tuple
                mono_rows = []
                for _, row in df.iterrows():
                    # check if row values are monotonically increasing or same and not all are 1.0
                    if list(row) == sorted(row) and not all(val == 1.0 for val in row):
                        mono_rows.append(row)
                mono_df = pd.DataFrame(mono_rows, columns=df.columns).reset_index(
                    drop=True
                )  # reset index of the new dataframe
                # store a tuple with the filename and the DataFrame
                mono_increasing_rows[metric].append((file, mono_df))
        return mono_increasing_rows

    def save_monotonically_increasing_rows(
        self, mono_increasing_rows, output_dir, min_entries=10
    ):
        os.makedirs(output_dir, exist_ok=True)
        for metric, results in mono_increasing_rows.items():
            for i, (file, df) in enumerate(results):
                if (
                    len(df) >= min_entries
                ):  # check if DataFrame has at least min_entries rows
                    # Parse the necessary parts from the file path
                    parts = file.split(os.sep)
                    # Remove the original file extension
                    parts[-1] = os.path.splitext(parts[-1])[0]
                    output_filename = "_".join(parts[-3:]) + f"_mono_increasing.csv"
                    output_path = os.path.join(output_dir, output_filename)
                    df.to_csv(output_path, index=True)

    def save_data_frames(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for metric, dfs in self.data_frames.items():
            for i, (file, df) in enumerate(dfs):  # unpack the tuple
                # Parse the necessary parts from the file path
                parts = file.split(os.sep)
                # Remove the original file extension
                parts[-1] = os.path.splitext(parts[-1])[0]
                output_filename = "_".join(parts[-3:]) + ".csv"
                output_path = os.path.join(output_dir, output_filename)
                df.to_csv(output_path, index=True)

    def pretty_print_csv_files(self):
        pprint.pprint(self.csv_files)

    def pretty_print_data_frames(self):
        pprint.pprint(self.data_frames)


if __name__ == "__main__":
    seed = Seed_Analysis(file_path="kp_test")
    mono_increasing = seed.get_monotonically_increasing_or_same_rows()
    seed.save_monotonically_increasing_rows(
        mono_increasing_rows=mono_increasing,
        output_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/mono_increasing_10",
    )
