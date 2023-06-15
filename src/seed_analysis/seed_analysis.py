import fnmatch
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    def save_data_frames(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for metric, dfs in self.data_frames.items():
            # unpack the tuple
            for i, (file, df) in enumerate(dfs):
                # Parse the necessary parts from the file path
                parts = file.split(os.sep)
                # Remove the original file extension
                parts[-1] = os.path.splitext(parts[-1])[0]
                output_filename = "_".join(parts[-3:]) + ".csv"
                output_path = os.path.join(output_dir, output_filename)
                df.to_csv(output_path, index=True)

    def count_unique_columns(self):
        column_counts = {"first_column": {}, "last_column": {}}
        for metric, dfs in self.data_frames.items():
            column_counts["first_column"][metric] = []
            column_counts["last_column"][metric] = []
            for file, df in dfs:  # unpack the tuple
                # get the first and last columns
                first_col = df.iloc[:, 0]
                last_col = df.iloc[:, -1]

                # count frequency
                first_col_freq = Counter(first_col)
                last_col_freq = Counter(last_col)

                # convert the Counter dict to a DataFrame
                first_col_df = pd.DataFrame.from_records(
                    list(first_col_freq.items()), columns=["value", "count"]
                )

                last_col_df = pd.DataFrame.from_records(
                    list(last_col_freq.items()), columns=["value", "count"]
                )

                # store a tuple with the filename and the DataFrame
                column_counts["first_column"][metric].append((file, first_col_df))
                column_counts["last_column"][metric].append((file, last_col_df))
        return column_counts

    def save_column_counts_to_csv(self, output_dir, column_counts):
        os.makedirs(output_dir, exist_ok=True)
        for column, metrics in column_counts.items():
            for metric, dfs in metrics.items():
                for i, (file, data) in enumerate(dfs):
                    # Parse the necessary parts from the file path
                    parts = file.split(os.sep)
                    # Remove the original file extension
                    parts[-1] = os.path.splitext(parts[-1])[0]
                    output_filename = "_".join(parts[-3:]) + f"_{column}_counts.csv"
                    output_path = os.path.join(output_dir, output_filename)
                    if column == "pairs":
                        # Convert the list of pairs to a DataFrame and save as CSV
                        pd.DataFrame(
                            data, columns=["first_col_value", "last_col_value"]
                        ).to_csv(output_path, index=False)
                    else:
                        data.to_csv(output_path, index=False)

    def plot_histograms(self, column_counts, column_name):
        for metric, dfs in column_counts[column_name].items():
            plt.figure(figsize=(20, 6))

            for file, df in dfs:  # unpack the tuple
                # Extract the name of the parent directory of the current file
                legend_name = os.path.basename(os.path.dirname(os.path.dirname(file)))

                sns.histplot(
                    data=df,
                    x="value",
                    weights="count",
                    bins=30,
                    kde=False,
                    label=legend_name,
                )

            if column_name == "first_column":
                title = "Starting values"
            elif column_name == "last_column":
                title = "Final values"

            plt.title(f"Histogram for metric: {metric} - {title}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend(title="Directory", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    seed = Seed_Analysis(file_path="kp_test")
    pair_count = seed.count_unique_columns()
    # value_pairs_counts = seed.count_unique_value_pairs()
    seed.plot_histograms(pair_count, "first_column")
