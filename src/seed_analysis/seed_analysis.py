import concurrent.futures
import fnmatch
import os
from ast import literal_eval
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Seed_Analysis:
    """
    A class used to perform various analysis and visualizations on seed data.


    Attributes
    ----------
    file_path : str
        The path to the directory containing the .csv.xz files
    csv_files : dict
        A dictionary where the keys are the metrics and the values are a list of all the .csv.xz files for that metric
    data_frames : dict
        A dictionary where the keys are the metrics and the values are a list of tuples. Each tuple contains the file name and the corresponding dataframe

    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.csv_files = self.find_csv_files()
        self.done_workload = self.load_done_workload()
        self.data_frames = self.load_data_frames()

    def find_csv_files(self) -> Dict[str, List[str]]:
        """
        Finds CSV files with specific metrics in the file path directory.

        Returns:
            Dict[str, List[str]]: A dictionary where the keys are metrics and the values are lists of file paths.
        """

        csv_files = {}
        metrics = [
            "accuracy",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1-score",
        ]

        for root, dirs, files in os.walk(self.file_path):
            for metric in metrics:
                for filename in sorted(fnmatch.filter(files, f"{metric}.csv.xz")):
                    if metric not in csv_files:
                        csv_files[metric] = []
                    csv_files[metric].append(os.path.join(root, filename))

        return csv_files

    def load_done_workload(self):
        done_workload_file = os.path.join(self.file_path, "05_done_workload.csv")

        if not os.path.exists(done_workload_file):
            raise FileNotFoundError(f"The file '{done_workload_file}' does not exist.")

        done_workload = pd.read_csv(
            done_workload_file,
            usecols=["EXP_UNIQUE_ID", "EXP_BATCH_SIZE"],
        )

        done_workload = done_workload.drop_duplicates()
        return done_workload

    def load_data_frames(self) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
        """
        Load data frames concurrently for CSV files associated with each metric.

        Returns:
            Dict[str, List[Tuple[str, pd.DataFrame]]]: A dictionary where the keys are metrics and the values are lists of tuples containing file path and the loaded DataFrame.
        """
        data_frames = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for metric, files in self.csv_files.items():
                future_to_file = {
                    executor.submit(self._load_dataframe_helper, file): file
                    for file in files
                }
                data_frames[metric] = []
                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        df = future.result()
                        merged_df = self.merge_dataframes(df)
                    except Exception as exc:
                        print("%r generated an exception: %s" % (file, exc))
                    else:
                        data_frames[metric].append((file, merged_df))
            return data_frames

    def _load_dataframe_helper(self, file: str) -> pd.DataFrame:
        """
        Helper function to load a data frame from a CSV file and drop a specific column.

        Args:
            file (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        df = pd.read_csv(file, compression="xz")
        return df

    def merge_dataframes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge input DataFrame with done_workload and replace the matching "EXP_UNIQUE_ID" with the Value from the "EXP_BATCH_SIZE" column. Then rename the column to "batch_size".

        Args:
            df (pd.DataFrame): DataFrame to be merged.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """

        if "EXP_UNIQUE_ID" not in df.columns:
            raise KeyError(
                "Column 'EXP_UNIQUE_ID' not found in the provided DataFrame."
            )

        merged_df = pd.merge(self.done_workload, df, on="EXP_UNIQUE_ID", how="inner")

        # Drop the 'EXP_UNIQUE_ID' column
        merged_df = merged_df.drop(columns=["EXP_UNIQUE_ID"])

        # Rename the 'EXP_BATCH_SIZE' column to 'batch_size' and move it to the end
        batch_size = merged_df.pop("EXP_BATCH_SIZE")
        merged_df = merged_df.assign(batch_size=batch_size)

        return merged_df

    def count_unique_start_and_end_frequency(
        self,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Count the frequency of unique combinations of the first/last column values and batch sizes.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]:
                A dictionary of DataFrames for each metric and column (first_column or last_column),
                representing the frequency of unique combinations.
        """
        column_counts = {"first_column": {}, "last_column": {}}
        for metric, dfs in self.data_frames.items():
            first_col_freq = Counter()
            last_col_freq = Counter()

            # unpack the tuple
            for file, df in dfs:
                # get the first and last columns
                first_col = df.iloc[:, 0]
                last_col = df.iloc[:, -2]  # -2 Because -1 is "batch_size" column
                batch_size = df["batch_size"]

                # count frequency considering the batch size as a separate value
                for value, size in zip(first_col, batch_size):
                    first_col_freq[(value, size)] += 1
                for value, size in zip(last_col, batch_size):
                    last_col_freq[(value, size)] += 1

            # convert the Counter dict to a DataFrame
            first_col_df = pd.DataFrame.from_records(
                [(*k, v) for k, v in first_col_freq.items()],
                columns=["value", "batch_size", "count"],
            )
            last_col_df = pd.DataFrame.from_records(
                [(*k, v) for k, v in last_col_freq.items()],
                columns=["value", "batch_size", "count"],
            )

            # store the DataFrame
            column_counts["first_column"][metric] = first_col_df
            column_counts["last_column"][metric] = last_col_df

        return column_counts

    def save_unique_start_and_end_frequency(
        self, output_dir: str, column_counts: Dict[str, Dict[str, pd.DataFrame]]
    ):
        """
        Save the count of unique start and end frequency as CSV files.

        Args:
            output_dir (str): The directory where to save the CSV files.
            column_counts (Dict[str, Dict[str, pd.DataFrame]]): The dictionary with the counts to save.
        """

        os.makedirs(output_dir, exist_ok=True)
        for column, metrics in column_counts.items():
            for metric, df in metrics.items():
                for batch_size in df["batch_size"].unique():
                    df_filtered = df[df["batch_size"] == batch_size]
                    output_filename = (
                        f"{metric}_{column}_counts_batch_size_{batch_size}.csv"
                    )
                    output_path = os.path.join(output_dir, output_filename)
                    df_filtered.to_csv(output_path, index=False)

    def load_saved_csvs(
        self, input_dir: str
    ) -> Dict[str, Dict[str, Dict[int, pd.DataFrame]]]:
        """
        Load the saved CSV files from a directory into a dictionary of data frames.

        Args:
            input_dir (str): The directory from which to load the CSV files.

        Returns:
            Dict[str, Dict[str, Dict[int, pd.DataFrame]]]: A dictionary of DataFrames for each metric, column, and batch_size.
        """

        data_frames = defaultdict(lambda: defaultdict(dict))

        # Create a list of csv files in the output directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

        for file in csv_files:
            # Define the complete file path
            file_path = os.path.join(input_dir, file)

            # Load the file into a DataFrame
            df = pd.read_csv(file_path)

            # Extract the metric, column, and batch_size from the file name
            file_name = file.replace(".csv", "")
            metric, column = file_name.rsplit("_", 2)[:2]
            batch_size = int(file_name.rsplit("_", 2)[-1])

            # Print out some debugging information to make sure everything is working correctly
            # print(
            #     f"Loading file '{file}' with parsed info - Metric: {metric}, Column: {column}, Batch size: {batch_size}"
            # )

            # Store the DataFrame in the corresponding dictionary
            data_frames[metric][column][batch_size] = df

        return data_frames

    def count_unique_start_and_end_pair_frequency(self) -> Dict[str, pd.DataFrame]:
        """
        Count the frequency of unique pairs of values in the first and last columns of data frames for each metric along with the batch size.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames for each metric, representing the frequency of unique pairs of values.
        """
        column_counts = {"pairs": {}}

        for metric, dfs in self.data_frames.items():
            row_sum = defaultdict(list)
            pairs_freq = Counter()

            # unpack the tuple
            for file, df in dfs:
                # get the first and last columns
                first_col = df.iloc[:, 0]
                # -2 because the last column is now "batch_size"
                last_col = df.iloc[:, -2]
                batch_size = df["batch_size"]
                total_row_sum = df.iloc[:, :-1].sum(axis=1)

                # count frequency of pairs along with the batch size
                for pair, sum_val, size in zip(
                    zip(first_col, last_col), total_row_sum, batch_size
                ):
                    row_sum[(pair, size)].append(sum_val)
                    pairs_freq[((pair, size), sum_val)] += 1

            # create a DataFrame for each unique pair and batch size
            for (pair, size), sums in row_sum.items():
                for sum_val in set(sums):
                    count = pairs_freq[((pair, size), sum_val)]
                    column_counts["pairs"].setdefault(metric, []).append(
                        {
                            "value": pair,
                            "total": sum_val,
                            "count": count,
                            "batch_size": size,
                        }
                    )

            if metric in column_counts["pairs"]:
                column_counts["pairs"][metric] = pd.DataFrame(
                    column_counts["pairs"][metric]
                )

        return column_counts

    def load_unique_pair_frequency(
        self, input_dir: str
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Load csv files of unique pair frequency data from a directory, convert them into DataFrames and store
        them in a dictionary where key is metric name and value is corresponding DataFrame.

        Args:
            input_dir (str): Directory path where csv files are located.

        Returns:
            Dict[str, Dict[int, DataFrame]]: A dictionary containing DataFrames.
        """

        data_frames = {}

        # Create a list of csv files in the output directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

        for file in csv_files:
            # Define the complete file path
            file_path = os.path.join(input_dir, file)

            # Load the file into a DataFrame
            df = pd.read_csv(file_path)

            # Extract the metric name and batch size from the file name
            metric, _, batch_size_str = file.replace(".csv", "").rpartition("_")

            batch_size = int(batch_size_str)

            # Store the DataFrame in the corresponding dictionary
            if metric not in data_frames:
                data_frames[metric] = {}

            data_frames[metric][batch_size] = df

        return data_frames

    def filter_top_k_pairs(
        self,
        column_counts: Dict[str, Dict[int, pd.DataFrame]],
        top_k: int,
        threshold: float,
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Filters the pairs based on the top_k results and a minimum threshold for the second value in a pair.

        Args:
            column_counts (Dict[str, Dict[int, DataFrame]]): Dictionary containing DataFrames with pair counts.
            top_k (int): Minimum cumulative sum of counts to reach.
            threshold (float): Minimum value for the second value in a pair.

        Returns:
            Dict[str, Dict[int, DataFrame]]: Dictionary containing filtered DataFrames.
        """

        filtered_counts = {}

        for metric, batch_sizes in column_counts.items():
            filtered_counts[metric] = {}

            for batch_size, df in batch_sizes.items():
                # Convert the string representation of tuple to actual tuple
                df["value"] = df["value"].apply(literal_eval)

                # Split the tuples into two separate columns
                df[["first_value", "second_value"]] = pd.DataFrame(
                    df["value"].tolist(), index=df.index
                )

                # Only keep rows where the second value of the tuple is greater than or equal to the threshold
                df = df[df["second_value"] >= threshold]

                # Sort DataFrame by the second value in descending order
                df = df.sort_values(by=["total"], ascending=False)

                # Initialize cumulative sum
                cumulative_sum = 0
                filtered_rows = []

                # Iterate over the DataFrame rows
                for _, row in df.iterrows():
                    # If the cumulative sum is less than top_k, add the row
                    if cumulative_sum < top_k:
                        cumulative_sum += row["count"]
                        row["cumulative_sum"] = cumulative_sum
                        filtered_rows.append(row)
                    else:
                        break  # If the cumulative sum has reached top_k, stop adding rows

                # Convert the list of filtered rows back to a DataFrame
                filtered_df = pd.DataFrame(filtered_rows)

                # Retain only the necessary columns
                filtered_df = filtered_df[["value", "count", "total", "cumulative_sum"]]

                # Store the filtered DataFrame
                filtered_counts[metric][batch_size] = filtered_df

        return filtered_counts

    def save_top_k_to_csv(
        self, output_dir: str, filtered_counts: Dict[str, Dict[int, pd.DataFrame]]
    ):
        """
        Save the filtered counts to csv files.

        Args:
            output_dir (str): Directory path where csv files will be saved.
            filtered_counts (Dict[str, Dict[int, DataFrame]]): Dictionary containing DataFrames with filtered counts.
        """

        os.makedirs(output_dir, exist_ok=True)

        for metric, batch_sizes in filtered_counts.items():
            for batch_size, df in batch_sizes.items():
                output_filename = f"{metric}_top_k_{batch_size}.csv"
                output_path = os.path.join(output_dir, output_filename)
                df.to_csv(output_path, index=False)

    def plot_histograms_batchsize(
        self,
        column_counts: Dict[str, Dict[str, Dict[int, pd.DataFrame]]],
        column_name: str,
        output_path: Optional[str] = None,
    ):
        """
        Plot histograms of the column counts for each batch size separately.

        Args:
            column_counts (Dict[str, Dict[str, Dict[int, pd.DataFrame]]]): Nested dictionary containing DataFrames with column counts.
            column_name (str): Name of the column to be plotted.
        """
        if column_name not in ["first", "last"]:
            raise KeyError(
                f"The provided column name '{column_name}' does not exist. Please use: 'first' or 'last'"
            )

        # Loop over the different metrics
        for metric, data in column_counts.items():
            # Loop over the different columns and batch sizes for each metric
            for column, batch_sizes in data.items():
                if column == column_name:
                    # Loop over batch sizes
                    for batch_size, df in batch_sizes.items():
                        # Create a new figure for each batch size
                        plt.figure(figsize=(20, 6))

                        # Create a label for the current batch size
                        batch_size = f"Batch size {batch_size}"

                        # Calculate total count for percentage calculation
                        total_count = df["count"].sum()

                        # Plot the histogram for the current batch size
                        plot = sns.histplot(
                            data=df,
                            x="value",
                            weights="count",
                            bins=30,
                            label=batch_size,
                        )

                        # Add percentage on top of each bar
                        for p in plot.patches:
                            height = p.get_height()
                            plot.text(
                                p.get_x() + p.get_width() / 2.0,
                                height + 3,
                                "{:1.2f}%".format(height / total_count * 100),
                                ha="center",
                                fontsize=10,
                            )

                        # Set the title and labels for each subplot
                        plt.title(f"{metric} Histogram for Final Values - {batch_size}")
                        plt.xlabel("Value")
                        plt.ylabel("Frequency")

                        # Save the figure if output_path is not None, otherwise show the plot window
                        if output_path is not None:
                            plt.savefig(
                                f"{output_path}/{metric}_batch_{batch_size}.svg",
                                format="svg",
                            )
                        else:
                            plt.show()

    def plot_histograms_top_k_pairs(
        self,
        filtered_counts: Dict[str, Dict[int, pd.DataFrame]],
        output_path: Optional[str] = None,
    ) -> None:
        """
        Plot histograms of the filtered counts.

        Args:
            filtered_counts (Dict[str, Dict[int, DataFrame]]): Dictionary containing DataFrames with filtered counts.
        """
        max_cumulative_sum = 0

        for metric, batch_sizes in filtered_counts.items():
            for batch_size, df in batch_sizes.items():
                max_cumulative_sum = df["cumulative_sum"].max()
                # Split the tuples into two separate columns
                df[["first_value", "second_value"]] = pd.DataFrame(
                    df["value"].tolist(), index=df.index
                )

                # Plot histogram of the "first_value"
                plt.figure(figsize=(20, 6))
                plot = sns.histplot(
                    data=df,
                    x="first_value",
                    weights="count",
                    bins=30,
                    kde=False,
                    label=f"Batch size {batch_size}",
                )
                total = float(df["count"].sum())
                for p in plot.patches:
                    percentage = "{:.1f}%".format(100 * p.get_height() / total)
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_y() + p.get_height()
                    plot.annotate(percentage, (x, y), size=12, ha="center", va="bottom")

                plt.title(
                    f"Histogram for top-k pairs ({max_cumulative_sum}): {metric} - Batch size {batch_size}"
                )
                plt.xlabel("Starting Values")
                plt.ylabel("Frequency")
                plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()

                # Save the figure if output_path is not None, otherwise show the plot window
                if output_path is not None:
                    plt.savefig(
                        f"{output_path}/{metric}_batch_{batch_size}.svg", format="svg"
                    )
                else:
                    plt.show()


def run(hpc=False, plot_start_end=False, plot_top_k=False):
    seed = Seed_Analysis(file_path="/Users/user/GitHub/Data-Mining/kp_test/strategies")

    if hpc:
        start_end_count = seed.count_unique_start_and_end_frequency()
        pair_count = seed.count_unique_start_and_end_pair_frequency()

        seed.save_unique_start_and_end_frequency(
            output_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/test_columns",
            column_counts=start_end_count,
        )

        seed.save_unique_start_and_end_frequency(
            output_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/test_columns_pairs",
            column_counts=pair_count,
        )
    else:
        start_end_count = seed.load_saved_csvs(
            input_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/batch"
        )
        if plot_start_end:
            # use 'first' or 'last' to show distribution for starting / final values.
            seed.plot_histograms_batchsize(start_end_count, "first")

        # pairs = seed.load_unique_pair_frequency(
        #     input_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/batch_pairs"
        # )
        # top = seed.filter_top_k_pairs(column_counts=pairs, top_k=20000, threshold=0)
        # seed.save_top_k_to_csv(
        #     output_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/top_k",
        #     filtered_counts=top,
        # )
        # if plot_top_k:
        #     seed.plot_histograms_top_k_pairs(filtered_counts=top)


if __name__ == "__main__":
    run(plot_start_end=True)
