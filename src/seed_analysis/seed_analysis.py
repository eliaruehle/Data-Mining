import concurrent.futures
import fnmatch
import os
from ast import literal_eval
from collections import Counter
from typing import Dict, List, Tuple
from collections import defaultdict
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
                    except Exception as exc:
                        print("%r generated an exception: %s" % (file, exc))
                    else:
                        data_frames[metric].append((file, df))
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
        df = df.drop(columns="EXP_UNIQUE_ID", errors="ignore")
        return df

    def count_unique_start_and_end_frequency(
        self,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Count the frequency of unique values in the first and last columns of data frames for each metric.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]:
                A dictionary of DataFrames for each metric and column (first_column or last_column),
                representing the frequency of unique values.
        """
        column_counts = {"first_column": {}, "last_column": {}}
        for metric, dfs in self.data_frames.items():
            first_col_freq = Counter()
            last_col_freq = Counter()

            # unpack the tuple
            for file, df in dfs:
                # get the first and last columns
                first_col = df.iloc[:, 0]
                last_col = df.iloc[:, -1]

                # count frequency
                first_col_freq += Counter(first_col)
                last_col_freq += Counter(last_col)

            # convert the Counter dict to a DataFrame
            first_col_df = pd.DataFrame.from_records(
                list(first_col_freq.items()), columns=["value", "count"]
            )

            last_col_df = pd.DataFrame.from_records(
                list(last_col_freq.items()), columns=["value", "count"]
            )

            # store a tuple with the filename and the DataFrame
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
                output_filename = f"{metric}_{column}_counts.csv"
                output_path = os.path.join(output_dir, output_filename)
                df.to_csv(output_path, index=False)

    def load_saved_csvs(self, input_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load the saved CSV files from a directory into a dictionary of data frames.

        Args:
            input_dir (str): The directory from which to load the CSV files.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: A dictionary of DataFrames for each metric and column (first_column or last_column).
        """

        data_frames = {"first_column": {}, "last_column": {}}

        # Create a list of csv files in the output directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

        for file in csv_files:
            # Define the complete file path
            file_path = os.path.join(input_dir, file)

            # Load the file into a DataFrame
            df = pd.read_csv(file_path)

            # Extract the metric name from the file name
            metric, _, _ = file.rpartition("_")

            # Store the DataFrame in the corresponding dictionary
            if "first_column" in file:
                data_frames["first_column"][metric] = df
            elif "last_column" in file:
                data_frames["last_column"][metric] = df

        return data_frames

    def count_unique_start_and_end_pair_frequency(self) -> Dict[str, pd.DataFrame]:
        """
        Count the frequency of unique pairs of values in the first and last columns of data frames for each metric.

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
                last_col = df.iloc[:, -1]
                total_row_sum = df.sum(axis=1)

                # count frequency of pairs
                for pair, sum_val in zip(zip(first_col, last_col), total_row_sum):
                    row_sum[pair].append(sum_val)
                    pairs_freq[(pair, sum_val)] += 1

            # create a DataFrame for each unique pair
            for pair, sums in row_sum.items():
                for sum_val in set(sums):
                    count = pairs_freq[(pair, sum_val)]
                    column_counts["pairs"].setdefault(metric, []).append(
                        {"value": pair, "total": sum_val, "count": count}
                    )

            if metric in column_counts["pairs"]:
                column_counts["pairs"][metric] = pd.DataFrame(
                    column_counts["pairs"][metric]
                )

        return column_counts

    def load_unique_pair_frequency(self, input_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load csv files of unique pair frequency data from a directory, convert them into DataFrames and store
        them in a dictionary where key is metric name and value is corresponding DataFrame.

        Args:
            input_dir (str): Directory path where csv files are located.

        Returns:
            Dict[str, DataFrame]: A dictionary containing DataFrames.
        """

        data_frames = {"pairs": {}}

        # Create a list of csv files in the output directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

        for file in csv_files:
            # Define the complete file path
            file_path = os.path.join(input_dir, file)

            # Load the file into a DataFrame
            df = pd.read_csv(file_path)

            # Extract the metric name from the file name
            metric, _, _ = file.rpartition("_")

            # Store the DataFrame in the corresponding dictionary
            if "pairs" in file:
                data_frames["pairs"][metric] = df

        return data_frames

    def filter_pairs(
        self,
        column_counts: Dict[str, pd.DataFrame],
        threshold_first: float,
        threshold_last: float,
        operator: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Filters the pairs based on the provided thresholds and operator.

        Args:
            column_counts (Dict[str, DataFrame]): Dictionary containing DataFrames with pair counts.
            threshold_first (float): Threshold for the first value in pair.
            threshold_last (float): Threshold for the second value in pair.
            operator (str): Comparison operator to use.

        Returns:
            Dict[str, DataFrame]: Dictionary containing filtered DataFrames.
        """
        allowed_operators = [
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "less_and_equal",
            "less_equal_and_equal",
            "equal_and_less",
            "equal_and_less_equal",
            "less_and_greater_equal",
        ]

        filtered_counts = {"pairs": {}}
        for metric, df in column_counts["pairs"].items():
            filtered_rows = []

            for _, row in df.iterrows():
                # Convert the string to a tuple of floats
                pair = literal_eval(row["value"])

                match operator:
                    case "less":
                        condition = (
                            pair[0] < threshold_first and pair[1] < threshold_last
                        )
                    case "less_equal":
                        condition = (
                            pair[0] <= threshold_first and pair[1] <= threshold_last
                        )
                    case "greater":
                        condition = (
                            pair[0] > threshold_first and pair[1] > threshold_last
                        )
                    case "greater_equal":
                        condition = (
                            pair[0] >= threshold_first and pair[1] >= threshold_last
                        )
                    case "less_and_equal":
                        condition = (
                            pair[0] < threshold_first and pair[1] == threshold_last
                        )
                    case "less_equal_and_equal":
                        condition = (
                            pair[0] <= threshold_first and pair[1] == threshold_last
                        )
                    case "equal_and_less":
                        condition = (
                            pair[0] == threshold_first and pair[1] < threshold_last
                        )
                    case "equal_and_less_equal":
                        condition = (
                            pair[0] == threshold_first and pair[1] <= threshold_last
                        )
                    case "less_and_greater_equal":
                        condition = (
                            pair[0] < threshold_first and pair[1] >= threshold_last
                        )

                    case _:
                        raise ValueError(
                            f"Unknown operator value: '{operator}'. Expected: '{allowed_operators}'"
                        )

                if condition:
                    filtered_rows.append(row)

            # Convert the list of filtered rows back to a DataFrame
            filtered_df = pd.DataFrame(filtered_rows)

            # store the filtered DataFrame
            filtered_counts["pairs"][metric] = filtered_df

        return filtered_counts

    def filter_top_k_pairs(
        self,
        column_counts: Dict[str, pd.DataFrame],
        top_k: int,
        threshold: float,
    ) -> Dict[str, pd.DataFrame]:
        """
        Filters the pairs based on the top_k results and a minimum threshold for the second value in a pair.

        Args:
            column_counts (Dict[str, DataFrame]): Dictionary containing DataFrames with pair counts.
            top_k (int): Minimum cumulative sum of counts to reach.
            threshold (float): Minimum value for the second value in a pair.

        Returns:
            Dict[str, DataFrame]: Dictionary containing filtered DataFrames.
        """

        filtered_counts = {"pairs": {}}
        for metric, df in column_counts["pairs"].items():
            # Convert the string representation of tuple to actual tuple
            df["value"] = df["value"].apply(lambda x: literal_eval(x))

            # Split the tuples into two separate columns
            df[["first_value", "second_value"]] = pd.DataFrame(
                df["value"].tolist(), index=df.index
            )

            # Only keep rows where the second value of the tuple is greater than or equal to the threshold
            df = df[df["second_value"] >= threshold]

            # Sort DataFrame by the second value in descending order
            df = df.sort_values(by=["second_value", "count"], ascending=False)

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
            filtered_df = filtered_df[["value", "count", "cumulative_sum"]]

            # store the filtered DataFrame
            filtered_counts["pairs"][metric] = filtered_df

        return filtered_counts

    def save_filtered_pairs_to_csv(
        self, output_dir: str, filtered_counts: Dict[str, pd.DataFrame]
    ):
        """
        Save the filtered counts to csv files.

        Args:
            output_dir (str): Directory path where csv files will be saved.
            filtered_counts (Dict[str, DataFrame]): Dictionary containing DataFrames with filtered counts.
        """

        os.makedirs(output_dir, exist_ok=True)
        for metric, df in filtered_counts["pairs"].items():
            output_filename = f"{metric}_filtered_pairs.csv"
            output_path = os.path.join(output_dir, output_filename)
            df.to_csv(output_path, index=False)

    def save_top_k_pairs_to_csv(
        self, filtered_counts: Dict[str, pd.DataFrame], output_dir: str
    ) -> None:
        """
        Save the filtered pairs to csv files.

        Args:
            filtered_counts (Dict[str, DataFrame]): Dictionary containing DataFrames with filtered counts.
            output_dir (str): The directory where to save the csv files.
        """
        for metric, df in filtered_counts["pairs"].items():
            output_path = os.path.join(output_dir, f"top_k_{metric}.csv")
            df.to_csv(output_path, index=False)

    def plot_histograms(self, column_counts: Dict[str, pd.DataFrame], column_name: str):
        """
        Plot histograms of the column counts.

        Args:
            column_counts (Dict[str, DataFrame]): Dictionary containing DataFrames with column counts.
            column_name (str): Name of the column to be plotted.
        """

        for metric, df in column_counts[column_name].items():
            plt.figure(figsize=(20, 6))

            # Extract the name of the parent directory of the current file
            legend_name = metric

            plot = sns.histplot(
                data=df,
                x="value",
                weights="count",
                bins=30,
                kde=False,
                label=legend_name,
            )

            total = float(df["count"].sum())
            for p in plot.patches:
                percentage = "{:.1f}%".format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2
                y = p.get_y() + p.get_height()
                plot.annotate(percentage, (x, y), size=12, ha="center", va="bottom")

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

    def plot_histograms_filtered_pairs(self, filtered_counts: Dict[str, pd.DataFrame]):
        """
        Plot histograms of the filtered counts for the first and second value in the pair in separate subplots.

        Args:
            filtered_counts (Dict[str, DataFrame]): Dictionary containing DataFrames with filtered counts.
        """

        for metric, df in filtered_counts["pairs"].items():
            # Convert the string representation of tuple to actual tuple
            df["value"] = df["value"].apply(lambda x: literal_eval(x))

            # Split the tuples into two separate columns
            df[["first_value", "second_value"]] = pd.DataFrame(
                df["value"].tolist(), index=df.index
            )

            # Create figure with two subplots
            fig, axes = plt.subplots(nrows=2, figsize=(20, 12))

            # Plot histogram of the "second_value" in the first subplot
            plot = sns.histplot(
                data=df,
                x="second_value",
                weights="count",
                bins=30,
                kde=False,
                label=metric,
                ax=axes[0],  # assign this plot to the first subplot
            )

            total = float(df["count"].sum())
            for p in plot.patches:
                percentage = "{:.1f}%".format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2
                y = p.get_y() + p.get_height()
                plot.annotate(percentage, (x, y), size=12, ha="center", va="bottom")

            # Labels and title for the first plot
            plot.set_title(f"Histogram for second values: {metric}")
            plot.set_xlabel("Second Value")
            plot.set_ylabel("Frequency")

            # Plot histogram of "first_value" in the second subplot
            plot = sns.histplot(
                data=df,
                x="first_value",
                weights="count",
                bins=30,
                kde=False,
                label=metric,
                ax=axes[1],  # assign this plot to the second subplot
            )

            total = float(df["count"].sum())
            for p in plot.patches:
                percentage = "{:.1f}%".format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2
                y = p.get_y() + p.get_height()
                plot.annotate(percentage, (x, y), size=12, ha="center", va="bottom")

            # Labels and title for the second plot
            plot.set_title(f"Histogram for first values: {metric}")
            plot.set_xlabel("First Value")
            plot.set_ylabel("Frequency")

            # Show the plots
            plt.tight_layout()
            plt.show()

    def plot_histograms_top_k_pairs(
        self, filtered_counts: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Plot histograms of the filtered counts.

        Args:
            filtered_counts (Dict[str, DataFrame]): Dictionary containing DataFrames with filtered counts.
        """
        max_cumulative_sum = 0
        for metric, df in filtered_counts["pairs"].items():
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
                label=metric,
            )
            total = float(df["count"].sum())
            for p in plot.patches:
                percentage = "{:.1f}%".format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2
                y = p.get_y() + p.get_height()
                plot.annotate(percentage, (x, y), size=12, ha="center", va="bottom")

            plt.title(f"Histogram for top-k pairs ({max_cumulative_sum}): {metric}")
            plt.xlabel("Starting Values")
            plt.ylabel("Frequency")
            plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.show()


def run(
    threshold_first: float,
    threshold_last: float,
    operator: str,
    hpc=False,
    save_filtered=False,
    plot=False,
    plot_filtered=False,
):
    if not 0 <= threshold_first <= 1:
        raise ValueError(
            f"The threshold of 'threshhold_first' can only be between [0, 1], provided: '{threshold_first}'"
        )

    if not 0 <= threshold_last <= 1:
        raise ValueError(
            f"The threshold of 'threshhold_last' can only be between [0, 1], provided: '{threshold_last}'"
        )

    allowed_operators = [
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "less_and_equal",
        "less_equal_and_equal",
        "equal_and_less",
        "equal_and_less_equal",
        "less_and_greater_equal",
    ]

    if operator not in allowed_operators:
        raise ValueError(
            f"The provided operator '{operator}' is not recognised, available operators are: '{allowed_operators}' "
        )

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
            input_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/columns"
        )
        pair_count = seed.load_unique_pair_frequency(
            input_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/column_pairs/"
        )

        filtered_pair_count = seed.filter_pairs(
            column_counts=pair_count,
            threshold_first=threshold_first,
            threshold_last=threshold_last,
            operator=operator,
        )

        if save_filtered:
            seed.save_filtered_pairs_to_csv(
                output_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/column_pairs_filtered",
                filtered_counts=filtered_pair_count,
            )

        if plot:
            seed.plot_histograms(
                column_counts=start_end_count, column_name="first_column"
            )

        if plot_filtered:
            seed.plot_histograms_filtered_pairs(filtered_counts=filtered_pair_count)


if __name__ == "__main__":
    # run(
    #     threshold_first=0,
    #     threshold_last=0,
    #     operator="greater_equal",
    #     save_filtered=True,
    #     plot=False,
    #     plot_filtered=True,
    # )
    seed = Seed_Analysis(file_path="/Users/user/GitHub/Data-Mining/kp_test/strategies")
    pair_counts = seed.load_unique_pair_frequency(
        input_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/column_pairs"
    )
    top_k = seed.filter_top_k_pairs(
        column_counts=pair_counts, top_k=50000, threshold=0.99
    )
    seed.save_top_k_pairs_to_csv(
        output_dir="/Users/user/GitHub/Data-Mining/src/seed_analysis/results/top_k",
        filtered_counts=top_k,
    )
    seed.plot_histograms_top_k_pairs(top_k)
