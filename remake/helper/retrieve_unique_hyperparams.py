"""
This script contains code to retrieve all the hyperparameters for which there are stable runs and which can be used for
methods like a mini-batch clustering.
"""
from multiprocessing import Pool, current_process
from typing import List, Set, Tuple
import os
import pandas as pd
from pandarallel import pandarallel
import csv
import time


all_hyper: pd.DataFrame = pd.read_csv("../05_done_workload.csv")
to_discard: List[str] = list()


def scrape_all_files(root: str) -> List[str]:
    if not os.path.exists(root):
        raise ValueError("File directory does not exist")
    files: List[str] = list()
    for root_dir, _, files_dir in os.walk(root):
        for file in files_dir:
            path = os.path.join(root_dir, file)
            files.append(path)
    return files


def get_hyperparameters_from_single_file(path: str) -> Set[Tuple[int, int, int, int]]:
    pandarallel.initialize(progress_bar=False, verbose=False)
    df = pd.merge(pd.read_csv(path), all_hyper, on="EXP_UNIQUE_ID")
    df = df[
        [
            "EXP_START_POINT",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
        ]
    ]
    current_process().daemon = False
    result = set(df.parallel_apply(tuple, axis=1))
    if len(result) < 300:
        to_discard.append(path)
        return set()
    current_process().daemon = True
    return result


def main():
    root: str = "../../kp_test_int/strategies"
    files = scrape_all_files(root)
    current_process().daemon = False
    start = time.time()
    print("Starting Pool!")
    print(len(files))
    with Pool() as pool:
        results = pool.map(get_hyperparameters_from_single_file, files)
    results_filtered = [sets for sets in results if sets]
    print(f"Finished in {time.time()-start} sec.")
    print("have filtered results")
    results_final = set.intersection(*results_filtered)
    print("start writing back")
    with open("../results/hyperparameter_final.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["EXP_START_POINT", "EXP_BATCH_SIZE", "EXP_LEARNER_MODEL", "EXP_TRAIN_TEST_BUCKET_SIZE"])
        for tuple_item in results_final:
            writer.writerow(tuple_item)

    with open("../results/to_discard.csv", "w", newline="") as discard:
        writer = csv.writer(discard)
        writer.writerow(["PATH"])
        for discard_path in to_discard:
            writer.writerow(discard_path)


if __name__ == "__main__":
    main()
