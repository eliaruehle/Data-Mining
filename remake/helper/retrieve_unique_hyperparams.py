"""
This script contains code to retrieve all the hyperparameters for which there are stable runs and which can be used for
methods like a mini-batch clustering.
"""
import sys
from typing import List, Set, Tuple
import os
import pandas as pd
import csv
import time
import sys
import multiprocessing as mp


to_discard: List[str] = list()
all_hyper: pd.DataFrame = pd.read_csv("../05_done_workload.csv")


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
    df = pd.merge(pd.read_csv(path), all_hyper, on="EXP_UNIQUE_ID")
    df = df[
        [
            "EXP_START_POINT",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
        ]
    ]
    result = set(df.apply(tuple, axis=1))
    if len(result) < 300:
        to_discard.append(path)
        return set()
    return result


def main(index: int):
    print("in main")
    root: str = "/home/vime121c/Workspaces/scratch/vime121c-db-project/Extrapolation"
    #root: str = "../../kp_test_int/strategies"
    strategy = sorted([entry.name for entry in os.scandir(root) if entry.is_dir()])[index]
    files = scrape_all_files(os.path.join(root, strategy))
    start = time.time()
    print("start loop")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(get_hyperparameters_from_single_file, files)
    pool.close()
    results_filtered = [sets for sets in results if sets]
    print(f"Finished in {time.time()-start} sec.")
    results_final = set.intersection(*results_filtered)

    with open(f"../results/hyperparameter_final_{index}.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["EXP_START_POINT", "EXP_BATCH_SIZE", "EXP_LEARNER_MODEL", "EXP_TRAIN_TEST_BUCKET_SIZE"])
        for tuple_item in results_final:
            writer.writerow(tuple_item)

    try:
        with open("../results/to_discard.csv", "w", newline="") as discard:
            writer = csv.writer(discard)
            writer.writerow(["PATH"])
            for discard_path in to_discard:
                writer.writerow(discard_path)
    except:
        print("Error occurred")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
    print("start main")
    main(index)

