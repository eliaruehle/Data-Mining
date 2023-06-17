"""
This script contains code to retrieve all the hyperparameters for which there are stable runs and which can be used for
methods like a mini-batch clustering.
"""
from typing import List, Set, Tuple
import os
import pandas as pd
import csv
import sys
import multiprocessing as mp


to_discard: List[str] = list()
all_hyper: pd.DataFrame = pd.read_csv("/scratch/ws/0/vime121c-db-project/Extrapolation/05_done_workload.csv")


def scrape_all_files(root: str) -> List[str]:
    if not os.path.exists(root):
        raise ValueError("File directory does not exist")
    files: List[str] = list()
    for root_dir, _, files_dir in os.walk(root):
        for file in files_dir:
            path = os.path.join(root_dir, file)
            files.append(path)
    return files


def get_hyperparameters_from_single_file(path: str) -> Tuple[str, Set[Tuple[int, int, int, int]]]:
    df = pd.merge(pd.read_csv(path), all_hyper, on="EXP_UNIQUE_ID")
    df = df[
        [
            "EXP_DATASET",
            "EXP_RANDOM_SEED",
            "EXP_START_POINT",
            "EXP_NUM_QUERIES",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
        ]
    ]
    result = set(df.apply(tuple, axis=1))
    if len(result) < 300:
        to_discard.append(path)
        return path, set()
    return path, result


def main(index: int):
    root: str = "/scratch/ws/0/vime121c-db-project/Extrapolation"
    #root: str = "../../kp_test_int/strategies"
    strategy = sorted([entry.name for entry in os.scandir(root) if entry.is_dir()])[index]
    print("Strategy name:", strategy)
    files = scrape_all_files(os.path.join(root, strategy))
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(get_hyperparameters_from_single_file, files)
    pool.close()
    results_filtered = [entry[1] for entry in results]
    print([entry[0] + "\n" for entry in results])
    results_final = set.union(*results_filtered)

    with open(f"/home/h9/elru535b/scratch/elru535b-workspace/Data-Mining/remake/results/hyperparameter_final_{index}.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "EXP_DATASET",
            "EXP_RANDOM_SEED",
            "EXP_START_POINT",
            "EXP_NUM_QUERIES",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
        ])
        for tuple_item in results_final:
            writer.writerow(tuple_item)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
    print("start main")
    main(index)

