import csv
from typing import Set, List
import pandas as pd
import os
import multiprocessing as mp


def process_file(path:str) -> Set:
    if os.path.exists(path):
        df = pd.read_csv(path)
        tuples = set(df.apply(tuple, axis=1))
        print("number entries: ", len(tuples), path)
        #if len(tuples) < 25000:
        #    return set()
        return tuples
    else:
        raise ValueError("Path not exists!")

def main() -> None:
    root = "../results/"
    if os.path.exists(root):
        files: List[str] = sorted([os.path.join(root, file.name) for file in os.scandir(root) if not file.is_dir() and file.name != "all_hyperparameters.csv"])
    else:
        raise ValueError("Path not exists!")

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_file, files)
    pool.close()
    final = [sets for sets in results if sets]
    final = set.intersection(*final)
    final_datasets = set([entry[0] for entry in final])
    print(final_datasets)

    with open(os.path.join(root, "all_hyperparameters.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "EXP_DATASET",
            "EXP_RANDOM_SEED",
            "EXP_START_POINT",
            "EXP_NUM_QUERIES",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
        ])
        for tuple_item in final:
            writer.writerow(tuple_item)


if __name__ == "__main__":
    main()