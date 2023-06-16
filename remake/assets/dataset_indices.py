import os
import pandas as pd
from typing import List

def main() -> None:
    root: str = "../../kp_test_int/strategies/ALIPY_UNCERTAINTY_LC"
    workload: pd.DataFrame = pd.read_csv("../05_done_workload.csv")
    datasets: List[str] = list(os.listdir(root))
    # use the first metric
    to_read: List[str] = [os.path.join(root, dataset + "/accuracy.csv.xz") for dataset in datasets]

    save: List = list()
    for file in to_read:
        df = pd.merge(pd.read_csv(file), workload, on="EXP_UNIQUE_ID")
        save.append(tuple([df["EXP_DATASET"][1], file.split("/")[-2]]))

    save: List[str] = sorted(save, key=lambda x: x[0])
    for entry in save:
        # a pretty print
        print(f"{entry[0]} : {entry[1]}")

