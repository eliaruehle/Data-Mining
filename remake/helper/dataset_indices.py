import os
import pandas as pd
from typing import List
from omegaconf import OmegaConf, dictconfig
import csv
from collections import defaultdict

def main() -> None:
    """
    Main function to call the other helpers.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    # get_dataset_index()
    get_dataset_to_hyper_dict()
    #get_valuable_metrices()

def get_dataset_to_hyper_dict() -> None:
    """
    Retrieves a dictionary which maps datasets to a set of hyperparameters sampled on them.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    config_path: str = "../assets/datasets_ind.yaml"
    hyper_root: str = "../results/all_hyperparameters.csv"
    with open(hyper_root, "r") as csv_file:
        reader = csv.reader(csv_file)
        hypers = list(reader)
        hypers = [list(map(lambda x: int(x), entry)) for entry in hypers[1:]]
    config_dict = dict(OmegaConf.load(config_path))
    mapping_dict = defaultdict(list)
    for entry in hypers:
        mapping_dict[config_dict[entry[0]]].append(entry[1:])
    final = dict(mapping_dict)
    print("DATASETS:")
    for entry in final.keys():
        print(f"- {entry}")
    new_config = OmegaConf.create(final)
    OmegaConf.save(new_config, "../assets/dataset_hyper.yaml")


def get_dataset_index():
    """
    Function to get the index of datasets regarding project sampling.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
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

def get_valuable_metrices():
    """
    Function to get metrics with useful data.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    root: str = "../../kp_test_int/strategies/ALIPY_UNCERTAINTY_LC/appendicitis"
    files: List[str] = [os.path.join(root, file) for file in os.listdir(root)]
    for file in files:
        df = pd.read_csv(file)
        if df.shape[1] < 99:
            continue
        else:
            print("- " + file.split("/")[-1])


if __name__ == "__main__":
    # calls the main function
    main()