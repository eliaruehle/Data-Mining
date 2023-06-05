import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os
from omegaconf import OmegaConf


class DataLoader:
    def __init__(self, base_dir: str, config_path: str):
        self.base_dir = base_dir
        self.strategies = self.find_strategies()
        self.datasets = self.find_datasets()
        self.metrices = OmegaConf.load(config_path)["metrices"]

    def find_strategies(self):
        # navigate into the right directory
        if os.getcwd().split("/")[-1] != self.base_dir:
            os.chdir(os.getcwd() + "/" + self.base_dir)
        return sorted(
            [strat for strat in os.listdir() if strat[0].isupper()], key=str.lower
        )

    def find_datasets(self):
        if os.getcwd().split("/")[-1] not in self.strategies:
            os.chdir(os.getcwd() + "/" + self.strategies[0])
        return sorted([dset for dset in os.listdir()], key=str.lower)

    def find_metrics(self):
        return


if __name__ == "__main__":
    loader = DataLoader("kp_test/")
    # loader.get_strategies()
