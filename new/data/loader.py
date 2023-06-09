import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
import os
from omegaconf import OmegaConf
from multiprocessing import Pool
from pandarallel import pandarallel


class DataLoader:
    def __init__(self, config_path="new/config/data.yaml"):
        self.config = OmegaConf.load(config_path)
        self.hyperparameter_csv = pd.read_csv(self.config["hyperparameter"])

    def get_strategies(self) -> List[str]:
        return self.config["strategies"]

    def get_metrices(self) -> List[str]:
        return self.config["metrices"]

    def get_datasets(self) -> List[str]:
        return self.config["datasets"]

    def load_files_per_metric_and_dataset(self, metric: str, dataset: str):
        all_files: List[str] = [
            "kp_test/" + strat + "/" + dataset + "/" + metric
            for strat in self.config["strategies"]
        ]
        with Pool() as pool:
            results = pool.map(self.read_file, all_files)
        results = sorted(list(results), key=lambda x: x[0])
        return results

    def read_file(self, path: str) -> Tuple[str, pd.DataFrame]:
        """
        Function to return a single file and the name of the corresponding strategy.

        Paramters:
        ----------
        path : str
            The path of the strategie.

        Returns:
        --------
        name, csv_file : Tuple[str, pd.DataFrame]
            The tuple consisting of a string and a data frame.
        """
        df = pd.read_csv(path)
        df = df.iloc[:, :-1]
        return tuple([path.split("/")[1], df])

    @property
    def get_hyperparamter_csv(self) -> pd.DataFrame:
        return self.hyperparameter_csv

    def get_hyperparamter_tuples(self) -> Set[Tuple[int, int, int, int]]:
        frame: pd.DataFrame = self.get_hyperparamter_csv().copy()
        # locate important columns in the frame
        frame = frame[
            [
                "EXP_START_POINT",
                "EXP_BATCH_SIZE",
                "EXP_LEARNER_MODEL",
                "EXP_TRAIN_TEST_BUCKET_SIZE",
            ]
        ]
        return set(frame.parallel_apply(tuple, axis=1))


"""
if __name__ == "__main__":
    pandarallel.initialize(progress_bar=True)
    loader = DataLoader()

    for metric in loader.get_metrices():
        for dataset in loader.get_datasets():
            all_data = loader.load_files_per_metric_and_dataset(metric, dataset)

"""
