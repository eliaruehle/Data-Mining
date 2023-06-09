from pandarallel import pandarallel
from typing import List, Tuple, Set
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
import time


class Data:
    """
    The data base class.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.base_dir)
            for file in files
            if file.endswith(".csv.xz")
        ]
        self.hyperparamters = pd.read_csv("new/05_done_workload.csv")
        self.all_frames = dict(self.read_all_files())

    def read_file_hyp(self, path: str):
        return tuple(
            [path, pd.merge(pd.read_csv(path), self.hyperparamters, on="EXP_UNIQUE_ID")]
        )

    def read_file(self, path: str):
        return tuple([path, pd.read_csv(path)])

    def read_all_files(self):
        with Pool() as pool:
            results = pool.map(self.read_file, self.all_files)
        return list(results)

    def get_single_csv(self, strategy: str, dataset: str, metric: str):
        return self.all_frames[
            self.base_dir + "/" + strategy + "/" + dataset + "/" + metric + "csv.xz"
        ]

    def get_dataset_names(self):
        return sorted(
            [
                dset
                for dset in os.listdir(
                    self.base_dir + "/" + self.get_strategy_names()[0] + "/"
                )
            ],
            key=str.lower,
        )

    def get_strategy_names(self):
        return sorted(
            [strat for strat in os.listdir(self.base_dir + "/") if strat[0].isupper()],
            key=str.lower,
        )

    def get_metric_names(self):
        metrics: List[str] = sorted(
            [
                metric[:-7]
                for metric in os.listdir(
                    self.base_dir
                    + "/"
                    + self.get_strategy_names()[0]
                    + "/"
                    + self.get_dataset_names()[0]
                    + "/"
                )
            ],
            key=str.lower,
        )
        metrics.remove("selected_indices")
        return metrics

    def get_hyperparameter_for_metric_filtering(
        self,
    ) -> Set[Tuple[int, int, int, int]]:
        frame = self.hyperparamters.copy()
        # get the frame with the important columns
        frame = frame[
            [
                "EXP_START_POINT",
                "EXP_BATCH_SIZE",
                "EXP_LEARNER_MODEL",
                "EXP_TRAIN_TEST_BUCKET_SIZE",
            ]
        ]
        hyperparam_set = set(frame.parallel_apply(tuple, axis=1))
        return hyperparam_set

    # TODO: Entferne alle Zeilen für die nur eine Konfigutration durchgelaufen ist

    # TODO: konkateniere die Files aller gleichen Metriken für eine Strategie

    # TODO: Load all files together in a tensor
    def get_divergent_params(self):
        initial_param = pd.read_csv("new/01_workload.csv")
        hyper_before = set(initial_param.parallel_apply(tuple, axis=1))
        hyper_after = set(self.hyperparamters.parallel_apply(tuple, axis=1))
        difference = hyper_before.difference(hyper_after)
        print(difference)


if __name__ == "__main__":
    pandarallel.initialize(progress_bar=False)
    start = time.time()
    data = Data("kp_test")
    print(data.get_divergent_params())
    end = time.time() - start
    print(end)
