import pandas as pd
import numpy as np
import os
from typing import List, Dict
from pprint import pprint


class SimilarityMatrix:
    labels: List[str]
    values: pd.DataFrame
    filename: str
    directory: str = "kp_test/cluster_results"

    def __init__(self, labels: List[str], cluster_strat_name: str) -> None:
        self.labels = labels
        self.values = pd.DataFrame(index=labels, columns=labels, data=0)
        self.filename = cluster_strat_name

    def __str__(self) -> str:
        return pprint(self.values)

    def update(self, strategies: List[str], labels: np.ndarray) -> None:
        strat_label: Dict[str, int] = {
            strat: label for strat, label in zip(strategies, labels)
        }
        for strat, label in strat_label.items():
            same_cluster: List[str] = [
                key
                for key, value in strat_label.items()
                if value == label and key != strat
            ]
            for sim_label_strat in same_cluster:
                self.values.loc[strat, sim_label_strat] += 1

    def get_orderd_similarities(self) -> Dict[str, List[str]]:
        similarities: Dict[str, List[str]] = dict()
        for index, row in self.values.iterrows():
            sorted_indexes = row.sort_values(ascending=False).index.to_list()
            sorted_indexes.remove(index)
            similarities[index] = sorted_indexes.copy()
        return similarities

    def write_to_csv(self, additional_tag: str) -> None:
        filepath: str = os.path.join(
            self.directory, self.filename + "_" + additional_tag + ".csv"
        )

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if os.path.exists(filepath):
            mode = "w"
        else:
            mode = "x"

        self.values.to_csv(filepath, index=True, mode=mode)


"""
def main():
    ma = SimilarityMatrix(["1", "2", "3"], "test_cluster")
    ma.update(["1", "2", "3"], np.array([1, 1, 2]))
    print(ma.get_orderd_similarities())
    ma.write_to_csv("tag_1")


if __name__ == "__main__":
    main()
"""
