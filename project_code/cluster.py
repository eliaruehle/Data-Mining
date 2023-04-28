from data import Data
from typing import List
import numpy as np
from sklearn.cluster import KMeans




data:Data = Data("kp_test/", "05_done_workload.csv")

# we want to cluster 

vectors:List[np.ndarray] = list()
labels:List[str] = data.get_all_strategy_names()

for exp_unique_id in data.get_all_unique_ids():
    for dataset in data.get_all_dataset_names():
        for metric in data.get_all_metric_names():
            for strategy in data.get_all_strategy_names():
                print(f"unique id: {exp_unique_id}, dataset {dataset}, metric {metric}, strategy {strategy}")
        break
    break            
            
