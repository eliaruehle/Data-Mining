import json
from typing import Dict, List, Tuple

from metrics import Metrics

TOP_N_OCCURENCES: int = 3
datasets: list = []

# Open the file and read the JSON data
with open(
    "/Users/user/GitHub/Data-Mining/src/dataset_metafeatures/performance.json", "r"
) as file:
    data = json.load(file)

# Need to replace the .csv endings
metrics = Metrics("/Users/user/GitHub/Data-Mining/kp_test/datasets")
for item in metrics.data_sets_list:
    datasets.append(item.replace(".csv", ""))


def get_top_n_strategies_for_datasets(
    datasets: list[str],
) -> Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]]:
    """
    Function to get top strategies for each dataset, based on some metrics.

    The function iterates over batch sizes, metrics and strategies in the data.
    For each dataset, the function collects strategy scores and sorts them.
    The top strategies, based on the scores, are selected and added to the results.

    Args:
        datasets (list[str]): A list of dataset names.

    Returns:
        Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]]:
        {
            "batch_size": {
                "metric": {
                    "AL_Strategy": [
                        ["Dataset", score: float],
                        ...
                    ],
                }
            }
        }
    """

    results = {}
    for batch_size, batch_data in data.items():
        results[batch_size] = {}
        for metric, strategies in batch_data.items():
            results[batch_size][metric] = {}
            for dataset in datasets:
                strategy_scores = []
                for strategy, values in strategies.items():
                    for item in values:
                        if item[0] == dataset:
                            strategy_scores.append((strategy, item[1]))
                strategy_scores.sort(key=lambda x: x[1], reverse=True)
                top_strategies = strategy_scores[:TOP_N_OCCURENCES]
                for strategy, score in top_strategies:
                    if strategy not in results[batch_size][metric]:
                        results[batch_size][metric][strategy] = []
                    results[batch_size][metric][strategy].append([dataset, score])
    return results


results = get_top_n_strategies_for_datasets(datasets=datasets)

with open("results/result.json", "w") as f:
    json.dump(results, f)
