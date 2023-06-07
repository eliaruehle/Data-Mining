import json


def get_best_strategy(batch_size="1", dataset="Iris", metric="accuracy",
                      path_to_json='strategy_recommendation/average_performance.json'):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    return data[batch_size][dataset][metric][0]
