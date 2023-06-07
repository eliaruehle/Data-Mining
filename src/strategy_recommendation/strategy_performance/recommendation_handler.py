import json


def get_best_strategy(batch_size="1", dataset="Iris", metric="accuracy",
                      path_to_json='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/'
                                   'strategy_performance/average_performance.json'):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    return data[batch_size][dataset][metric][0]


def get_top_k_strategies(batch_size="1", dataset="Iris", metric="accuracy", k=5,
                         path_to_json='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/'
                                      'strategy_performance/average_performance.json'):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    length = len(data[batch_size][dataset][metric])
    return_list = []
    i = 0
    while i < k and i < length:
        return_list.append(data[batch_size][dataset][metric][i])
        i += 1
    print(return_list)


def main():
    get_top_k_strategies()


if __name__ == '__main__':
    main()
