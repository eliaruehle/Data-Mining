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
    if dataset not in data[batch_size]:
        print(f"Warning! The dataset {dataset} is not in the performance.json. For convenience the dataset is changed"
              f" to Iris.")
        dataset = "Iris"
    length = len(data[batch_size][dataset][metric])
    return_list = []
    i = 0
    while i < k and i < length:
        return_list.append(data[batch_size][dataset][metric][i])
        i += 1
    return return_list


def get_all_metrics(batch_size="1", dataset="Iris",
                    path_to_json='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/'
                                      'strategy_performance/average_performance.json'):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    metric_dict = data[batch_size][dataset]
    return_list = []
    for key in metric_dict:
        return_list.append(key)
    return return_list


def get_all_strategies(batch_size="1", dataset="Iris", metric="accuracy",
                         path_to_json='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/'
                                      'strategy_performance/average_performance.json'):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    if dataset not in data[batch_size]:
        print(f"Warning! The dataset {dataset} is not in the performance.json. For convenience the dataset is changed"
              f" to Iris.")
        dataset = "Iris"
    length = len(data[batch_size][dataset][metric])
    return_dict = {}
    i = 0
    while i < length:
        return_dict[data[batch_size][dataset][metric][i][0]] = data[batch_size][dataset][metric][i][1]
        i += 1
    return return_dict


def main():
    list = get_all_strategies()
    print(list)
    #  print(list)
    print(get_all_metrics())


if __name__ == '__main__':
    main()
