import json


def get_best_strategy(batchsize="1", dataset="Iris", metric="accuracy",
                      path_to_json='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/'
                                      'strategy_performance/average_performance.json'):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    if dataset not in data[batchsize]:
        print(f"Warning! The dataset {dataset} is not in the performance.json. For convenience the dataset is changed")
        print(f"the batchsize is {batchsize}")
        return ['OPTIMAL_GREEDY_20', 5.01847053131044]
    if metric not in data[batchsize][dataset]:
        print(f"Warning! The metric {metric} is not in the performance.json. For convenience the metric is changed")
        print(f"the batchsize is {batchsize}")
        print(f"the dataset is {dataset}")
        return ['OPTIMAL_GREEDY_20', 5.01847053131044]
    #  print(len(data[batchsize][dataset][metric][0]))
    #  "COUNT_WRONG_CLASSIFICATIONS": [], "COUNT_WRONG_CLASSIFICATIONS_time_lag": [], :(
    #  COUNT_WRONG_CLASSIFICATIONS is often wrong
    #  COUNT_WRONG_CLASSIFICATIONS_time_lag same same
    #  INCLUDED_IN_OPTIMAL_STRATEGY??? INCLUDED_IN_OPTIMAL_STRATEGY_time_lag
    #  SWITCHES_CLASS_OFTEN SWITCHES_CLASS_OFTEN_time_lag
    #  auc_COUNT_WRONG_CLASSIFICATIONS and auc_COUNT_WRONG_CLASSIFICATIONS_time_lag
    #  auc_INCLUDED_IN_OPTIMAL_STRATEGY auc_INCLUDED_IN_OPTIMAL_STRATEGY_time_lag
    #  auc_SWITCHES_CLASS_OFTEN auc_SWITCHES_CLASS_OFTEN_time_lag
    if data[batchsize][dataset][metric] == []:
        print("there is an error, for convenience the data is changed is changed ")
        print(f"the batchsize is {batchsize}")
        print(f"the dataset is {dataset}")
        print(f"the metric is {metric}")
        return ['OPTIMAL_GREEDY_20', 5.01847053131044]
    return data[batchsize][dataset][metric][0]


def get_all_strategy(batchsize="1", dataset="Iris", metric="accuracy",
                      path_to_json='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/'
                                      'strategy_performance/average_performance.json'):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    if dataset not in data[batchsize]:
        print(f"Warning! The dataset {dataset} is not in the performance.json. For convenience the dataset is changed")
        print(f"the batchsize is {batchsize}")
        return ['OPTIMAL_GREEDY_20', 5.01847053131044]
    if metric not in data[batchsize][dataset]:
        print(f"Warning! The metric {metric} is not in the performance.json. For convenience the metric is changed")
        print(f"the batchsize is {batchsize}")
        print(f"the dataset is {dataset}")
        return ['OPTIMAL_GREEDY_20', 5.01847053131044]
    #  print(len(data[batchsize][dataset][metric][0]))
    #  "COUNT_WRONG_CLASSIFICATIONS": [], "COUNT_WRONG_CLASSIFICATIONS_time_lag": [], :(
    #  COUNT_WRONG_CLASSIFICATIONS is often wrong
    #  COUNT_WRONG_CLASSIFICATIONS_time_lag same same
    #  INCLUDED_IN_OPTIMAL_STRATEGY??? INCLUDED_IN_OPTIMAL_STRATEGY_time_lag
    #  SWITCHES_CLASS_OFTEN SWITCHES_CLASS_OFTEN_time_lag
    #  auc_COUNT_WRONG_CLASSIFICATIONS and auc_COUNT_WRONG_CLASSIFICATIONS_time_lag
    #  auc_INCLUDED_IN_OPTIMAL_STRATEGY auc_INCLUDED_IN_OPTIMAL_STRATEGY_time_lag
    #  auc_SWITCHES_CLASS_OFTEN auc_SWITCHES_CLASS_OFTEN_time_lag
    if data[batchsize][dataset][metric] == []:
        print("there is an error, for convenience the data is changed is changed ")
        print(f"the batchsize is {batchsize}")
        print(f"the dataset is {dataset}")
        print(f"the metric is {metric}")
        return ['OPTIMAL_GREEDY_20', 5.01847053131044]
    return data[batchsize][dataset][metric]


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


def get_metric_names():
    name_list = ["number of features",
                 "number of examples",
                 "examples feature_ratio",
                 "average min",
                 "median min",
                 "overall mean",
                 "average max",
                 "total geometric mean",
                 "total harmonic mean"
                 "standard deviation_mean",
                 "variance mean",
                 "quantile mean",
                 "skewness mean",
                 "kurtosis mean",
                 "percentile",
                 "column cosine similarity mean",
                 "range mean",
                 "coefficient variation mean",
                 "number of positive covariance",
                 "number of exact_zero covariance",
                 "number of negative covariance",
                 "entropy mean",
                 "feature dummy 1",
                 "feature dummy 2",
                 "feature dummy 3"
                 #  Todo ask Tony about the names of the features
                 ]
    return name_list


def main():
    list = get_all_strategies()
    print(list)
    #  print(list)
    print(get_all_metrics())


if __name__ == '__main__':
    main()
