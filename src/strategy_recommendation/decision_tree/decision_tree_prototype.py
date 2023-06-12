from sklearn import tree
import numpy as np
import graphviz
from strategy_recommendation.strategy_performance.recommendation_handler import get_best_strategy, get_all_metrics
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics


def draw_dataset_classification_tree(path_to_datasets):
    """Takes path to dataset.csv files and calculates a decisiontree where every class is a dataset."""
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    x = []
    for dataset in evaluate_metrics.metric.metafeatures_dict:
        x.append(evaluate_metrics.metric.metafeatures_dict[dataset])
    y = list(range(0, len(evaluate_metrics.metric.metafeatures_dict)))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("drawings/classification_tree")
    tree.plot_tree(clf)


def draw_strategy_classification_tree(path_to_datasets, batchsize):
    """Takes path to dataset.csv files and calculates a decisiontree where every class is a strategy."""
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    metric_names = get_metric_names()
    x = []
    for dataset in evaluate_metrics.metric.metafeatures_dict:
        if dataset == "Iris.csv" or dataset == "seeds.csv" or dataset == "wine_origin.csv":
            x.append(evaluate_metrics.metric.metafeatures_dict[dataset])
            #  dataset_name = dataset.split(".")[0]
    y = get_y_for_batch_size_dummy(batchsize)
    print(f"this is x {x}")
    print(f"this is y {y}")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=metric_names,
                                    class_names=["ALIPY_RANDOM", "OPTIMAL_GREEDY_10", "OPTIMAL_GREEDY_10"],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("drawings/batch_size_1_strategy_classification_tree")
    tree.plot_tree(clf)


def draw_strategy_classification_tree_with_different_metrics(path_to_datasets, batchsize):
    """Takes path to dataset.csv files and calculates a decisiontree where every class is a strategy."""
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    metric_names = get_metric_names()
    x, y = get_y_for_batch_size(path_to_datasets, batchsize)
    print(f"this is x {x}")
    print(len(x))
    print(f"this is y {y}")
    print(len(y))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    class_names = []
    for element in y:
        class_names.append(element.split("=")[0])
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=metric_names,
                                    class_names=class_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("drawings/batch_size_"+str(batchsize)+"_strategy_classification_tree_different_metrics")
    tree.plot_tree(clf)


def get_y_for_batch_size(path_to_datasets, batchsize):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    #  metric_names = get_metric_names()
    little_metric_list = ["accuracy", "learner_training_time", "class_distributions_manhattan_added_up", "macro_recall",
                          "weighted_f1-score", "nr_decreasing_al_cycles_per_weighted_precision",
                          "biggest_drop_per_weighted_precision"]
    #  full list can be generated with
    #  get_all_metrics(batchsize)
    dataset_list = ["Iris.csv", "seeds.csv", "wine_origin.csv"]
    x = []
    y = []
    for metric in little_metric_list:
        for dataset in dataset_list:
            to_append_x = evaluate_metrics.metric.metafeatures_dict[dataset]
            if metric == "accuracy":
                to_append_x = np.append(to_append_x, [1, 0, 0, 0, 0, 0, 0])
            if metric == "learner_training_time":
                to_append_x = np.append(to_append_x, [0, 1, 0, 0, 0, 0, 0])
            if metric == "class_distributions_manhattan_added_up":
                to_append_x = np.append(to_append_x, [0, 0, 1, 0, 0, 0, 0])
            if metric == "macro_recall":
                to_append_x = np.append(to_append_x, [0, 0, 0, 1, 0, 0, 0])
            if metric == "weighted_f1-score":
                to_append_x = np.append(to_append_x, [0, 0, 0, 0, 1, 0, 0])
            if metric == "nr_decreasing_al_cycles_per_weighted_precision":
                to_append_x = np.append(to_append_x, [0, 0, 0, 0, 0, 1, 0])
            if metric == "biggest_drop_per_weighted_precision":
                to_append_x = np.append(to_append_x, [0, 0, 0, 0, 0, 0, 1])
            x.append(to_append_x)
            to_append_y = get_best_strategy(batch_size=batchsize, dataset=dataset.split('.')[0], metric=metric)[0] + '=' + metric
            y.append(to_append_y)
    return x, y


def get_y_for_batch_size_dummy(batchsize):
    if batchsize == 1:
        return ["ALIPY_RANDOM", "OPTIMAL_GREEDY_10", "OPTIMAL_GREEDY_10"]
    if batchsize == 5:
        return ["ALIPY_RANDOM", "OPTIMAL_GREEDY_10", "OPTIMAL_GREEDY_10"]
    if batchsize == 10:
        return ["ALIPY_RANDOM", "ALIPY_RANDOM", "ALIPY_RANDOM"]


def get_metric_names():
    name_list = ["number of features",
                  "number of examples",
                  "examples feature_ratio",
                  "average min",
                  "median min",
                  "overall mean",
                  "average max",
                  "standard deviation_mean",
                  "variance mean",
                  "quantile mean",
                  "skewness mean",
                  "kurtosis mean",
                  "number of positive covariance",
                  "number of exact_zero covariance",
                  "number of negative covariance",
                  "entropy mean",
                 "accuracy",
                 "learner_training_time",
                 "class_distributions_manhattan_added_up",
                 "macro_recall",
                 "weighted_f1-score",
                 "nr_decreasing_al_cycles_per_weighted_precision",
                 "biggest_drop_per_weighted_precision"
                 ]
    return name_list


def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    draw_strategy_classification_tree_with_different_metrics(path_to_datasets, '10')


if __name__ == "__main__":
    main()
