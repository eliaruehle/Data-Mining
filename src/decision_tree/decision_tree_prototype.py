import sys
from sklearn import tree
import graphviz

# from src.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics

from dataset_metafeatures.metrics import Metrics
from dataset_metafeatures.evaluate_metrics import Evaluate_Metrics


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
    """Takes path to dataset.csv files and calculates a decisiontree where every class is a dataset."""
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    metric_names = get_metric_names()
    x = []
    for dataset in evaluate_metrics.metric.metafeatures_dict:
        if dataset == "Iris.csv" or dataset == "seeds.csv" or dataset == "wine_origin.csv":
            x.append(evaluate_metrics.metric.metafeatures_dict[dataset])
            print(dataset)
    y = get_y_for_batch_size(batchsize)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=metric_names,
                                    class_names=["ALIPY_RANDOM", "OPTIMAL_GREEDY_10", "OPTIMAL_GREEDY_10"],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("drawings/batch_size_1_strategy_classification_tree")
    tree.plot_tree(clf)


def get_y_for_batch_size(batchsize):
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
                 ]
    return name_list

def main():
    path_to_datasets = "/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets"
    draw_strategy_classification_tree(path_to_datasets, 1)


if __name__ == "__main__":
    main()
