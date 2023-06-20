from sklearn import tree
import numpy as np
import graphviz
from strategy_recommendation.strategy_performance.recommendation_handler import get_best_strategy, get_metric_names, get_all_metrics
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics


def draw_classification_tree(evaluate_metrics, metric_names, batchsize, datasets, metric, name_of_tree):
    x, y = get_x_and_y(evaluate_metrics, batchsize, datasets, metric)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=metric_names, class_names=y, filled=True,
                                    rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("drawings/" + name_of_tree)
    tree.plot_tree(clf)


def get_x_and_y(evaluate_metrics, batchsize, datasets, metric):
    x = []
    y = []
    for dataset in datasets:
        to_append_x = evaluate_metrics.metric.metafeatures_dict[dataset]
        x.append(to_append_x)
        to_append_y = get_best_strategy(batchsize=batchsize, dataset=dataset.split('.')[0], metric=metric)[0]
        y.append(to_append_y)
    return x, y


def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    draw_forrest(path_to_datasets)
    
    
def draw_forrest(path_to_datasets):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    metric_list = get_all_metrics()
    dataset_list = []
    for dataset in evaluate_metrics.metric.metafeatures_dict:
        #  'ThinCross' is not in the json
        if dataset != 'ThinCross.csv':
            dataset_list.append(dataset)
        #  print(evaluate_metrics.metric.metafeatures_dict[dataset])
    print(f"this are the datasets: {dataset_list}")
    print(f"this are the metrics: {metric_list}")
    metric_names = get_metric_names()
    #  'ThinCross' is not there
    datasets = dataset_list
    batchsize_list = ["1", "5", "10"]
    for batchsize in batchsize_list:
        for metric in metric_list:
            draw_classification_tree(evaluate_metrics, metric_names, batchsize=batchsize, datasets=datasets, metric=metric,
                                     name_of_tree=batchsize + "_" + metric + "_tree")


if __name__ == '__main__':
    main()
    