import sys
from sklearn import tree
import graphviz

# from src.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
sys.path.append('/home/wilhelm/Uni/data_mining/Data-Mining/src/dataset_metafeatures/')

from metrics import Metrics
from evaluate_metrics import Evaluate_Metrics


def draw_classification_tree(path_to_datasets):
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


def main():
    path_to_datasets = "/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets"
    draw_classification_tree(path_to_datasets)


if __name__ == "__main__":
    main()
