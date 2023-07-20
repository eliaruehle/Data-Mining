import pandas as pd
import numpy as np
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler
import strategy_recommendation.decision_tree.decision_tree as decision_tree_class
from scipy.spatial.distance import cdist
import os


def evaluate_1_run(evaluate_metrics, pca, path_to_datasets='/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets',
                   path_to_vector_space_normalization_lists="/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/normalization_lists.pkl",
                   normalized=False):
    test_set, clf = decision_tree_class.create_decision_tree_split_at_n(evaluate_metrics, pca)
    print(f"this is the test set {test_set}")
    hit = 0
    miss = 0
    for vector_name in test_set:
        if pca:
            vector = np.array(evaluate_metrics.reduced_metafeatures_dict['pca'][vector_name])
        else:
            vector = np.array(evaluate_metrics.metric.metafeatures_dict[vector_name])
        if normalized:
            vector = create_normalized_vector(vector, path_to_vector_space_normalization_lists)
        #  scoring_df = calculate_scoring(vector)
        calculated_first_place = clf.predict([vector])
        top_k_list = rec_handler.get_top_k_strategies(dataset=vector_name.split(".")[0])
        if is_hit_a_hit(calculated_first_place, top_k_list):
            hit = hit + 1
        else:
            miss = miss + 1
    return hit, miss


def is_hit_a_hit(calculated_first_place, top_k_list):
    # 2% and best 5 hits
    first_place_score = top_k_list[0][1]
    score_to_beat = (first_place_score / 100) * 98
    for element in top_k_list:
        if calculated_first_place == element[0]:
            if element[1] >= score_to_beat:
                return True
    return False


def evaluate_100_runs(path_to_datasets='/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets', pca=0):
    evaluation_df = pd.DataFrame(columns=['run_number', 'hits', 'misses'])
    path_to_runs = '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/evaluate/evaluate_decision_tree/runs.csv'
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    if pca:
        evaluate_metrics.generate_evaluations(
            dimension_reductions=[
                ["pca", {"n_components": pca}]
            ])
    i = 0
    while i < 10:
        hit, miss = evaluate_1_run(evaluate_metrics, pca)
        new_row = {'run_number': i, 'hits': hit, 'misses': miss}
        evaluation_df = evaluation_df._append(new_row, ignore_index=True)
        evaluation_df.to_csv(path_to_runs)
        i = i + 1
    #  get_averages(path_to_runs)


def create_normalized_vector(vector,
                             path_to_vector_space_normalization_lists="/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/normalization_lists.pkl"):
    normalization_df = pd.read_pickle(path_to_vector_space_normalization_lists)
    max_list = normalization_df.loc[0, "max_list"]
    min_list = normalization_df.loc[0, "min_list"]
    i = 0
    new_list = []
    for element in vector:
        new_list.append((element - min_list[i]) / (max_list[i] - min_list[i]))
        i = i + 1
    new_list = np.array(new_list)
    return new_list


def check_folder(path_to_folder):
    directory = path_to_folder
    overall_hits = 0
    overall_misses = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            hits, misses = get_averages(f)
            overall_hits = overall_hits + hits
            overall_misses = overall_misses + misses
    print(overall_hits / (overall_misses + overall_hits))


def get_averages(
        path_to_runs='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/evaluate/evaluate_vector_space/100_runs.csv'):
    df = pd.read_csv(path_to_runs)
    overall_hits = 0
    overall_misses = 0
    for index, row in df.iterrows():
        overall_hits = overall_hits + row['hits']
        overall_misses = overall_misses + row['misses']
    return overall_hits, overall_misses


def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    #print(get_averages())
    evaluate_100_runs(path_to_datasets, pca=4)


if __name__ == '__main__':
    main()
