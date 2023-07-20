import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

import strategy_recommendation.dataset_metafeatures.evaluate_metrics
from strategy_recommendation.strategy_performance.recommendation_handler import get_best_strategy
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler


def read_vector_space(path_to_vector_space= "/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.pkl"):
    vector_space = pd.read_pickle(path_to_vector_space)
    print(vector_space)

def create_vector_space(path_to_datasets):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    vector_space = evaluate_metrics.metric.metafeatures_dict
    return vector_space


def calculate_similarities(vector_space, vector, evaluate_metrics):
    vector = np.array(vector)
    y_reshaped = vector.reshape(1, -1)
    for element in vector_space:
        x_array = np.array(evaluate_metrics.metric.metafeatures_dict[element])
        x_reshaped = x_array.reshape(1, -1)
        cosine_sim = 1.0 - cdist(x_reshaped, y_reshaped, "cosine")
        print(element)
        dataset = element.split(".")[0]
        top_k_list=rec_handler.get_all_strategies(dataset=dataset)
        print(top_k_list)
        print(cosine_sim)


def get_dummy_vec():
    dummy_vec = [ 6.02059991e-01,  2.17609126e+00,  2.66666667e-02, 0.00000000e+00,
                  0.00000000e+00,  4.48304692e-01,  0.00000000e+00,  0.00000000e+00,
                  1.00000000e+00,  2.56930154e-01,  6.90237786e-02,  4.75282486e-01,
                  6.73757010e-02, -7.50739488e-01,  1.85145951e-01,  4.75282486e-01,
                  6.32062147e-01,  8.83949858e-01,  1.00000000e+00,  5.70534983e+01,
                  3.00000000e+00,  0.00000000e+00,  3.00000000e+00,  3.10759025e+00]
    return dummy_vec


def get_evaluate_metric(path_to_datasets):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    return evaluate_metrics


def main():
    """path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    vector_space = create_vector_space(path_to_datasets)
    dummy_vec = get_dummy_vec()
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    calculate_similarities(vector_space, dummy_vec, evaluate_metrics)"""

    read_vector_space()
    """df_cosine_similarities = evaluate_metrics.calculate_all_cosine_similarities()
    for element in evaluate_metrics.metric.metafeatures_dict:
        print(element)
        print(evaluate_metrics.metric.metafeatures_dict[element])
    print(type(evaluate_metrics.metric.metafeatures_dict))"""
    # df_cosine_similarities.to_csv("take_a_look.csv")


if __name__ == '__main__':
    main()
