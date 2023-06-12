import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

import strategy_recommendation.dataset_metafeatures.evaluate_metrics
from strategy_recommendation.strategy_performance.recommendation_handler import get_best_strategy
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler


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
    dummy_vec = [3.01029996e-01,  3.30103000e+00,  1.00000000e-03, 0.00000000e+00,
     0.00000000e+00,  4.97702808e-01,  1.00000000e+00, 2.89675643e-01,
     8.39158805e-02,  4.98164912e-01,  1.46535835e-02, - 1.22123342e+00,
     1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.60090246e+00]
    return dummy_vec


def get_evaluate_metric(path_to_datasets):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    return evaluate_metrics

def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    vector_space = create_vector_space(path_to_datasets)
    dummy_vec = get_dummy_vec()



    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    calculate_similarities(vector_space, dummy_vec, evaluate_metrics)
    """df_cosine_similarities = evaluate_metrics.calculate_all_cosine_similarities()
    for element in evaluate_metrics.metric.metafeatures_dict:
        print(element)
        print(evaluate_metrics.metric.metafeatures_dict[element])
    print(type(evaluate_metrics.metric.metafeatures_dict))"""
    # df_cosine_similarities.to_csv("take_a_look.csv")



if __name__ == '__main__':
    main()
