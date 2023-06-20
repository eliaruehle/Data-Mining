import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler
import strategy_recommendation.vector_space.vector_space as vector_space
import strategy_recommendation.vector_matching.vector_matching_prototype as vector_matching


def evaluate_1_run(path_to_datasets='/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    test_set = vector_space.create_vector_space_split_at_n(evaluate_metrics)
    hit = 0
    miss = 0
    for vector_name in test_set:
        vector = np.array(evaluate_metrics.metric.metafeatures_dict[vector_name])
        scoring_df = vector_matching.calculate_scoring(vector)
        calculated_first_place = scoring_df.loc[0, "al-strategy"]
        real_first_place = rec_handler.get_best_strategy(dataset=vector_name.split(".")[0])[0]
        if calculated_first_place == real_first_place:
            hit = hit + 1
        else:
            miss = miss + 1
    print(f"this is hit {hit}")
    print(f"this is miss {miss}")
    return hit, miss


def evaluate_100_runs(path_to_datasets='/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'):
    evaluation_df = pd.DataFrame(columns=['run_number', 'hits', 'misses'])
    i = 0
    while i < 100:
        hit, miss = evaluate_1_run()
        new_row = {'run_number': i, 'hits': hit, 'misses': miss}
        evaluation_df = evaluation_df._append(new_row, ignore_index=True)
        evaluation_df.to_csv(
        '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/evaluate/evaluate_vector_space/100_runs.csv'
        )
        i = i + 1


def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    evaluate_100_runs(path_to_datasets)


if __name__ == '__main__':
    main()
