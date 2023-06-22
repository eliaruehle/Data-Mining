import pandas as pd
import numpy as np
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler
import strategy_recommendation.vector_space.vector_space as vector_space_class
from scipy.spatial.distance import cdist


def calculate_scoring(vector, path_to_vector_space="/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.pkl"):
    scoring_df = pd.DataFrame(columns=['al-strategy', 'score_sum', 'summand_number'])
    vector_space = vector_space_class.read_vector_space(path_to_vector_space)
    vector = np.array(vector)
    y_reshaped = vector.reshape(1, -1)
    for index, row in vector_space.iterrows():
        metric_vector = row['metric_vector']
        x_reshaped = metric_vector.reshape(1, -1)
        cosine_sim = 1.0 - cdist(x_reshaped, y_reshaped, "cosine")
        if cosine_sim > 0:
            als_dict = row['als_dict']
            for key in als_dict:
                if key in scoring_df['al-strategy'].values:
                    #  add score to sum and increase summand_number by 1
                    key_index = scoring_df.index[scoring_df['al-strategy'] == key].tolist()[0]
                    old_score = scoring_df.at[key_index, 'score_sum']
                    new_score = float(als_dict[key] * cosine_sim)
                    scoring_df.at[key_index, 'score_sum'] = old_score + new_score
                    scoring_df.at[key_index, 'summand_number'] = scoring_df.at[key_index, 'summand_number'] + 1
                else:
                    #  add row to the df, with key , value, 1
                    new_row = {'al-strategy': str(key), 'score_sum': float(als_dict[key] * cosine_sim), 'summand_number': 1}
                    scoring_df = scoring_df._append(new_row, ignore_index=True)
    """
    scoring df looks like this:
                                      al-strategy    score_sum summand_number
    0                      PLAYGROUND_MIXTURE  2204.267837             40
    1                       PLAYGROUND_MARGIN  2203.521726             40
    2          PLAYGROUND_INFORMATIVE_DIVERSE  2203.281387             40
    3                       PLAYGROUND_BANDIT  2202.550327             40
    4                               ALIPY_QBC  2195.377969             40
    5               SKACTIVEML_COST_EMBEDDING  2184.472079             40
    6                    ALIPY_CORESET_GREEDY  2155.434913             40
    7               PLAYGROUND_KCENTER_GREEDY  2147.936080             40
    8                       OPTIMAL_GREEDY_10  2271.314929             40
    9   SKACTIVEML_EXPECTED_AVERAGE_PRECISION  1541.372976             25
    """


    """
    There is another problem: 
                                      al-strategy    score_sum summand_number
    0                       OPTIMAL_GREEDY_20  1761.993551             30
    1                       OPTIMAL_GREEDY_10  1760.555648             30
    """
    scoring_df = scoring_df.sort_values("score_sum", ascending=False)
    scoring_df.reset_index(drop=True, inplace=True)
    return scoring_df


def evaluate_1_run_old(path_to_datasets='/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    test_set = vector_space_class.create_vector_space_split_at_n(evaluate_metrics)
    hit = 0
    miss = 0
    for vector_name in test_set:
        vector = np.array(evaluate_metrics.metric.metafeatures_dict[vector_name])
        scoring_df = calculate_scoring(vector)
        calculated_first_place = scoring_df.loc[0, "al-strategy"]
        real_first_place = rec_handler.get_best_strategy(dataset=vector_name.split(".")[0])[0]
        if calculated_first_place == real_first_place:
            hit = hit + 1
        else:
            miss = miss + 1
    print(f"this is hit {hit}")
    print(f"this is miss {miss}")
    return hit, miss


def evaluate_1_run(path_to_datasets='/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    test_set = vector_space_class.create_vector_space_split_at_n(evaluate_metrics)
    hit = 0
    miss = 0
    for vector_name in test_set:
        vector = np.array(evaluate_metrics.metric.metafeatures_dict[vector_name])
        scoring_df = calculate_scoring(vector)
        calculated_first_place = scoring_df.loc[0, "al-strategy"]
        is_calculated_a_hit(calculated_first_place, vector_name)
        real_first_place = rec_handler.get_best_strategy(dataset=vector_name.split(".")[0])[0]
        if calculated_first_place == real_first_place:
            hit = hit + 1
        else:
            miss = miss + 1
    print(f"this is hit {hit}")
    print(f"this is miss {miss}")
    return hit, miss


def is_calculated_a_hit(calculated_first_place, vector_name):
    hit_list = rec_handler.get_all_strategy(dataset=vector_name.split(".")[0])
    #  print(hit_list)



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


def get_averages(path_to_run='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/evaluate/evaluate_vector_space/100_runs.csv'):
    df = pd.read_csv(path_to_run)
    overall_hits = 0
    overall_misses = 0
    for index, row in df.iterrows():
        overall_hits = overall_hits + row['hits']
        overall_misses = overall_misses + row['misses']
    print(overall_hits / (overall_misses + overall_hits))


def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    get_averages()
    evaluate_100_runs(path_to_datasets)


if __name__ == '__main__':
    main()
