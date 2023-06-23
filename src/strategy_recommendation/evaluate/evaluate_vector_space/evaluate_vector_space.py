import pandas as pd
import numpy as np
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler
import strategy_recommendation.vector_space.vector_space as vector_space_class
from scipy.spatial.distance import cdist
import os


def calculate_scoring(vector,
                      path_to_vector_space="/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.pkl"):
    scoring_df = pd.DataFrame(columns=['al-strategy', 'score_sum', 'summand_number'])
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
        
                                          al-strategy   score_sum summand_number
        0                       OPTIMAL_GREEDY_20  523.061445             30
        1                       OPTIMAL_GREEDY_10  522.877561             30
        2                      PLAYGROUND_MIXTURE  493.895180             30
        3                       PLAYGROUND_MARGIN  493.649462             30
        4                    ALIPY_CORESET_GREEDY  493.449328             30
        5          PLAYGROUND_INFORMATIVE_DIVERSE  493.379412             30
        6                       PLAYGROUND_BANDIT  493.159398             30
        7               PLAYGROUND_KCENTER_GREEDY  492.377570             30
        8                      PLAYGROUND_UNIFORM  490.389636             30
        9             SKACTIVEML_QBC_VOTE_ENTROPY  490.150198             30
        
        
        0                       OPTIMAL_GREEDY_20  632.764921             30
        1                       OPTIMAL_GREEDY_10  632.515250             30
        2                      PLAYGROUND_MIXTURE  599.943677             30
        3                       PLAYGROUND_MARGIN  599.644660             30
        4          PLAYGROUND_INFORMATIVE_DIVERSE  599.411375             30
        5                       PLAYGROUND_BANDIT  599.116917             30
        6                    ALIPY_CORESET_GREEDY  596.078547             30
        7                               ALIPY_QBC  595.303463             30
        8               PLAYGROUND_KCENTER_GREEDY  594.758348             30
        9                      PLAYGROUND_UNIFORM  594.558963             30
        
        
        0                       OPTIMAL_GREEDY_20  466.770176             30
        1                       OPTIMAL_GREEDY_10  466.612943             30
        2                      PLAYGROUND_MIXTURE  441.472980             30
        3                       PLAYGROUND_MARGIN  441.252021             30
        4          PLAYGROUND_INFORMATIVE_DIVERSE  441.025046             30
        5                       PLAYGROUND_BANDIT  440.827151             30
        6                    ALIPY_CORESET_GREEDY  440.649467             30
        7               PLAYGROUND_KCENTER_GREEDY  439.704317             30
        8                      PLAYGROUND_UNIFORM  438.156531             30
        9             SKACTIVEML_QBC_VOTE_ENTROPY  437.981330             30
        
        
        0                       OPTIMAL_GREEDY_20  590.582541             30
        1                       OPTIMAL_GREEDY_10  590.303431             30
        2                      PLAYGROUND_MIXTURE  562.043017             30
        3                       PLAYGROUND_MARGIN  561.772818             30
        4          PLAYGROUND_INFORMATIVE_DIVERSE  561.636718             30
        5                       PLAYGROUND_BANDIT  561.345759             30
        6                               ALIPY_QBC  558.294493             30
        7               SKACTIVEML_COST_EMBEDDING  555.825944             30
        8                      PLAYGROUND_UNIFORM  555.748184             30
        9             SKACTIVEML_QBC_VOTE_ENTROPY  555.550569             30
    """
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
                    new_row = {'al-strategy': str(key), 'score_sum': float(als_dict[key] * cosine_sim),
                               'summand_number': 1}
                    scoring_df = scoring_df._append(new_row, ignore_index=True)
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
    return hit, miss


def evaluate_1_run(path_to_datasets='/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets',
                   path_to_vector_space_normalization_lists="/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/normalization_lists.pkl",
                   normalized=False):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    test_set = vector_space_class.create_vector_space_split_at_n(evaluate_metrics)
    hit = 0
    miss = 0
    for vector_name in test_set:
        vector = np.array(evaluate_metrics.metric.metafeatures_dict[vector_name])
        if normalized:
            vector = create_normalized_vector(vector, path_to_vector_space_normalization_lists)
        scoring_df = calculate_scoring(vector)
        calculated_first_place = scoring_df.loc[0, "al-strategy"]
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



def evaluate_100_runs(path_to_datasets='/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'):
    evaluation_df = pd.DataFrame(columns=['run_number', 'hits', 'misses'])
    path_to_runs = '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/evaluate/evaluate_vector_space/100_runs.csv'
    i = 0
    while i < 100:
        hit, miss = evaluate_1_run()
        new_row = {'run_number': i, 'hits': hit, 'misses': miss}
        evaluation_df = evaluation_df._append(new_row, ignore_index=True)
        evaluation_df.to_csv(path_to_runs)
        i = i + 1
    get_averages(path_to_runs)


def create_normalized_vector(vector, path_to_vector_space_normalization_lists="/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/normalization_lists.pkl"):
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
    # get_averages()
    evaluate_1_run(path_to_datasets, normalized=True)


if __name__ == '__main__':
    main()
