import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

import strategy_recommendation.dataset_metafeatures.evaluate_metrics
from strategy_recommendation.strategy_performance.recommendation_handler import get_best_strategy
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler


def read_vector_space(path_to_vector_space='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/'):
    # ToDo make this more compatible with the measurement like "accuracy". For this we could only read the column
    #  with the metrics and the al-strategies
    if len(path_to_vector_space.split('.')) < 2:
        path_to_vector_space = path_to_vector_space + 'vector_space.pkl'
    vector_space = pd.read_pickle(path_to_vector_space)
    return vector_space


def calculate_scoring(vector, path_to_vector_space="/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.pkl"):
    scoring_df = pd.DataFrame(columns=['al-strategy', 'score_sum', 'summand_number'])
    vector_space = read_vector_space(path_to_vector_space)
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
    #  ToDo make this more of an average and sort it
    return scoring_df


def calculate_similarities(vector_space, vector, evaluate_metrics):
    vector = np.array(vector)
    y_reshaped = vector.reshape(1, -1)
    for element in vector_space:
        x_array = np.array(evaluate_metrics.metric.metafeatures_dict[element])
        x_reshaped = x_array.reshape(1, -1)
        cosine_sim = 1.0 - cdist(x_reshaped, y_reshaped, "cosine")
        print(element)
        dataset = element.split(".")[0]
        top_k_list = rec_handler.get_all_strategies(dataset=dataset)
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


def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    dummy_vec = get_dummy_vec()
    path_to_vector_space = '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/'
    calculate_scoring(path_to_vector_space, dummy_vec)




    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    read_vector_space()
    """df_cosine_similarities = evaluate_metrics.calculate_all_cosine_similarities()
    for element in evaluate_metrics.metric.metafeatures_dict:
        print(element)
        print(evaluate_metrics.metric.metafeatures_dict[element])
    print(type(evaluate_metrics.metric.metafeatures_dict))"""
    # df_cosine_similarities.to_csv("take_a_look.csv")



if __name__ == '__main__':
    main()
