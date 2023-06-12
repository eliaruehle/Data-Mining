import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

import strategy_recommendation.dataset_metafeatures.evaluate_metrics
from strategy_recommendation.strategy_performance.recommendation_handler import get_best_strategy
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler


def read_vector_space(path_to_vector_space='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/'):
    if len(path_to_vector_space.split('.')) < 2:
        path_to_vector_space = path_to_vector_space + 'vector_space.pkl'
    vector_space = pd.read_pickle(path_to_vector_space)
    return vector_space


def calculate_scoring(path_to_vector_space, vector):
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
    print(scoring_df)


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
