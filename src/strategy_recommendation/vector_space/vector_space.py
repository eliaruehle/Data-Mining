import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler
import random


def read_vector_space(path_to_vector_space='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/'):
    if len(path_to_vector_space.split('.')) < 2:
        path_to_vector_space = path_to_vector_space + 'vector_space.pkl'
    vector_space = pd.read_pickle(path_to_vector_space)
    return vector_space


def normalize_vectorspace(path_to_vector_space='/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/', number_of_features = 21):
    #  if you have to read this, I am sorry. The code is very ugly and runs very slow. Out of time.
    max_list = [0] * number_of_features  # 21 is number of metrics
    min_list = [0] * number_of_features  #  same same
    i = 0
    vector_space = read_vector_space(path_to_vector_space)
    for index, row in vector_space.iterrows():
        for element in row["metric_vector"]:
            if element > max_list[i]:
                max_list[i] = element
            if element < min_list[i]:
                min_list[i] = element
            i = i + 1
        i = 0
    print(max_list)
    print(min_list)
    data = {'max_list': [max_list], 'min_list': [min_list]}
    normalization_lists = pd.DataFrame(data=data)
    normalization_lists.to_pickle(
        '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/normalization_lists.pkl')
    normalization_lists.to_csv(
        '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/normalization_lists.csv')
    for index, row in vector_space.iterrows():
        metric_vector = row['metric_vector']
        i = 0
        new_list = []
        for element in metric_vector:
            new_list.append((element - min_list[i]) / (max_list[i] - min_list[i]))
            i = i + 1
        new_list = np.array(new_list)
        #  Todo get the rest of the vector space (als_dict)
        vector_space.at[index, 'metric_vector'] = new_list
    vector_space.to_pickle(
        '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.pkl')
    vector_space.to_csv(
        '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.csv')
    

def normalize_value(number, max_val, min_val):
    return


def create_vector_space(path_to_datasets):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    metric_vector_space = evaluate_metrics.metric.metafeatures_dict
    print(evaluate_metrics.calculate_normalisation_for_all_metafeatures(metafeatures=metric_vector_space))
    vector_space = pd.DataFrame(columns=['dataset_name', 'metric_vector', 'als_dict'])
    for element in metric_vector_space:
        dataset_name = element.split('.')[0]
        if dataset_name != "ThinCross":
            metric_vector = np.array(evaluate_metrics.metric.metafeatures_dict[element])
            als_dict = rec_handler.get_all_strategies(dataset=dataset_name)
            new_row = {'dataset_name': dataset_name, 'metric_vector': metric_vector, 'als_dict': als_dict}
            vector_space = vector_space._append(new_row, ignore_index=True)
        else:
            print("skipped ThinCross")
    vector_space.to_pickle('/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.pkl')
    vector_space.to_csv('/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.csv')


def create_vector_space_split_at_n(evaluate_metrics, pca, n=10, normalized=False):
    #  Todo normalize vector here
    if pca:
        metric_vector_space = evaluate_metrics.reduced_metafeatures_dict
    else:
        metric_vector_space = evaluate_metrics.metric.metafeatures_dict
    whole_metric_vector_space = []
    if pca:
        for element in metric_vector_space['pca']:
            whole_metric_vector_space.append(element)
    else:
        for element in metric_vector_space:
            whole_metric_vector_space.append(element)
    random.shuffle(whole_metric_vector_space)
    test_set = whole_metric_vector_space[0:n]
    training_set = whole_metric_vector_space[n:len(whole_metric_vector_space)]
    vector_space = pd.DataFrame(columns=['dataset_name', 'metric_vector', 'als_dict'])
    for element in training_set:
        dataset_name = element.split('.')[0]
        if pca:
            metric_vector = np.array(evaluate_metrics.reduced_metafeatures_dict['pca'][element])
        else:
            metric_vector = np.array(evaluate_metrics.metric.metafeatures_dict[element])
        als_dict = rec_handler.get_all_strategies(dataset=dataset_name)
        """print(dataset_name)
        print(metric_vector)
        print(als_dict)"""
        new_row = {'dataset_name': dataset_name, 'metric_vector': metric_vector, 'als_dict': als_dict}
        vector_space = vector_space._append(new_row, ignore_index=True)
    vector_space.to_pickle(
        '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.pkl')
    vector_space.to_csv(
        '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.csv')
    return test_set


def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    eval_metrics = Evaluate_Metrics(file_path=path_to_datasets)

    eval_metrics.generate_evaluations(
        dimension_reductions=[
            ["pca", {"n_components": 8}]
        ])
    print(eval_metrics.reduced_metafeatures_dict)
    #normalize_vectorspace()
    """path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    #  create_vector_space_split_at_n(evaluate_metrics)
    create_vector_space(path_to_datasets)"""
    

if __name__ == '__main__':
    main()
