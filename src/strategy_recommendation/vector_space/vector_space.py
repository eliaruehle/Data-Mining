import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from strategy_recommendation.dataset_metafeatures.evaluate_metrics import Evaluate_Metrics
import strategy_recommendation.strategy_performance.recommendation_handler as rec_handler


def create_vector_space(path_to_datasets):
    evaluate_metrics = Evaluate_Metrics(path_to_datasets)
    evaluate_metrics.calculate_all_metrics()
    metric_vector_space = evaluate_metrics.metric.metafeatures_dict
    vector_space = pd.DataFrame(columns=['dataset_name', 'metric_vector', 'als_dict'])
    for element in metric_vector_space:
        dataset_name = element.split('.')[0]
        metric_vector = np.array(evaluate_metrics.metric.metafeatures_dict[element])
        als_dict = rec_handler.get_all_strategies(dataset=dataset_name)
        """print(dataset_name)
        print(metric_vector)
        print(als_dict)"""
        new_row = {'dataset_name': dataset_name, 'metric_vector': metric_vector, 'als_dict': als_dict}
        vector_space = vector_space._append(new_row, ignore_index=True)
    vector_space.to_pickle('/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.pkl')
    vector_space.to_csv('/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/vector_space.csv')


def main():
    path_to_datasets = '/home/wilhelm/Uni/data_mining/Data-Mining/kp_test/datasets'
    create_vector_space(path_to_datasets)



if __name__ == '__main__':
    main()
