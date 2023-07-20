import pandas as pd
import numpy as np
import os
import strategy_recommendation.vector_space.vector_space as vector_space_class




def check_vector_usage(path_to_vector_space):
    vector_space = vector_space_class.read_vector_space(path_to_vector_space)
    vector_counter = [0] * 21
    print(vector_counter)
    for index, row in vector_space.iterrows():
        i = 0
        print(row["metric_vector"])
        for element in row["metric_vector"]:
            print(element)
            if element != 0:
                vector_counter[i] = vector_counter[i] + 1
            i += 1
    print(vector_counter)


def main():
    path_to_vector_space = '/home/wilhelm/Uni/data_mining/Data-Mining/src/strategy_recommendation/vector_space/'
    check_vector_usage(path_to_vector_space)


if __name__ == '__main__':
    main()
