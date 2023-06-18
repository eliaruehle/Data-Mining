import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
from numpy import genfromtxt
from similarity_matrix import SimilarityMatrix
from typing import List, Dict
import os   


def comparison():
    julius_object = {'Uncertainly': ['ALIPY_UNCERTAINTY_LC','LIBACT_UNCERTAINTY_LC','ALIPY_UNCERTAINTY_MM','ALIPY_UNCERTAINTY_ENTROPY','ALIPY_UNCERTAINTY_DTB',
                                    'LIBACT_UNCERTAINTY_SM','LIBACT_UNCERTAINTY_ENT','SMALLTEXT_LEASTCONFIDENCE','SMALLTEXT_PREDICTIONENTROPY',
                                'SMALLTEXT_SEALS','SMALLTEXT_BREAKINGTIES','SKACTIVEML_US_MARGIN','SKACTIVEML_US_LC','SKACTIVEML_US_ENTROPY'],
                    'LC': ['ALIPY_UNCERTAINTY_LC','LIBACT_UNCERTAINTY_LC', 'SMALLTEXT_LEASTCONFIDENCE','SKACTIVEML_US_LC'], 
                    'Ent': ['ALIPY_UNCERTAINTY_ENTROPY','LIBACT_UNCERTAINTY_ENT','SMALLTEXT_PREDICTIONENTROPY','SKACTIVEML_US_ENTROPY'], 
                    'MM': ['PLAYGROUND_MARGIN','ALIPY_UNCERTAINTY_MM','LIBACT_UNCERTAINTY_SM','SMALLTEXT_BREAKINGTIES','SKACTIVEML_US_MARGIN'], 
                    'Random': ['ALIPY_RANDOM','SMALLTEXT_RANDOM'], 
                    'Quire': ['LIBACT_QUIRE','SKACTIVEML_QUIRE'], 
                    'EER': ['ALIPY_EXPECTED_ERROR_REDUCTION','LIBACT_EER','SKACTIVEML_EXPECTED_AVERAGE_PRECISION'],
                    'Coreset': ['SMALLTEXT_GREEDYCORESET','SMALLTEXT_LIGHTWEIGHTCORESET']}

    filepath = './generated/cl_res'

    for filename in os.listdir(filepath):
        print(filename)
        ordered_lists = SimilarityMatrix.get_orderd_similarities(SimilarityMatrix.from_csv(filepath + "/" + filename))
        dict = {}      
        for key, value in julius_object.items():
            for element in julius_object[key]:
                list_len = len(julius_object[key])
                if element in ordered_lists.keys():
                    for similar_strategy in julius_object[key]:
                            ordered_lists[element] = ordered_lists[element][:list_len+3]
                            if similar_strategy in ordered_lists[element]:
                                if element in dict.keys():
                                    dict[element].append(similar_strategy)
                                else:
                                    dict[element] = [similar_strategy]

        print(dict)


if __name__ == "__main__":
    comparison()