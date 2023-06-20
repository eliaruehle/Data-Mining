import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
from numpy import genfromtxt
from similarity_matrix import SimilarityMatrix


def create_heatmap(clustering_method: str, title: str):
  filepath = "./remake/results/final_cluster_normalized.csv"
  labels = ['ALIPY_CORESET_GREEDY','ALIPY_DENSITY_WEIGHTED','ALIPY_GRAPH_DENSITY','ALIPY_RANDOM','ALIPY_UNCERTAINTY_LC'
            ,'ALIPY_UNCERTAINTY_MM','LIBACT_DWUS','LIBACT_QUIRE','LIBACT_UNCERTAINTY_ENT','LIBACT_UNCERTAINTY_LC','LIBACT_UNCERTAINTY_SM',
            'OPTIMAL_GREEDY_10','OPTIMAL_GREEDY_20','PLAYGROUND_BANDIT','PLAYGROUND_GRAPH_DENSITY','PLAYGROUND_INFORMATIVE_DIVERSE',
            'PLAYGROUND_KCENTER_GREEDY','PLAYGROUND_MARGIN','PLAYGROUND_MIXTURE','PLAYGROUND_UNIFORM','SKACTIVEML_COST_EMBEDDING',
            'SKACTIVEML_DAL','SKACTIVEML_DWUS','SKACTIVEML_QBC','SKACTIVEML_QBC_VOTE_ENTROPY','SKACTIVEML_QUIRE','SKACTIVEML_US_ENTROPY',
            'SKACTIVEML_US_LC','SKACTIVEML_US_MARGIN','SMALLTEXT_BREAKINGTIES','SMALLTEXT_CONTRASTIVEAL','SMALLTEXT_EMBEDDINGKMEANS',
            'SMALLTEXT_GREEDYCORESET','SMALLTEXT_LEASTCONFIDENCE','SMALLTEXT_LIGHTWEIGHTCORESET','SMALLTEXT_PREDICTIONENTROPY','SMALLTEXT_RANDOM']

  #data_origin = pd.read_csv('./src/cl_res/'+ clustering_method +'.csv', index_col=0)
  #data = SimilarityMatrix.from_csv(filepath='./src/cl_res/'+ clustering_method +'.csv').normalize().as_2d_list()
  data = SimilarityMatrix.from_csv(filepath=filepath).as_2d_list()

  fig, ax = plt.subplots(figsize=(30,20))
  sns.heatmap(data, ax=ax, annot=True, yticklabels=labels, xticklabels=labels, cbar_kws={'label': 'Similarity'}, cmap="YlGnBu")
  plt.yticks(rotation=0)
  plt.xticks(rotation=45, ha='right')
  plt.title(title)
  

  plt.savefig('./src/clustering/generated/new_plots/'+ clustering_method + '.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
  create_heatmap("all", "all")
   