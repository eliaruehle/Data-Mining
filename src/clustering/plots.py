import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
from numpy import genfromtxt
from similarity_matrix import SimilarityMatrix


def create_heatmap(clustering_method: str, title: str):
  data_origin = pd.read_csv('./src/cl_res/'+ clustering_method +'.csv', index_col=0)
  data = SimilarityMatrix.from_csv(filepath='./src/cl_res/'+ clustering_method +'.csv').normalize().as_2d_list()

  fig, ax = plt.subplots(figsize=(12,10))
  sns.heatmap(data, ax=ax, annot=True, yticklabels=data_origin.index, xticklabels=data_origin.columns, cbar_kws={'label': 'Similarity'}, cmap="YlGnBu")
  plt.yticks(rotation=0)
  plt.xticks(rotation=45, ha='right')
  plt.title(title)
  

  plt.savefig('./src/clustering/generated/new_plots/'+ clustering_method + '.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
  #create_heatmap("kmeans_Iris_3_centers", "KMeans Iris 3 centrer")
  #create_heatmap("kmeans_4_centers", "KMeans 4 centrer")
  #create_heatmap("kmeans_5_centers", "KMeans 5 centrer")
  #create_heatmap("optics_Iris", "OPTICS Iris")
  #create_heatmap("dbscan_Iris", "DBSCAN Iris")
  #create_heatmap("gaussian_mixture_Iris", "Gaussian Mixture Iris")
  #create_heatmap("spec_Iris_3_centers", "Spectral Iris 3 centrer")
  #create_heatmap("spec_Iris_4_cnt", "Spectral Iris 4 centrer")
  #create_heatmap("spec_5_cnt", "Spectral 5 centrer")
  #create_heatmap("spec_wine_origin_5_centers", "Spectral wine origin 5 centers")
  #create_heatmap("optics_wine_origin", "OPTICS wine origin")
  create_heatmap("kmeans_wine_origin_5_centers", "KMeans wine origin 5 centers")
   