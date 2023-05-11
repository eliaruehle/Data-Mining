import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
from numpy import genfromtxt


def create_heatmap(clustering_method: str, title: str):
  data = pd.read_csv('./src/cl_res/'+ clustering_method +'.csv', index_col=0)

  fig, ax = plt.subplots(figsize=(8, 8))
  heatmap = ax.imshow(data, cmap='YlGnBu')

  # Add values on the heatmap
  for i in range(len(data.index)):
      for j in range(len(data.columns)):
          ax.text(j, i, data.iloc[i, j], ha='center', va='center', color='white', fontsize=8)

  # Set tick labels and rotate them by 45 degrees
  ax.set_xticks(range(len(data.columns)))
  ax.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=8)
  ax.set_yticks(range(len(data.index)))
  ax.set_yticklabels(data.index, fontsize=8)
  ax.set_title(title)

  # Show color bar
  cbar = ax.figure.colorbar(heatmap, ax=ax)

  #save plot
  plt.savefig('./src/clustering/plot/'+ clustering_method + '.png', dpi=300, bbox_inches='tight')


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
  create_heatmap("optics_wine_origin", "OPTICS wine origin")
  create_heatmap("kmeans_wine_origin_5_centers", "KMeans wine origin 5 centrer")
   