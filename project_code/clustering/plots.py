import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
from numpy import genfromtxt


def create_heatmap(clustering_method: str, title: str):
  data = pd.read_csv('./kp_test/cluster_results/'+ clustering_method +'.csv', index_col=0)

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
  plt.savefig('./project_code/clustering/plot/'+ clustering_method + '.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
  #create_heatmap("kmeans_4centers", "KMeans 4 centrer")
  #create_heatmap("optics_clustering_centers", "OPTICS")
  create_heatmap("gaussian_mixture_centers", "Gaussian Mixture")
   