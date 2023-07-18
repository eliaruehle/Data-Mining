import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
from numpy import genfromtxt
from similarity_matrix import SimilarityMatrix
from ast import literal_eval
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_heatmap(clustering_method: str, title: str):
  
  filepath = "./remake/results/" + clustering_method + ".csv"

  labels = ['ALIPY_CORESET_GREEDY(1)','ALIPY_DENSITY_WEIGHTED(2)','ALIPY_GRAPH_DENSITY(3)','ALIPY_RANDOM(4)','ALIPY_UNCERTAINTY_LC(5)',
            'ALIPY_UNCERTAINTY_MM(6)','LIBACT_DWUS(7)','LIBACT_QUIRE(8)','LIBACT_UNCERTAINTY_ENT(9)','LIBACT_UNCERTAINTY_LC(10)','LIBACT_UNCERTAINTY_SM(11)',
            'OPTIMAL_GREEDY_10(12)','OPTIMAL_GREEDY_20(13)','PLAYGROUND_BANDIT(14)','PLAYGROUND_GRAPH_DENSITY(15)','PLAYGROUND_INFORMATIVE_DIVERSE(16)',
            'PLAYGROUND_KCENTER_GREEDY(17)','PLAYGROUND_MARGIN(18)','PLAYGROUND_MIXTURE(19)','PLAYGROUND_UNIFORM(20)','SKACTIVEML_COST_EMBEDDING(21)',
            'SKACTIVEML_DAL(22)','SKACTIVEML_DWUS(23)','SKACTIVEML_QBC(24)','SKACTIVEML_QBC_VOTE_ENTROPY(25)','SKACTIVEML_QUIRE(26)','SKACTIVEML_US_ENTROPY(27)',
            'SKACTIVEML_US_LC(28)','SKACTIVEML_US_MARGIN(29)','SMALLTEXT_BREAKINGTIES(30)','SMALLTEXT_CONTRASTIVEAL(31)','SMALLTEXT_EMBEDDINGKMEANS(32)',
            'SMALLTEXT_GREEDYCORESET(33)','SMALLTEXT_LEASTCONFIDENCE(34)','SMALLTEXT_LIGHTWEIGHTCORESET(35)','SMALLTEXT_PREDICTIONENTROPY(36)','SMALLTEXT_RANDOM(37)']
  
  data = SimilarityMatrix.from_csv(filepath=filepath).as_2d_list()
  data_percent = np.array(data) * 100
  np.fill_diagonal(data_percent, np.nan)
  data_percent_rounded = np.round(data_percent, decimals=1)

  fig, ax = plt.subplots(figsize=(30,20))
  sns.heatmap(data_percent_rounded, ax=ax, annot=True, yticklabels=labels, xticklabels=labels, cmap='rocket_r', annot_kws={'fontsize': 20}, vmin=0, vmax=100)
  plt.yticks(rotation=0, fontsize=20, fontweight='bold')
  plt.xticks(rotation=45, ha='right', fontsize=20, fontweight='bold')
  
  cbar = ax.collections[0].colorbar
  cbar.ax.tick_params(labelsize=25)
  cbar.set_label('Similarity in Percentage', fontsize=25, weight='bold')

  plt.savefig('./src/clustering/generated/new_plots/'+ clustering_method + '.pdf', dpi=300, bbox_inches='tight')

def create_stats():
    als = ["OPTIMAL_GREEDY_20", "OPTIMAL_GREEDY_10", "PG_KCENTER_GREEDY", "PG_INFORMATIVE_DIVERSE",
          "PG_MIXTURE", "PG_BANDIT", "ALIPY_QBC", "ALIPY_CORESET_GREEDY", "PG_GRAPH_DENSITY",
          "SMALLTEXT_LEASTCONFIDENCE"]

    performance = [61.73, 61.41, 58.87, 57.88, 57.86,
                  57.84, 57.81, 57.41, 57.07, 56.43]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(als, performance, width=0.6, color = "#868ad1")  
    ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

    for i in ax.patches:
        plt.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.2, 
                str(round(i.get_height(), 2)),
                fontsize=10, fontweight='bold',
                color='black',
                ha='center', va='bottom', rotation=90)

    plt.ylabel("Performance Score")
    plt.title("Performance of Different Active Learning Strategies")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(55, 63)  
    
    plt.tight_layout()  
    plt.savefig('./src/clustering/generated/new_plots/stats.pdf')


def table_visualisation():
    x_achse_vec_space = ['Without PCA', 'PCA 1 Dimension', 'PCA 2 Dimension', 'PCA 3 Dimension', 'PCA 4 Dimension', 'PCA 5 Dimension', 'PCA 6 Dimension', 'PCA 7 Dimension', 'PCA 8 Dimension', 'PCA 9 Dimension']
    x_achse_trees = ['Without PCA', 'PCA 1 Dimension', 'PCA 2 Dimension', 'PCA 3 Dimension', 'PCA 4 Dimension', 'PCA 5 Dimension', 'PCA 6 Dimension', 'PCA 7 Dimension', 'PCA 8 Dimension', 'PCA 9 Dimension']
    y_achse_vec_space = [64.66, 68.24, 67.78, 66.86, 65.70, 65.45, 65.61, 65.60, 65.32, 65.40]
    y_achse_trees = [51.44, 54.54, 54.46, 55.62, 54.78, 53.82, 53.30, 52.81, 52.30, 51.37]


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_achse_vec_space, y_achse_vec_space, width=0.6, label='Vector Space')
    ax.bar(x_achse_trees, y_achse_trees, width=0.6, label='Trees')
    ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

    for i in ax.patches:
        plt.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.4, 
                str(round(i.get_height(), 2)),
                fontsize=12, fontweight='bold',
                color='black',
                ha='center', va='bottom', rotation=90)
    
    plt.ylabel("Performance Score")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(50, 73)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./src/clustering/generated/new_plots/feature_extraction.pdf')

if __name__ == '__main__':
    create_heatmap("final_cluster_normalized", "final cluster normalized")