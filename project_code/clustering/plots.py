import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# define the data as a numpy array
data = np.array([[0, 275, 276, 254, 363, 313, 274, 293],
                [275, 0, 345, 301, 274, 250, 225, 224],
                [276, 345, 0, 344, 295, 251, 226, 223],
                [254, 301, 344, 0, 269, 241, 196, 211],
                [363, 274, 295, 269, 0, 324, 277, 280],
                [313, 250, 251, 241, 324, 0, 247, 254],
                [274, 225, 226, 196, 277, 247, 0, 235],
                [293, 224, 223, 211, 280, 254, 235, 0]])

# define the row and column labels
labels = ['ALIPY_RANDOM', 'ALIPY_UNCERTAINTY_ENTROPY', 'ALIPY_UNCERTAINTY_LC',
          'ALIPY_UNCERTAINTY_MM', 'OPTIMAL_GREEDY_10', 'SKACTIVEML_DAL',
          'SMALLTEXT_EMBEDDINGKMEANS', 'SMALLTEXT_LIGHTWEIGHTCORESET']

# create a heatmap plot using seaborn
sns.set(font_scale=0.5)
sns.heatmap(data, annot=True, cmap='Blues', square=True,
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Query Strategies')
plt.ylabel('Query Strategies')
plt.title('Distances between Active Learning Query Strategies')
plt.show()
