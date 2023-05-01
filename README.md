# Data Mining

## Introduction

Active Learning (AL) describes the iterative and intelligent pre-selectio nof unlabeled training data before it is labeled by a human doman expert, in order to subsequently train a classifier on the labeled training data. With the usage of so-called `Active Learning Strategies` is it possible to rank the unlabeled data points without any knowledge of the actual labeling of each data point in prior, meaning that the human labels these data points first in order to advance the trained classification model the furthest. The key message of Active Learning is to reduce the necessacity of human effort to create a labeled dataset.

Within the scope of a current research project, a large dataset (>100GB of numerical CSV files) has been generated, which evaluates ~60 Active Learning Strategies on ~40 datasets with >200 metrics in total. This analysis the given dataset was the key idea of our student research project at the University of Technology in Dresden with the goal to investigate the following research questions:

### Which Active Learning Strategies behave similar?

One potential objective of this study is to investigate the similarities among various Active strategies by examining their performance on different datasets. This can be achieved through the comparison of multiple metrics across datasets, as well as the application of clustering algorithms on the dataset in order to possibly identify patterns and trends in the strategies' behaviour.

### Which Active Learning Strategy is the best for the given scenario?

Another potential research objective is to explore the specific scenarios under which certain Active Learning strategies demonstrate superior performance (and possibly reason why so). By conducting a comprehensive analysis of these scenarios the ultimate goal is to develop a decision tree that would provide guidance to potential Active Learning users by selecting the most effective strategy tailored to their specific unlabeled dataset.

## Prerequisites

Before you can work with Poetry, you need to have it installed on your system. You can install it via pip:

```
$ pip install poetry
```
