# Active Learning - Data Mining

## Introduction

Active Learning (AL) has improved the field of Machine Learning (ML) by enabling efficient data labeling and improving model performance. AL describes an iterative process which begins with a sparsely labeled or unlabeled data set to train a ML model. A query strategy selects a batch of informative instances of this data set and sends them to an expert for labeling. By retrieving the new labels, the ML model can be updated. This process continues until a predefined stopping criterion is reached. The advantage of AL is that only data which yields the most promising increase in performance is labeled. However, the abundance of such AL strategies and frameworks has created a need for comparative evaluations to guide researchers and practitioners in selecting the most suitable approach.

In this project, we've conducted a comprehensive analysis of AL strategies with a strong focus on three major topics:

- Investigation of similarities between different AL strategies.
- Introduction of a decision-tree and vector-space based framework for strategy recommendation to create guidance for AL strategy selection.
- Analyzing the relationship between the starting performance and overall performance of AL strategies.

### Which Active Learning Strategies behave similar?

The AL frameworks contain strategies, each with the goal to optimize the performance of a certain ML model. Since different approaches are used within the frameworks and performance is not uniquely defined, we investigated whether there are AL strategies across all frameworks that often deliver similar results for a selection of metrics and therefore can be considered similar. Our findings indicate the existence of AL strategies exhibiting noteworthy similarity. Notably, similar approaches from distinct frameworks show substantial concurrence. Consequently, we've also identified a subset of AL strategies that infrequently yield comparable results, thereby precluding them from being considered similar.

### Which Active Learning Strategy is the best for the given scenario?

In order to recommend fitting AL strategies for an unseen data set, one must calculate its [meta-feature vector](/src/strategy_recommendation/dataset_metafeatures/metrics.py) and then the trained predictors will use that vector to suggest suiting AL strategy candidates. In this project, we have utilized two distinct predictors, a decision-tree and a vector-space based model.

### Does the first Iteration impact the final Performance?

In order to address the practical question of whether a high-performing time series in the first iteration will sustain a high performance overall, we explored the correlation between the score of the first AL cycle and its overall performance. The goal was to estimate the likelihood of securing a top-k result, based upon a given starting value in the first AL cycle. Consequently, one can determine if the estimated probability of obtaining a top-k result is not satisfactory and decide to randomly select alternative parameters for the time series to (possibly) obtain a better chance of getting into the top-k than before.

## Getting started

In order to get started, make sure to have the following installed:

- **Python version 3.10.x**
- Installing the required packages $\rightarrow$ `pip install -r requirements.txt`

## Project Structure

```

```

## Conclusion
