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

Another potential research objective is to explore the specific scenarios under which certain Active Learning strategies demonstrate superior performance (and possibly reason why so). By conducting a comprehensive analysis of these scenarios the ultimate goal is to develop a decision tree that would provide guidance to potential Active Learning users by selecting the most effective strategy tailored to their specific unlabeled data set.

### Does the first Iteration impact the final Performance?

## Getting started

For this project we have to decided to use [Poetry](https://github.com/python-poetry/poetry), a Python packaging and dependency manager, as it enabled us students to have a uniform installation of packages, python version and dependencies across all our systems. Thus we recommend the usage of **Poetry** when working with this project in order to replicate our research results.

In order to get started, make sure to have the following installed:

- **Python version 3.11.3**
- Poetry $\rightarrow$ `pip install poetry`

After successfully installing the required pre-requisites follow the steps of [how to work with poetry](https://github.com/paul1995tu/Data-Mining/blob/main/documents/poetry.md).

## Project Structure

```
./
├── documents/
│   ├── meetings/
│   ├── CodeConvention.md
│   ├── data_analysis.md
│   ├── poetry.md
│   ├── project_ideas.md
│   └── working_steps.md
├── kp_test/
│   ├── ALIPY_RANDOM/
│   ├── ALIPY_UNCERTAINTY_ENTROPY/
│   ├── ALIPY_UNCERTAINTY_LC/
│   ├── ALIPY_UNCERTAINTY_MM/
│   ├── OPTIMAL_GREEDY_10/
│   ├── SKACTIVEML_DAL/
│   ├── SMALLTEXT_EMBEDDINGKMEANS/
│   ├── SMALLTEXT_LIGHTWEIGHTCORESET/
│   ├── cluster_results/
│   ├── datasets/
│   ├── hpc_deploy/
│   ├── 01_workload.csv
│   ├── 05_done_workload.csv
│   └── 05_started_oom_workloads.csv.xz
├── logs/
│   ├── app.log
│   ├── app.log.1
│   └── app.log.2
├── project_code/
│   ├── __pycache__/
│   ├── clustering/
│   ├── datasets/
│   ├── project_helper/
│   ├── side_handler/
│   └── main.py
├── README.md
├── alg.txt
├── poetry.lock
├── pyproject.toml
├── test.py
└── test.txt
```

- `documents/`: Contains information that has general importance for all members of the project, for example:

  - a subdirectory called `meetings/` which contains a list of our all our meetings
  - `CodeConvention.md`, a general set of rules of coding conventions which every participant of the project has to follow
  - `data_analysis.md`, essentail details about hyperparameters, structure of files, active learning startegies and more

- `kp_test/`: Contains a collection of datasets which were analysed during the process of this research project

- `logs/`: Log files which contain information clustering iterations, errors and more

- `project_code/`: The main essence of this project, where our - as the name already suggests - our project code resides.

  - `clustering/`: All our clustering methods and clustering-related code resides within this subdirectory.

## Conclusion
