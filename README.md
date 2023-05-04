# Data Mining

## Introduction

Active Learning (AL) describes the iterative and intelligent pre-selectio nof unlabeled training data before it is labeled by a human doman expert, in order to subsequently train a classifier on the labeled training data. With the usage of so-called `Active Learning Strategies` is it possible to rank the unlabeled data points without any knowledge of the actual labeling of each data point in prior, meaning that the human labels these data points first in order to advance the trained classification model the furthest. The key message of Active Learning is to reduce the necessacity of human effort to create a labeled dataset.

Within the scope of a current research project, a large dataset (>100GB of numerical CSV files) has been generated, which evaluates ~60 Active Learning Strategies on ~40 datasets with >200 metrics in total. This analysis the given dataset was the key idea of our student research project at the University of Technology in Dresden with the goal to investigate the following research questions:

### Which Active Learning Strategies behave similar?

One potential objective of this study is to investigate the similarities among various Active strategies by examining their performance on different datasets. This can be achieved through the comparison of multiple metrics across datasets, as well as the application of clustering algorithms on the dataset in order to possibly identify patterns and trends in the strategies' behaviour.

### Which Active Learning Strategy is the best for the given scenario?

Another potential research objective is to explore the specific scenarios under which certain Active Learning strategies demonstrate superior performance (and possibly reason why so). By conducting a comprehensive analysis of these scenarios the ultimate goal is to develop a decision tree that would provide guidance to potential Active Learning users by selecting the most effective strategy tailored to their specific unlabeled dataset.

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
