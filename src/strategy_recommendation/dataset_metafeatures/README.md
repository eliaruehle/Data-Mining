# Data set Metafeatures Script

The `dataset_metafeatures` library provides a `metrics.py` file that allows you to load CSV datasets and compute metafeatures for analysis. The analysis of the computed metafeatures is done within the `evaluate_metrics.py` file. This README provides an overview of the library and explains how to use the `dataset_metafeatures` package to load datasets, extract and evaluate their metafeatures.

## Usage of `evaluate_metrics.py`

1. Open the `evaluate_metrics.py` script in a text editor
2. Locate the `main()` function in the script. This function provides various options for the seed analysis. Depending on your intentions, you may customize these options to suit your needs.

- **Paths and Directories**
  Inside the `main()` method, the following paths and directories are used:

  - `file_path`: Path to the directory containing the data sets for analysis. Make sure when instantiating the `Evaluate_Metrics` class to choose the correct file path.
