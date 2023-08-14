# Seed Analysis Script

The `seed_analysis.py` script is designed to perform analysis on seed data. This README provides instructions on how to use the script and make the necessary changes to file paths before running it.

## Usage

1. Open the `seed_analysis.py` script in a text editor
2. Locate the `run()` function in the script. This function provides various otpions for the seed analysis. Depending on your intentions, you may customize these options to suit your needs.

- **Paths and Directories**
  Inside the `run()` method, the following paths and directories are used:

  - `file_path`: Path to the directory containing the seed data for analysis.

  - `output_dir`: Directory where the analysis results will be saved. This includes unique start and end frequency data and top-k pair data.

  - `input_dir`: Directory where saved CSVs are loaded from. Used for loading previously generated data for analysis.

  - `plot_start_end_path`: Path to save the histograms depicting the distribution of starting or final values.

  - `plot_top_k_path`: Path to save the histograms for the top-k pairs.


3. Configure the function parameters to tailor your analysis:
   - `hpc`: Set to `True` if you are running the analysis on a high-performance computing environment (HPC). This option generates and saves unique start and end frequency data.
   - `first_or_last`: Specify either '`first`' or '`last`' to analyze the distribution of starting or final values.
   - `save_top_k`: Set to `True` to save the `top-k` pairs into a CSV file.
   - `plot_start_end`: Set to `True` to generate histograms for the distribution of starting or final values.
   - `plot_start_end_path`: Specify the path to save the generated start/end histograms.
   - `plot_top_k`: Set to `True` to generate histograms for the top-k pairs.
   - `plot_top_k_path`: Specify the path to save the generated top-k histograms.
