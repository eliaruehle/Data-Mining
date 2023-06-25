from clustering.cpu.matrix import Matrix
import os
import multiprocessing as mp
from typing import List, Dict
import pandas as pd
import json


def read_single_result(result_path: str) -> pd.DataFrame:
    """
    Function to read in a single result.

    Parameters:
    -----------
    result:path : str
        The path to the dataframe.

    Returns:
    --------
    df : pd.Dataframe
        The dataframe associated with the path.
    """
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"File on path {result_path} does not exist.")
    df = pd.read_csv(result_path, index_col=0)
    return df

def read_all_results() -> Matrix:
    """
    Function to read all results.

    Parameters:
    -----------
    None

    Returns:
    --------
    matrix : Matrix
        A matrix object containing all combined values.
    """
    root:str = "../results/clust2"
    result_path = "../results"
    if not os.path.exists(root):
        raise FileNotFoundError(f"Directory {root} does not exist.")
    if "init.txt" in os.listdir(root):
        os.remove(os.path.join(root, "init.txt"))
    all_files = [os.path.join(root, file) for file in os.listdir(root) if file.split("_")[3] != "normalized"]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(read_single_result, all_files)
    pool.close()
    merged_df = results[0]
    labels: List[str] = merged_df.columns.to_list()[1:]
    print(labels)
    for df in results[1:]:
        merged_df += df
    print(merged_df)
    matrix = Matrix(labels, result_path)
    matrix.set_values(merged_df)
    return matrix


def check_against_assumptions():
    """
    Function to check if the guessed similarities correspond to the calculated ones.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    with open("../results/cpu_cluster_result.json", "r") as file:
        json_data = file.read()
    result_dict = json.loads(json_data)

    similar: List[List[str]] = [
        ['ALIPY_UNCERTAINTY_LC','LIBACT_UNCERTAINTY_LC','ALIPY_UNCERTAINTY_MM','ALIPY_UNCERTAINTY_ENTROPY','ALIPY_UNCERTAINTY_DTB',
         'LIBACT_UNCERTAINTY_SM', 'LIBACT_UNCERTAINTY_ENT', 'SMALLTEXT_LEASTCONFIDENCE', 'SMALLTEXT_PREDICTIONENTROPY',
         'SMALLTEXT_SEALS', 'SMALLTEXT_BREAKINGTIES', 'SKACTIVEML_US_MARGIN', 'SKACTIVEML_US_LC', 'SKACTIVEML_US_ENTROPY'],
        ['ALIPY_UNCERTAINTY_LC','LIBACT_UNCERTAINTY_LC', 'SMALLTEXT_LEASTCONFIDENCE','SKACTIVEML_US_LC'],
        ['ALIPY_UNCERTAINTY_ENTROPY', 'LIBACT_UNCERTAINTY_ENT', 'SMALLTEXT_PREDICTIONENTROPY', 'SKACTIVEML_US_ENTROPY'],
        ['PLAYGROUND_MARGIN', 'ALIPY_UNCERTAINTY_MM', 'LIBACT_UNCERTAINTY_SM', 'SMALLTEXT_BREAKINGTIES',
         'SKACTIVEML_US_MARGIN'],
        ['ALIPY_RANDOM', 'SMALLTEXT_RANDOM'],
        ['LIBACT_QUIRE', 'SKACTIVEML_QUIRE'],
        ['ALIPY_EXPECTED_ERROR_REDUCTION', 'LIBACT_EER', 'SKACTIVEML_EXPECTED_AVERAGE_PRECISION'],
        ['SMALLTEXT_GREEDYCORESET', 'SMALLTEXT_LIGHTWEIGHTCORESET']
    ]

    for cluster in similar:
        for entry in cluster:
            guessed_sim = [strat for strat in cluster if strat != entry and strat in result_dict.keys()]
            # select a threshold in which the strategies can occur -> 6 seems to fit good for 36 calculated experiments
            # 12 is also ok, for this case we have only PLAYGROUND_MARGIN which is not that good
            threshold = len(guessed_sim) + 12
            if entry in result_dict.keys():
                cluster_res = result_dict[entry][:threshold]
            else:
                continue
            if all(names in cluster_res for names in guessed_sim):
                print("Guessed similarities and cluster results correspond to each other.")
            else:
                print("NO correspondence between guessed and clustered similarities.")
                print(entry)


def main():
    """
    The main function of the apllication.
    Retrieves all helpful results.
    """
    matrix = read_all_results()
    # now you can write results
    matrix.write_final_numeric_csv()
    # write normalized results
    matrix.write_final_numeric_normalized_csv()
    # get result dict
    result_dict = matrix.get_results_as_dict()
    print(result_dict)
    # get result_dict as json
    matrix.get_results_as_json()
    # get result_dict as csv
    matrix.get_result_as_csv()
    # check against the guesses
    # check_against_assumptions()
    print("Terminated normally and wrote all results!")


if __name__ == "__main__":
    main()
    #check_against_assumptions()
