from clustering.cpu.matrix import Matrix
import os
import multiprocessing as mp
from typing import List
import pandas as pd


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
    df = pd.read_csv(result_path)
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
    root:str = "../results/clust"
    result_path = "../results"
    if not os.path.exists(root):
        raise FileNotFoundError(f"Directory {root} does not exist.")
    if "init.txt" in os.listdir(root):
        os.remove(os.path.join(root, "init.txt"))
    all_files = [file for file in os.listdir(root) if file.split("_")[3] != "normalized"]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(read_single_result, all_files)
    pool.close()
    labels: List[str] = results[0].index.to_list()
    combined_df = pd.DataFrame.sum(results)
    matrix = Matrix(labels, result_path)
    matrix.set_values(combined_df)
    return matrix

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
    print("Terminated normally and wrote all results!")


if __name__ == "__main__":
    main()
