from omegaconf import OmegaConf
from data.loader import DataLoader
from clustering.runner import ClusterRunner, MODE
import sys
import os

# we have 93 metrices in the config file
def main(index: int) -> None:
    """

    Parameters:
    -----------
    index : int
        The index of the metric file we want to run the calculation on.

    Returns:
    --------
    None

    """
    # load the main config file
    config = OmegaConf.load("/home/h9/elru535b/scratch/elru535b-workspace/Data-Mining/remake/config/data.yaml")
    # create the global DataLoader Object
    data = DataLoader()
    # create the cluster runner
    runner = ClusterRunner(MODE.CPU, config, data)
    # start running
    runner.run(index)


if __name__ == "__main__":
    print(os.getcwd())
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
    else:
        exit("No parameter provided!")
    print(f"Starting main with index {index}")
    main(index)