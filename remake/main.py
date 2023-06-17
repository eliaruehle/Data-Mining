from omegaconf import OmegaConf
from data.loader import DataLoader
from clustering.runner import ClusterRunner, MODE
import sys
import os

# we have 93 metrices in the config file
def main(index: int):
    config = OmegaConf.load("/home/h9/elru535b/scratch/elru535b-workspace/Data-Mining/remake/config/data.yaml")
    data = DataLoader()
    runner = ClusterRunner(MODE.CPU, config, data)
    runner.run(index)


if __name__ == "__main__":
    print(os.getcwd())
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
    else:
        exit("No parameter provided!")
    print(f"Starting main with index {index}")
    main(index)