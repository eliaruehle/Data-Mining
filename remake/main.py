from omegaconf import OmegaConf
from remake.data.loader import DataLoader
from remake.clustering.runner import ClusterRunner, MODE
import sys

# we have 93 metrices in the config file
def main(index: int):
    config = OmegaConf.load("config/test.yaml")
    data = DataLoader()
    runner = ClusterRunner(MODE.CPU, config, data)
    runner.run(index)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
    else:
        exit("No Parameter provided!")
    print(f"Starting main with index {index}")
    main(index)