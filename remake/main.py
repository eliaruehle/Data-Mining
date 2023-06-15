import os 
import sys
from omegaconf import OmegaConf, DictConfig
from data.loader import DataLoader
from clustering.runner import ClusterRunner, MODE, STACKED


def main():
    config = OmegaConf.load("new/config/test.yaml")
    data = DataLoader()
    print("all fine")
    runner = ClusterRunner(MODE.GPU, config, STACKED.SINGLE, data)
    runner.run()

if __name__ == "__main__":
    main()