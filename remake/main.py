from omegaconf import OmegaConf, DictConfig
from remake.data.loader import DataLoader
from remake.clustering.runner import ClusterRunner, MODE, STACKED


def main():
    config = OmegaConf.load("config/clustering.yaml")
    data = DataLoader()
    runner = ClusterRunner(MODE.GPU, config, STACKED.SINGLE, data)
    runner.run()

if __name__ == "__main__":
    main()