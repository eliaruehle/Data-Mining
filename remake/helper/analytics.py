from omegaconf import OmegaConf

def get_number_clustered_strategies(conf: OmegaConf) -> int:
    return len(conf["strategies"])

def get_number_clustered_datasets(conf: OmegaConf) -> int:
    return len(conf["datasets"])

def get_number_clustered_metrics(conf: OmegaConf) -> int:
    return len(conf["metrices"])


if __name__ == "__main__":
    config = OmegaConf.load("../config/data.yaml")
    print(f"Number of strategies: {get_number_clustered_strategies(config)}")
    print(f"Number of data sets: {get_number_clustered_datasets(config)}")
    print(f"Number of metrics: {get_number_clustered_metrics(config)}")
