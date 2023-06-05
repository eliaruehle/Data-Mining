import os
from omegaconf import OmegaConf
from typing import Dict, List, Set

global conf_dict


def get_strategies():
    strategies: List[str] = sorted(
        [strat for strat in os.listdir("kp_test") if strat[0].isupper()],
        key=str.lower,
    )
    conf_dict["strategies"] = strategies
    return conf_dict


def get_datasets():
    datasets: List[str] = sorted(
        [dset for dset in os.listdir("kp_test/" + conf_dict["strategies"][0])],
        key=str.lower,
    )
    conf_dict["datasets"] = datasets


def get_metrices():
    metrices: List[str] = []  # enter the selected metrices here
    metrices = sorted(metrices, key=str.lower)
    # validate the metrices
    blacklist: Set = set()
    for metric in metrices:
        for strat in conf_dict["strategies"]:
            for dset in conf_dict["datasets"]:
                if not os.path.exists("kp_test/" + strat + "/" + dset + "/" + metric):
                    blacklist.add(metric)
    metrices = list(set(metrices) - blacklist)
    conf_dict["metrices"] = metrices


def get_data() -> Dict[str, List[str]]:
    get_strategies()
    get_datasets()
    get_metrices()


# this need to be adjustet for new environments -> for example after merge
FILE_DIR: str = os.getcwd() + "/new/config/"
FILE_NAME: str = "data.yaml"

if os.path.exists(FILE_DIR + FILE_NAME):
    os.remove(FILE_DIR + FILE_NAME)

conf_dict: Dict[str, List[str]] = {}
get_data()
conf = OmegaConf.create(conf_dict)
OmegaConf.save(conf, FILE_DIR + FILE_NAME)
