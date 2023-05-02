from datasets.loader import Loader
from project_helper.method_types import CLUSTER_STRAT
from base_runner import BaseRunner
from typing import Any, List


class ClusterRunner(BaseRunner):
    label: List[str]

    def __init__(
        self, data: Loader, name: str, components: List[CLUSTER_STRAT]
    ) -> None:
        super().__init__(data, name, components)
        self.label = self.data.get_strategy_names()
