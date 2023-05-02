from __future__ import annotations
from abc import ABC, abstractmethod
from datasets.loader import Loader
from project_helper.Logger import Logger
from typing import List, Any


class BaseRunner(ABC):
    data: Loader
    name: str
    components: List[Any]

    def __init__(self, data: Loader, name: str, components: List[Any]) -> None:
        self.data = data
        self.name = name

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def run(self) -> None:
        ...

    @abstractmethod
    def get_indicator(self) -> str:
        ...

    @abstractmethod
    def get_components(self) -> List[Any]:
        ...
