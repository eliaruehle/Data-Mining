from __future__ import annotations
from abc import ABC, abstractmethod
from datasets.loader import Loader
from typing import List, Any


class BaseRunner(ABC):
    """
    The base runner class. Provides underlying structure for every experiment runner.
    """

    data: Loader
    name: str
    components: List[Any]

    def __init__(self, data: Loader, name: str, components: List[Any]) -> None:
        """
        Initialize the BaseRunner object.

        Parameters:
        ----------
        data : Loader
            the loaded data
        name : str
            the name of the runner object
        components : List[Any]
            the components registered

        Returns:
        --------
        None
        """
        self.data = data
        self.name = name
        self.components = components

    def __str__(self) -> str:
        """
        Override of the string function for the base runner object.

        Parameters:
        -----------
        None

        Returns:
        --------
        name : str
            the name of the object
        """
        return self.name

    @abstractmethod
    def run(self, index: int) -> None:
        """
        Runs all the experiments.
        """
        ...

    @abstractmethod
    def get_components(self) -> List[Any]:
        """
        Returns all the registered components.
        """
        ...
