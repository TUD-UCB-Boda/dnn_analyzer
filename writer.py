from typing import Dict, Tuple, Optional, List, Callable, Any


class Writer():
    """
    Represents objects responsible for a structured and clear output

    Attributes:
        _outputs: list of lists containing all features for each layer
        _total: list of all features summarized
        _durations: array of duration of each layer
    """

    def __init__(self) -> None:
        """
        Inits Writer with empty lists which will be filled by analysing methods
        """

        self._outputs: List[Any] = []
        self._total: List[Any] = []
        self._durations: List[float] = []

    def printout(self) -> None:
        """prints collected features"""
        for idx in range(0, len(self._outputs)):
            print(self._outputs[idx])
