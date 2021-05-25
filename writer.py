from typing import Dict, Tuple, Optional, List, Callable, Any


class Writer():
    """
    Collects the extracted features using lists
    and provides a clear and structured output.
    It computes the durations ratio of each layer by dividing the duration
    time of each layer by total duration time.

    Attributes:
        _features: list of lists containing all features for each layer
        _total: list of all features summarized
        _durations: array of duration of each layer
    """

    def __init__(self) -> None:
        """
        Inits Writer with empty lists which will be filled by analysing methods
        """

        self._features: List[Any] = []
        self._total: List[Any] = []
        self._durations: List[float] = []
        self._total_storage = 0
        self._total_parameter = 0
        self._total_macs = 0
        self._total_memory = 0

    def printout(self) -> None:
        """prints collected features"""
        for idx in range(0, len(self._features)):
            print(self._features[idx])
            self._total_parameter += self._features[idx][1]
            self._total_storage += self._features[idx][2]
            self._total_memory += self._features[idx][3]
            self._total_macs += self._features[idx][4]

        self._total_storage = self._total_storage / (8 * 10 ** 6)

        print("Total Storage: ", round(self._total_storage, 2), "MByte")
        print("Total Parameters: ", self._total_parameter)
        print("Total inference memory: ", self._total_memory)
        print("Total number of Macs: ", self._total_macs)
