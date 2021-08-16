from typing import List, Any
from dnn_analyzer import utils
from tabulate import tabulate


class Writer:
    """
    Collects the extracted features using lists
    and provides a clear and structured output.
    It computes the durations ratio of each layer by dividing
    the duration time of each layer by total duration time.

    Attributes:
        _features: list of dicts containing all features for each layer
        _durations: array of duration of each layer
    """

    def __init__(self) -> None:
        """
        Inits Writer with empty lists which will be filled by
        analysing methods
        """

        self._features: List[Any] = []
        self._total_storage = 0
        self._total_parameter = 0
        self._total_macs = 0
        self._total_memory = 0
        self._total_readwrite = 0

    def total_results(self) -> None:
        """
        Calculates the total results by iterating over each
        entry within the extracted feature dict and converts
        them to nearest rounded unit
        """
        for idx in range(0, len(self._features)):
            self._total_parameter += self._features[idx]['parameters']
            self._total_readwrite += self._features[idx]['mem read (MB)'] \
                                     + self._features[idx]['mem write (MB)']
            self._total_storage += self._features[idx]['storage (MB)']
            self._total_memory += self._features[idx]['inference (MB)']
            self._total_macs += self._features[idx]['MACs (Mega)']

        self._total_storage = utils.round_units(self._total_storage, "B")
        self._total_memory = utils.round_units(self._total_memory, "B")
        self._total_readwrite = utils.round_units(self._total_readwrite, "B")
        self._total_macs = utils.round_units(self._total_macs, "MAC", True)

    def duration_ratio(self) -> None:
        """
        Calculates the duration ratio for each duration within
        the _feature list.
        total_duration is the total duration time.
        Each duration is divided by total_duration, rounded and
        converted to percentages
        """
        total_duration = 0
        for idx in range(0, len(self._features)):
            total_duration += self._features[idx]['duration[%]']

        for idx in range(0, len(self._features)):
            ratio = self._features[idx]['duration[%]'] / total_duration
            self._features[idx]['duration[%]'] = \
                str(round(ratio * 100, 2)) + ' %'

    def printout(self) -> None:
        """
        Prints each extracted feature of all layers collected
        in the _features list by creating a structured table.
        """
        self.total_results()
        self.duration_ratio()
        self.convert_features()

        table = lambda df: tabulate(df, headers='keys', tablefmt='psql')

        print(table(self._features))

        print("Total Storage: ", self._total_storage)
        print("Total Parameters: ", self._total_parameter)
        print("Total inference memory: ", self._total_memory)
        print("Total number of Macs: ", self._total_macs)
        print("Total Memory Read + Write: ", self._total_readwrite)

    def convert_features(self) -> None:
        """
        Converts each entry of the extracted feature list to
        mega unit by calling the round_mega function
        """
        for idx in range(0, len(self._features)):
            self._features[idx]['mem read (MB)'] = \
                utils.round_mega(self._features[idx]['mem read (MB)'])
            self._features[idx]['mem write (MB)'] = \
                utils.round_mega(self._features[idx]['mem write (MB)'])
            self._features[idx]['storage (MB)'] = \
                utils.round_mega(self._features[idx]['storage (MB)'])
            self._features[idx]['inference (MB)'] = \
                utils.round_mega(self._features[idx]['inference (MB)'])
            self._features[idx]['MACs (Mega)'] = \
                utils.round_mega(self._features[idx]['MACs (Mega)'], True)
