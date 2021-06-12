from typing import List, Any


def round_units(
        value: int, types: str, desired_mega: bool = False,
        decimal: bool = False) -> str:
    """
    Rounds the passed values to the nearest unit and
    appends a unit description.

    :param decimal: true if conversion to decimal is desired
    :param desired_mega: true if conversion to mega unit is desired
    :param value: value to be rounded
    :param types: type of unit to be appended
    :return: string of rounded value and appended unit
    """
    if decimal:
        basis = 1000
    else:
        basis = 1024

    if desired_mega:
        return str(round((value / basis ** 2), 3))

    if 0 < (value // basis ** 4):
        return str(round((value / basis ** 4), 2)) + " T" + types
    elif 0 < (value // basis ** 3):
        return str(round((value / basis ** 3), 2)) + " G" + types
    elif 0 < (value // basis ** 2):
        return str(round((value / basis ** 2), 2)) + " M" + types
    elif 0 < (value // basis):
        return str(round((value / basis), 2)) + " K" + types
    else:
        return str(round(value, 2)) + types


class Writer:
    """
    Collects the extracted features using lists
    and provides a clear and structured output.
    It computes the durations ratio of each layer by dividing
    the duration time of each layer by total duration time.

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
        self._total_readwrite = 0

    def total_results(self) -> None:
        """
        Calculates the total results by iterating over each
        entry within the extracted feature list
        """
        for idx in range(0, len(self._features)):
            self._total_parameter += self._features[idx][1]
            self._total_readwrite += self._features[idx][2] + self._features[idx][3]
            self._total_storage += self._features[idx][4]
            self._total_memory += self._features[idx][5]
            self._total_macs += self._features[idx][6]

    def duration_ratio(self) -> None:
        """
        Calculates the duration ratio for each duration within
        the _durations list.
        total_duration is the total duration time.
        Each duration is divided by total_duration, rounded and
        converted to percentages
        """
        total_duration = sum(self._durations)

        for idx in range(0, len(self._durations)):
            self._durations[idx] = \
                round((self._durations[idx] / total_duration) * 100, 2)

    def printout(self) -> None:
        """
        Prints each extracted feature of all layers collected
        in the _features list by creating a structured table.
        """
        self.total_results()
        self.duration_ratio()

        print("|     module name     | parameters | mem read (MB) |"
              " mem write (MB) |  storage (MB)  | inference memory (MB) |"
              "  MAC (Mega)  | duration ratio |")

        iterations = 0  # counter for access to duration ratio

        for item in self._features:
            item[2] = round_units(item[2], "", True)  # converts to mega unit
            item[3] = round_units(item[3], "", True)
            item[4] = round_units(item[4], "", True)
            item[5] = round_units(item[5], "", True)
            item[6] = round_units(item[6], "", True, True)

            print("|", item[0], " " * (18 - len(item[0])), "|",
                  item[1], " " * (9 - len(str(item[1]))), "|",
                  item[2], " " * (12 - len(str(item[2]))), "|",
                  item[3], " " * (13 - len(str(item[3]))), "|",
                  item[4], " " * (13 - len(str(item[4]))), "|",
                  item[5], " " * (20 - len(str(item[5]))), "|",
                  item[6], " " * (11 - len(str(item[6]))), "|",
                  self._durations[iterations], "%",
                  " " * (11 - len(str(self._durations[iterations]))), "|", )

            iterations += 1

        print("-" * 143)
        print("Total Storage: ", round_units(self._total_storage, "Bytes"))
        print("Total Parameters: ", self._total_parameter)
        print("Total inference memory: ", round_units(self._total_memory, "Bytes"))
        print("Total number of Macs: ", round_units(self._total_macs, "MAC", False, True))
        print("Total Memory Read + Write: ", round_units(self._total_readwrite, "Bytes"))
