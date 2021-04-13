
class writer():  # class responsible for a structured and clear output
    def __init__(self):
        self._outputs = []  # list of lists containing all features for each layer
        self._total = []  # list of all features summarized
        self._durations = []  # array of duration of each layer

