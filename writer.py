class Writer():
    """
    Represents objects responsible for a structured and clear output

    Attributes:
        _outputs: list of lists containing all features for each layer
        _total: list of all features summarized
        _durations: array of duration of each layer
    """

    def __init__(self):
        """Inits Writer with empty lists which will be filled by analysing methods"""

        self._outputs = []
        self._total = []
        self._durations = []
