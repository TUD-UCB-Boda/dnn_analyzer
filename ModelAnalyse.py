from dnn_analyzer import writer as wr

import torch
import torch.nn as nn


class ModelAnalyse(object):
    '''
    Represents object responsible for organized analysis

    Attributes:
        _origin: dictionary filled with all layers to be analysed
        _model: copy of passed model
        _inp_size: size of inputs the neural network gets
        _writer: object of Writer class printing calculated outputs
    '''

    def __init__(self, model, inp_size):
        """Inits model for analysis and creates object of Writer class. Then it calls the analyse methode"""
        assert isinstance(model, nn.Module)
        assert isinstance(inp_size, (list, tuple))
        assert len(inp_size) == 3

        self._origin = dict()
        self._model = model
        self._inp_size = inp_size
        self._writer = wr.Writer()

        self.analyse()

    def analyse(self):
        """Provides a structured analysis by calling required methods and performing an evaluation with randomised input"""
        # self._modify_submodules()

        x = torch.rand(1, *self._inp_size)
        self._model.eval()
        self._model(x)
        self._writer.printout()

    def _modify_submodules(self):
        """modifies each submodule for analysis"""
        return 0
