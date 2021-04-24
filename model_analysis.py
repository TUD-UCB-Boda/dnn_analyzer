import typing

import writer as wr
import parameter
import torch
import torch.nn as nn
import time
from typing import Dict, Tuple, Optional, List, Callable, Any, Type


class ModelAnalyse(object):
    '''
    Represents object responsible for organized analysis

    Attributes:
        _origin: dictionary filled with all layers to be analysed
        _model: copy of passed model
        _inp_size: size of inputs the neural network gets
        _writer: object of Writer class printing calculated outputs
    '''

    def __init__(
            self, model: nn.Module,
            inp_size: Tuple[int, int, int]) -> None:
        """Inits model for analysis and creates object of Writer class.
         Then it calls the analyse method
        """
        assert isinstance(model, nn.Module)
        assert isinstance(inp_size, (list, tuple))
        assert len(inp_size) == 3

        self._origin: Dict[Callable[..., Any],
                           Callable[..., Any]] = dict()
        self._model: nn.Module = model
        self._inp_size: Tuple[int, int, int] = inp_size
        self._writer: wr.Writer = wr.Writer()

        self.analyse()

    def analyse(self) -> None:
        """
        Provides a structured analysis by calling required methods
        and performing an evaluation with randomised input
        """

        self._modify_submodules()

        rand_input: torch.Tensor = torch.rand(1, *self._inp_size)
        self._model.eval()
        self._model(rand_input)
        self._writer.printout()

    def _modify_submodules(self) -> None:
        """
        Iterates over all layers/modules contained in given model.
        Collects all types of modules to be analysed and
        modifies the calling functions of each layer for analysis
        """

        def analyse_each_layer(
                layer: nn.Module, *inp_tensor: type[torch.Tensor],
                **vars: Any) -> type[torch.Tensor]:
            """
            Modifies the calling functions of all modules for analyse them.
            When the module is called during an evaluation process,
            each required featureis analysed

            :param layer: given module to be analysed
            :param inp_tensor: tensor which serves as input for given module
            :param vars: allows to pass unspecific number of further params
            :return: calculated output of layer for a given input
            """
            assert layer.__class__ in self._origin

            feature_list: list = [type(layer).__name__]

            params: int = 0  # call function responsible for counting params
            feature_list.append(params)

            disk_storage: int = 0 # call function for calculating storage
            feature_list.append(disk_storage)

            ram_mem: int = 0  # call function for calculating RAM usage
            feature_list.append(ram_mem)

            flops: int = 0  # call function for calculating flops
            feature_list.append(flops)

            macs: int = 0  # call function for calculating macs
            feature_list.append(macs)

            start: float = time.time()
            layer_output = self._origin[layer.__class__](
                layer, *inp_tensor, **vars)
            end: float = time.time()

            self._writer._durations.append(end - start)

            self._writer._outputs.append(feature_list)

            return layer_output

        for layer in self._model.modules():
            if len(list(layer.children())) == 0 and \
                    layer.__class__ not in self._origin:
                self._origin[layer.__class__] = layer.__class__.__call__
                layer.__class__.__call__ = analyse_each_layer
