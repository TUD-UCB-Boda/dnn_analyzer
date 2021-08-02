from dnn_analyzer import writer, parameter, macs, disk_storage, memory
import torch
import torch.cuda
import torch.nn as nn
from typing import Dict, Tuple, Callable, Any


class ModelAnalyse(object):
    """
    Responsible for organized analysis. Initializes an object
    of writer class to store extracted features in lists.
    Modifies the given model to analyze it. Each layer within
    the new model gets a modified calling function to not
    only compute the output when the layer is called during
    evaluation process but to call each analyzing function too.
    Run Runs an evaluation on the modified model and thereby
    analyzes it and stores each calculated features layer by layer.
    After that it calls the print method of the writer class for
    printing the extracted features.

    Attributes:
        _origin: dictionary filled with all layers to be analysed
        _model: copy of passed model
        _inp_size: size of inputs the neural network gets:
        [channels, height, width]
        _writer: object of Writer class printing calculated outputs
    """

    def __init__(
            self, model: nn.Module,
            inp_size: Tuple[int, int, int], batch_size=1) -> None:
        """
        Inits model for analysis and creates object of Writer class.
        Then it calls the analyse method

        :param model: passed model to analyze
        :param inp_size: dimensions of input tensor
        :param batch_size: batch size for analysis process
        """
        assert isinstance(model, nn.Module)
        assert isinstance(inp_size, (list, tuple))
        assert len(inp_size) == 3

        self._origin: Dict[Callable[..., Any],
                           Callable[..., Any]] = dict()
        self._model: nn.Module = model
        self._inp_size: Tuple[int, int, int] = inp_size
        self._writer: writer.Writer = writer.Writer()
        self._batch_size = batch_size

        self.analyse()

    def analyse(self) -> None:
        """
        Provides a structured analysis by calling required methods
        and performing an evaluation with randomised input
        """

        self._modify_submodules()

        rand_input = torch.rand(self._batch_size, *self._inp_size)
        print(rand_input.shape)
        self._model.eval()
        self._model(rand_input)
        self._writer.printout()

    def _modify_submodules(self) -> None:
        """
        Iterates over all layers/modules contained in given model.
        Collects all types of modules to be analysed and
        modifies the calling functions of each layer for analysis.
        _Origin serves as a dictionary filled with all layer
        types to be analysed. When the for loop is executed, for each
        layer types within the model (such as Conv2d, ReLU, ...)
        the calling function is modified once. _origin remembers
        if a given type of module has already been modified.
        """

        def analyse_each_layer(
                layer: nn.Module, *inp_tensor: type[torch.Tensor],
                **vars: Any) -> type[torch.Tensor]:
            """
            The new calling function for given layer type.
            During evaluation, when calculating the output for given
            input, this modified function is called instead of the
            normally called functions for calculation.
            It calls any function responsible for extracting a
            particular feature. It uses torch.cuda.Event() and
            torch.cuda.synchronize() to measure the time required to
            compute the output. After a feature is extracted, it is
            transferred to feature_dict.
            Each Layer has its own feature_dict which is passed to main
            _features list of the writer object. At the end of the
            whole evaluation process, _features list has dicts of
            extracted features for each layer. The durations are stored
            in a separate list called _durations.

            :param layer: given module to be analysed
            :param inp_tensor: tensor which serves as input for layer
            :param vars: allows to pass further params
            :return: calculated output of layer for a given input
            """
            assert layer.__class__ in self._origin

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            layer_output = self._origin[layer.__class__](
                layer, *inp_tensor, **vars)

            end.record()

            torch.cuda.synchronize()
            duration_time = start.elapsed_time(end)

            params = parameter.count_parameters(layer)

            if len(inp_tensor) == 1:
                mem_read, mem_write = memory.read_write(
                    layer, params, inp_tensor[0], layer_output)
            elif len(inp_tensor) > 1:
                mem_read, mem_write = memory.read_write(
                    layer, params, inp_tensor, layer_output)

            storage = disk_storage.calculate_storage(inp_tensor, params)

            inference_mem = memory.inference_memory(layer_output)

            if len(inp_tensor) == 1:
                macs_counted = \
                    macs.calculate_macs(layer, inp_tensor[0], layer_output)
            elif len(inp_tensor) > 1:
                macs_counted = \
                    macs.calculate_macs(layer, inp_tensor, layer_output)

            feature_dict = {
                'module name': type(layer).__name__, 'parameters': params,
                'mem read (MB)': mem_read, 'mem write (MB)': mem_write,
                'storage (MB)': storage, 'inference (MB)': inference_mem,
                'MACs (Mega)': macs_counted, 'duration[%]': duration_time
            }

            self._writer._features.append(feature_dict)

            return layer_output

        for layer in self._model.modules():
            if layer.__class__ not in self._origin and \
                    len(list(layer.children())) == 0:
                self._origin[layer.__class__] = layer.__class__.__call__
                layer.__class__.__call__ = analyse_each_layer
