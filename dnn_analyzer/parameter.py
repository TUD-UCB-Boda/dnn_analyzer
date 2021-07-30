import torch
import torch.nn as nn


def count_parameters(layer: nn.Module) -> int:
    """
    Iterates over all parameters stored as tensors,
    counts the number of parameters for each tensor
    and adds the result to the output

    :param layer: given layer to count its number of parameters
    :return: counted number of parameters as integer
    """
    counted_params = 0
    for name, parameters in layer._parameters.items():
        if not (parameters is None):
            counted_params += torch.numel(parameters.data)
    return counted_params
