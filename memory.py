import torch
import numpy as np


def inference_memory(output: torch.Tensor) -> int:
    """
    Computes memory consumption during inference for
    a specific layer by analyzing the passed output.
    Calculates the number of elements the output
    consists of.
    output[0].detach().numpy().itemsize extracts how
    many bytes a single element consists of.

    :param output: passed output tensor of the related layer
    :return: required memory for passed output tensor as bytes
    """
    memory = int(np.prod(output.shape[1:]))
    data_type_size = output[0].detach().numpy().itemsize

    memory = memory * data_type_size

    return memory
