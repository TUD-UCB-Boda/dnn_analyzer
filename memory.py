import torch
import torch.nn as nn
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


def read_write(
        layer: nn.Module, params: int, inp: torch.Tensor,
        output: torch.Tensor) -> (int, int):
    """
    Selects the correct function to calculate the amount of
    data that the passed layer reads and writes during inference.

    :param layer: instance of a specific layer type to be calculated
    :param params: number of parameters required for the passed layer
    :param inp: tensor serving as input for function
    :param output: calculated output of passed layer
    :return: tuple (memRead, MemWrite) of caomputed mem. read and write
    """
    if isinstance(layer, (nn.ReLU6, nn.ReLU, nn.LeakyReLU, nn.PReLU)):
        return read_write_relu(layer, params, inp)
    elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return read_write_conv(params, inp, output)
    elif isinstance(
            layer, (nn.MaxPool1d, nn.AvgPool1d, nn.AvgPool2d,
                    nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool3d,
                    nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d,
                    nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d,
                    nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d)):
        return read_write_pool(inp, output)
    elif isinstance(layer, nn.Linear):
        return read_write_linear(params, inp, output)
    elif isinstance(
            layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return read_write_batch_norm(inp, output)
    else:
        print(
            'Not supported for memory read and write calculations:',
            type(layer).__name__)
        return 0, 0


def read_write_relu(
        layer: nn.Module, params: int, inp: torch.Tensor) -> (int, int):
    """
    Calculates the amount of data a ReLU function reads and
    writes during inference.
    Since ReLU does not change the size of a passed input
    tensor and does not require any parameters,
    it only needs to read the passed input tensor and write
    the same amount of data as output.

    If the passed layer is an instance of the PReLU function,
    the amount of parameters required for this layer must be
    included in the calculation.

    data_type_bytes = input[0].detach().numpy().itemsize extracts how
    many bytes a single element consists of.

    :param layer: instance of layer type ReLU to be calculated
    :param params: number of parameters required for passed module
    :param inp: tensor serving as input for ReLU function
    :return: tuple (memRead, MemWrite) of computed mem. read and write
    """
    mem_tensors = int(np.prod(inp.shape))
    mem_read = mem_tensors
    mem_write = mem_tensors

    if isinstance(layer, nn.PReLU):
        mem_read = mem_read + params

    return convert_bytes(inp, mem_read, mem_write)


def read_write_conv(
        params: int, inp: torch.Tensor,
        output: torch.Tensor) -> (int, int):
    """
    Calculates the amount of data convolutional functions read and
    write during inference.
    Since convolutional modules change the size of a passed input
    tensor and require parameters, they need to read the passed
    input tensor and all required parameters.
    Then write the modified tensor as output.

    :param params: number of parameters required for passed module
    :param inp: tensor serving as input for convolutional function
    :param output: tensor serving as output for convolutional function
    :return: tuple (memRead, MemWrite) of computed mem. read and write
    """
    batch_size = inp.size()[0]
    mem_read = batch_size * (inp.size()[1:].numel() + params)
    mem_write = int(np.prod(output.shape))

    return convert_bytes(inp, mem_read, mem_write)


def read_write_pool(
        inp: torch.Tensor, output: torch.Tensor) -> (int, int):
    """
    Calculates the amount of data a pooling function reads and
    writes during inference.
    Since pooling layers reduce reduce the spatial size of a
    passed input tensor and does not require any parameters,
    it needs to read the passed input tensor and write
    the output tensor with modified size.

    :param inp: tensor serving as input for Pooling function
    :param output: tensor serving as output for Pooling function
    :return: tuple (memRead, memWrite) of computed mem. read and write
    """
    inp_tensors = int(np.prod(inp.shape))
    output_tensors = int(np.prod(output.shape))

    mem_read = inp_tensors
    mem_write = output_tensors

    return convert_bytes(inp, mem_read, mem_write)


def read_write_linear(
        params: int, inp: torch.Tensor,
        output: torch.Tensor) -> (int, int):
    """
    Calculates the amount of data a linear function reads and
    writes during inference.
    Since linear modules change the size of a passed input
    tensor and require parameters, they need to read the passed
    input tensor and all required parameters.
    Then write the modified tensor as output.

    :param params: number of parameters required for passed module
    :param inp: tensor serving as input for ReLU function
    :param output: tensor serving as output for Pooling function
    :return: tuple (memRead, MemWrite) of computed mem. read and write
    """
    batch_size = inp.size()[0]
    mem_read = batch_size * (int(np.prod(output[1:].shape)) + params)
    mem_write = int(np.prod(output.shape))

    return convert_bytes(inp, mem_read, mem_write)


def read_write_batch_norm(
        inp: torch.Tensor, output: torch.Tensor) -> (int, int):
    """
    Calculates the amount of data batch normalization functions
    read and write during inference.
    Since batch normalization modules don't change the size of the
    passed input tensor and don't require parameters, they need to read
    the passed input tensor. Since batch normalization functions
    calculate two further parameters: mean and variance, each input
    channel must be read twice.
    Then write the modified tensor as output.

    :param inp: tensor serving as input for convolutional function
    :param output: tensor serving as output for convolutional function
    :return: tuple (memRead, MemWrite) of computed mem. read and write
    """
    batch_size = inp.size()[0]
    in_channels = inp.size()[1] * 2
    mem_read = batch_size * (int(np.prod(output.size()[1:])) + in_channels)
    mem_write = int(np.prod(output.shape))

    return convert_bytes(inp, mem_read, mem_write)


def convert_bytes(
        inp: torch.Tensor, mem_read: int, mem_write: int) -> (int, int):
    """
    Converts passed mem_read and mem_write to number of bytes.
    The passed parameters represent the number of single input elements
    a module reads and writes.
    data_type_bytes = inp[0].detach().numpy().itemsize extractes
    how many bytes a single element consists of.

    mem_read|mem_write *= data_type_bytes computes the amount of bytes
    a module reads and writes.

    :param inp: tensor serving as input for convolutional function
    :param mem_read: memory read as number of single input elements
    :param mem_write: memory write as number of single input elements
    :return: (mem_read, mem_write) converted to bytes
    """
    data_type_bytes = inp[0].detach().numpy().itemsize
    mem_read *= data_type_bytes
    mem_write *= data_type_bytes

    return mem_read, mem_write
