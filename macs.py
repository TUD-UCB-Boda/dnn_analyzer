import torch
import torch.nn as nn
import numpy as np


def calculate_macs(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Selects the correct function for calculating the required
    number of macs depending on the passed module type

    :param layer: instance of a specific layer type to be calculated
    :param inp: tensor serving as input for function
    :param output: calculated output of passed layer
    :return: number of required MAC operations
    """
    if isinstance(layer, (nn.ReLU6, nn.ReLU)):
        return macs_relu(layer, inp, output)
    elif isinstance(layer, nn.Conv2d):
        return macs_conv2d(layer, inp, output)
    elif isinstance(
            layer, (nn.MaxPool1d, nn.AvgPool1d, nn.AvgPool2d,
                    nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool3d,
                    nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d,
                    nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d,
                    nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d)):
        return macs_pooling(layer, inp, output)
    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return macs_batchnorm(layer, inp, output)
    elif isinstance(layer, nn.Linear):
        return macs_linear(layer, inp, output)
    else:
        print('Not supported for MACs calculation:', type(layer).__name__)
        return 0


def macs_relu(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Calculates the number of MACs required for the ReLU function.
    Since ReLU calculates the simple function: y = max(0, x),
    it only needs to multiply each dimension of the passed input size

    :param layer: instance of layer type ReLU to be calculated
    :param inp: tensor serving as input for ReLU function
    :param output: calculated output of passed layer
    :return: number of required MAC operations
    """
    macs_counted: int = 1
    batch_size = inp.size()[0]
    for idx in inp.size()[1:]:
        macs_counted *= idx

    return macs_counted * batch_size


def macs_conv2d(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Calculates the number of MACs required for a convolutional function.
    The formula is:
    K x K x Channels_in x Height_out x Weight_out x Channels_out
    For all pixels in the output feature a convolutional layer takes
    a K x K frame of input values and a dot product of the weights,
    across all input channels


    :param layer: instance of convolutional layer to be calculated
    :param inp: tensor serving as input for convolutional function
    :param output: calculated output of passed layer
    :return: number of required MAC operations
    """
    batch_size = inp.size()[0]
    kernel_height, kernel_width = layer.kernel_size
    channels_in = inp.size()[1]
    channels_out, height_out, width_out = output.size()[1:]

    stride_x, stride_y = layer.stride
    width_out = width_out / stride_x
    height_out = height_out / stride_y

    groups = layer.groups
    channels_out = channels_out // groups
    channels_in = channels_in // groups

    macs_counted = kernel_height * kernel_width * channels_in \
                   * height_out * width_out * channels_out * groups

    return macs_counted * batch_size


def macs_pooling(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    batch_size = inp.size()[0]
    return int(np.prod(inp.shape)) * batch_size


def macs_batchnorm(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    batch_size = inp.size()[0]
    macs_counted = np.prod(inp[0].shape)
    if layer.affine:
        macs_counted *= 2

    return macs_counted * batch_size


def macs_linear(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    batch_size = inp.size()[0]
    macs_counted = inp.size()[1] * output.size()[1]

    return macs_counted * batch_size
