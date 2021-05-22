import torch
import torch.nn as nn
import numpy as np


def calculate_flops(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Selects the correct function for calculating the required
    number of flops depending on the passed module type

    :param layer: instance of a specific layer type to be calculated
    :param inp: tensor serving as input for function
    :param output: calculated output of passed layer
    :return: number of required FLOP operations
    """
    if isinstance(layer, (nn.ReLU6, nn.ReLU)):
        return flops_relu(layer, inp, output)
    elif isinstance(layer, nn.Conv2d):
        return flops_conv2d(layer, inp, output)
    elif isinstance(
            layer, (nn.MaxPool1d, nn.AvgPool1d, nn.AvgPool2d,
                    nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool3d,
                    nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d,
                    nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d,
                    nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d)):
        return flops_pooling(layer, inp, output)
    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return flops_batchnorm(layer, inp, output)
    elif isinstance(layer, nn.Linear):
        return flops_linear(layer, inp, output)
    else:
        print('Not supported for Flops calculation:', type(layer).__name__)
        return 0


def flops_relu(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Calculates the number of flops required for the ReLU function.
    Since ReLU calculates the simple function: y = max(0, x),
    it only needs to multiply each dimension of the passed input size

    :param layer: instance of layer type ReLU to be calculated
    :param inp: tensor serving as input for ReLU function
    :param output: calculated output of passed layer
    :return: number of required FLOP operations
    """
    flops_counted = 1
    batch_size = inp.size()[0]
    for idx in inp.size()[1:]:
        flops_counted *= idx

    return flops_counted * batch_size


def flops_conv2d(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Calculates the number of flops required for a convolutional function.
    The formula is:
    K x K x Channels_in x Height_out x Weight_out x Channels_out
    For all pixels in the output feature a convolutional layer takes
    a K x K frame of input values and a dot product of the weights,
    across all input channels. It is repeated Channels_out times since
    the layer has channels_out different convolution kernels.

    Since the number of calculation steps depends on the stride length,
    height_out and width_out must be divided by the stride length.

    The amount of Flops also depends on the number of blocked connections
    from input channels to output channels. Therefore the output channels
    is divided by groups.

    The formula measures the required flops for a batch size of one.
    To calculate the required flops depending on the batch size, the
    result has to be multiplied by the batch size.


    :param layer: instance of convolutional layer to be calculated
    :param inp: tensor serving as input for convolutional function
    :param output: calculated output of passed layer
    :return: number of required FLOP operations
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

    flops_counted = kernel_height * kernel_width * channels_in \
                    * height_out * width_out * channels_out

    return flops_counted * batch_size


def flops_pooling(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Calculates the number of flops required for pooling layers.
    Since their function is to gradually reduce the spatial
    size of the representation by operating independently on
    each feature map, each element in each channel of the feature
    map is compared separately with the other elements within
    the filter.
    This leads to the following formula:
    Flops = batch_size * inp_channels * inp_width * inp_height

    Since the number of flops required for pooling layers are
    relatively low, it hardly makes any noticeable difference
    to the total number of Flops.

    :param layer: instance of pooling layer to be calculated
    :param inp: tensor serving as input for pooling functions
    :param output: calculated output of passed layer
    :return: number of required FLOP operations
    """

    return int(np.prod(inp.shape))


def flops_batchnorm(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Calculates the number of flops required for a batch
    normalization function.

    In batch normalization, each value of the given input is
    normalized separately within one Flop.

    This leads to the following formula:
    Flops = batch_size * inp_channels * inp_width * inp_height

    If affine is enabled (True), the layer has learnable affine
    parameters: Mean and Variance.
    In this case, the mean and variance parameters are updated
    in each step within one Flop, resulting in twice as many
    flops as without affine enabled.

    :param layer: instance of batch norm layer to be calculated
    :param inp: tensor serving as input for batch norm functions
    :param output: calculated output of passed layer
    :return: number of required FLOP operations
    """
    batch_size = inp.size()[0]
    flops_counted = np.prod(inp[0].shape)

    if layer.affine:
        flops_counted *= 2

    return flops_counted * batch_size


def flops_linear(
        layer: nn.Module, inp: torch.Tensor,
        output: torch.Tensor) -> int:
    """
    Calculates the number of flops required for a linear
    function.

    In linear layers each input node is connected to each
    output node.
    This leads to number of connections = input * output.
    Since each output is calculated by the simple formula:
    y = w * x, each calculation requires exactly one Flop.

    So the total number of Flops is input * output * 1.

    :param layer: instance of linear layer to be calculated
    :param inp: tensor serving as input for linear functions
    :param output: calculated output of passed layer
    :return: number of required FLOP operations
    """

    batch_size = inp.size()[0]
    flops_counted = inp.size()[1] * output.size()[1]

    return flops_counted * batch_size
