import torch


def calculate_storage(inp: torch.Tensor, params: int):
    """
    Calculates the number of bits for storing the current layer
    to be analyzed in disk memory.

    inp[0].detach().numpy().itemsize extracts how many bytes a
    single parameter consists of.
    storage_bit = data_type_size * params * 8 calculates the
    number of parameters times the number of bytes per parameter.
    To convert bytes to bits, it is multiplied by 8.

    :param inp: tensor to extract the number of bits per parameter
    :param params: number of parameters of the current layer
    :return: calculated number of bits of the current layer
    """
    data_type_size = inp[0].detach().numpy().itemsize
    storage_bit = data_type_size * params * 8

    return storage_bit
