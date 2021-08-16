import torch


def calculate_storage(inp: torch.Tensor, params: int) -> int:
    """
    Calculates the number of bytes for storing the current layer
    to be analyzed in disk memory.

    inp[0].detach().numpy().itemsize extracts how many bytes a
    single parameter consists of.
    storage_bit = data_type_size * params calculates the
    number of parameters times the number of bytes per parameter.

    :param inp: tensor to extract the number of bits per parameter
    :param params: number of parameters of the current layer
    :return: calculated number of bytes of the current layer
    """
    data_type_size = inp[0].detach().numpy().itemsize
    storage_bit = data_type_size * params

    return storage_bit
