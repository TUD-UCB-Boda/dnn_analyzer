import torch


def inference_memory(output: torch.Tensor):
    memory = 1
    for s in output.size()[1:]:
        memory *= s

    memory = memory * 4 / (1024 ** 2)
    return round(memory, 2)
