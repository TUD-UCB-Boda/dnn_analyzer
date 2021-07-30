from dnn_analyzer import ModelAnalyse
import importlib.util
import fire


def run(file, model, input=(3, 224, 224), batch=3) -> None:
    """
    Receives a file path and a model name and looks inside
    the provided file for the model with the specified model name.
    It creates an instance of the specified model and starts the
    analysis process.
    :param file: file path where model is located
    :param model: name of the specified model
    :param input: dimensions of the input tensors
    :param batch: batch size for analysis process
    """
    module_specs = importlib.util.spec_from_file_location('models', file)
    module = importlib.util.module_from_spec(module_specs)
    module_specs.loader.exec_module(module)
    nn = getattr(module, model)()
    ModelAnalyse(nn, input, batch)


if __name__ == '__main__':
    fire.Fire(run)
