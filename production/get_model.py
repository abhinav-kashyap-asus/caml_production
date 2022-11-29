# Forms the CAML ConvAttn model from a set of parameters
from rich.console import Console
import torch
from learn.models import ConvAttnPool
import json
import torch.nn as nn


def get_model(
        vocab_dicts_file: str,
        model_path: str,
        filter_size: int,
        num_filter_maps: int,
        lmbda: float,
        gpu: bool,
) -> nn.Module:
    """

    Parameters
    ----------
    lmbda :
    vocab_dicts_file :
    model_path : str
        File where the model is stored
    filter_size : int
        The size of the kernel
    num_filter_maps : int
        The number of output channels in the convolution
    gpu : bool

    Returns
    -------
    nn.Module

    """
    with open(vocab_dicts_file) as fp:
        dicts = json.load(fp)

    console = Console()

    # Load the saved parameters
    model_params = torch.load(str(model_path))

    # Instantiate the model
    model = ConvAttnPool(
        len(dicts["ind2c"]),
        None,
        filter_size,
        num_filter_maps,
        lmbda,
        gpu,
        dicts,
    )

    model.load_state_dict(model_params)

    console.print("Load Pytorch model :white_check_mark:")

    return model
