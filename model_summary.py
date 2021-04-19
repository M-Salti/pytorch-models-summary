from collections import namedtuple
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from IPython.display import display
from pandas import DataFrame
from tabulate import tabulate
from torch.utils.hooks import RemovableHandle

__all__ = ["create_summary_table", "summary"]


LayerInfo = namedtuple("LayerInfo", ["class_name", "output_shape", "num_params"])

num_params = lambda module: sum(p.numel() for p in module.parameters())
num_trainable = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)


@torch.no_grad()
def create_summary_table(model: nn.Module, input_shape: Tuple[int, ...]) -> DataFrame:
    """Creates a pandas DataFrame containing a summary of the model layers, output shapes and number of parameters

    Args:
        model (nn.Module): This is your PyTorch model
        input_shape (Tuple[int, ...]): Model input shape without batch dimension

    Returns:
        DataFrame: Model summary
    """

    flip_back_train = False
    if model.training:
        flip_back_train = True
        model.eval()

    handles: list[RemovableHandle] = []
    infos: dict[str, LayerInfo] = dict()

    def save_layer_info(module: nn.Module, input, output, name: str):
        infos[name] = LayerInfo(
            class_name=module.__class__.__name__,
            output_shape=tuple(output.shape),
            num_params=num_params(module),
        )

    for name, module in model.named_children():
        handle = module.register_forward_hook(partial(save_layer_info, name=name))
        handles.append(handle)

    dummy_input = torch.empty(input_shape).unsqueeze(0)
    model(dummy_input)

    if flip_back_train:
        model.train()

    for handle in handles:
        handle.remove()
    handles.clear()

    summary = DataFrame(columns=["Layer (type)", "Output Shape", "Param #"], index=range(len(infos)))
    for index, (k, v) in enumerate(infos.items()):
        summary.loc[index] = [
            f"{k} ({v.class_name})",
            f"{(None, *v.output_shape)}",
            f"{v.num_params:,}",
        ]

    return summary


def summary(model: nn.Module, input_shape: Tuple[int, ...], view_df: bool = False, tablefmt: str = "fancy_grid"):
    """Print a summary of the model layers and number of trainable/non-trainable parameters.

    Args:
        model (nn.Module): This is your PyTorch model
        input_shape (Tuple[int, ...]): Model input shape without batch dimension
        view_df (bool, optional): If true, will display a pandas DataFrame. Defaults to False.
        tablefmt (str, optional): The table print format. Check the `tabulate` package docs for more info. Defaults to "fancy_grid".
    """

    table = create_summary_table(model, input_shape)

    print("Model Summary:")
    if view_df:
        widest = 12  # column names width
        widest = max(widest, table.apply(axis=1, func=lambda row: row.apply(len).max()).max())
        widest += widest // 10

        table_styles = dict(selector="th", props=[("text-align", "center")])
        df_properties = {"width": f"{widest}ch", "text-align": "left"}
        styler = table.style.set_properties(**df_properties).set_table_styles([table_styles])
        display(styler)
    else:
        print(tabulate(table, headers="keys", tablefmt=tablefmt))

    model_num_params = num_params(model)
    model_trainable = num_trainable(model)
    model_nontrainable = model_num_params - model_trainable
    print(f"Total params: {model_num_params:,}")
    print(f"Trainable params: {model_trainable:,}")
    print(f"Non-trainable params: {model_nontrainable:,}")
