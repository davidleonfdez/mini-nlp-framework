from typing import Type
import torch
import torch.nn as nn


def get_best_available_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_layers_of_type(model:nn.Module, type:Type[nn.Module]):
    return (m for m in model.modules() if isinstance(m, type))


def set_requires_grad(m:nn.Module, requires_grad:bool):
    for p in m.parameters():
        p.requires_grad = requires_grad
