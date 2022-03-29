from dataclasses import dataclass
from mini_nlp_framework.losses import LossFunction
from mini_nlp_framework.metrics import Metric
from mini_nlp_framework.models import BaseModelProvider
from mini_nlp_framework.train import BaseTrainer
from mini_nlp_framework.torch_utils import get_best_available_device
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Callable, List, Tuple


class BiasModel(nn.Module):
    "Module that returns the input plus a scalar parameter `a`."
    def __init__(self, a_init:float=10):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float, requires_grad=True))
    
    def forward(self, x):
        return x + self.a


@dataclass
class BiasHyperParameters:
    lr:float=1e-3
    wd:float=0
    adam_betas:Tuple[float,float]=(0.9, 0.999)


class BiasModelProvider(BaseModelProvider):
    def __init__(self, bias_init:float=10):
        self.bias_init = bias_init
        self.create_call_args = []

    def create(self, hp:BiasHyperParameters=None, device=None) -> Tuple[nn.Module, Optimizer, LossFunction]:
        self.create_call_args.append((hp, device))
        if hp is None: hp = BiasHyperParameters()
        if device is None: device = get_best_available_device()
        model = BiasModel(a_init=self.bias_init)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.wd, betas=hp.adam_betas)
        loss = nn.MSELoss()
        return model, opt, loss


class FakeTrainer(BaseTrainer):
    def __init__(self, train_return_args_by_call:List, update_params_fn:Callable[[nn.Module], None]):
        self.call_args = []
        self.call_kwargs = []
        self.train_return_args_by_call = train_return_args_by_call
        self.update_params_fn = update_params_fn

    def train(self, *args, **kwargs):
        call_idx = len(self.call_args)
        self.call_args.append(args)
        self.call_kwargs.append(kwargs)
        model = args[1]
        self.update_params_fn(model)
        return self.train_return_args_by_call[call_idx]


class FakeMetric(Metric):
    def __call__(*args, **kwargs): return 0

    def lower_is_better(self) -> bool:
        return False
