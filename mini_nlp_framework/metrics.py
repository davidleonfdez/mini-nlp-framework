from abc import ABC, abstractmethod
from functools import partial
from mini_nlp_framework.predict import predict_binary, predict_dl, predict_multiclass
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable


def metric_lower_is_better(metric_fn:Callable):
    if metric_fn in (accuracy_score, f1_score):
        return False
    #if metric_fn in (,):
    #    return True
    raise ValueError("Unsupported metric function")


class Metric(ABC):
    @property
    @abstractmethod
    def lower_is_better(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, model:nn.Module, dl:DataLoader, **predict_kwargs) -> float:
        "Calculate the value of the metric using the inputs and labels given by `dl`."


class BinaryClassificationMetric(Metric):
    def __init__(self, metric_fn=f1_score):
        self.metric_fn = metric_fn

    def __call__(self, model:nn.Module, dl:DataLoader, **predict_kwargs) -> float:
        with torch.no_grad():
            preds, y = predict_dl(model, dl, **predict_kwargs)
            return self.metric_fn(y, preds)

    @property
    def lower_is_better(self) -> bool:
        return metric_lower_is_better(self.metric_fn)

    @property
    def name(self) -> str:
        return self.metric_fn.__name__


class MulticlassClassificationMetric(Metric):
    def __init__(self, metric_fn=partial(f1_score, average='weighted')):
        self.metric_fn = metric_fn
        self.inner_metric_fn = metric_fn
        while isinstance(self.inner_metric_fn, partial):
            self.inner_metric_fn = self.inner_metric_fn.func

    def __call__(self, model:nn.Module, dl:DataLoader, **predict_kwargs) -> float:
        with torch.no_grad():
            preds, y = predict_dl(model, dl, predict=predict_multiclass, **predict_kwargs)
            return self.metric_fn(y, preds)

    @property
    def lower_is_better(self) -> bool:
        return metric_lower_is_better(self.inner_metric_fn)

    @property
    def name(self) -> str:
        return self.inner_metric_fn.__name__


class LanguageModelMetric(Metric):
    def __init__(self, metric_fn=accuracy_score, pad_idx=0):
        self.metric_fn = metric_fn
        self.pad_idx = pad_idx

    def __call__(self, model:nn.Module, dl:DataLoader, **predict_kwargs) -> float:
        with torch.no_grad():
            preds, y = predict_dl(model, dl, predict=predict_multiclass, **predict_kwargs)
            mask = y.view(-1) != self.pad_idx
            return self.metric_fn(y.view(-1)[mask], preds.view(-1)[mask])

    @property
    def lower_is_better(self) -> bool:
        return metric_lower_is_better(self.metric_fn)

    @property
    def name(self) -> str:
        return self.metric_fn.__name__
