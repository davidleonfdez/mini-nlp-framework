from abc import ABC, abstractmethod
from mini_nlp_framework.predict import predict, predict_dl, predict_multiclass
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

    @abstractmethod
    def __call__(self, model:nn.Module, dl:DataLoader, **predict_kwargs) -> float:
        "Calculate the value of the metric using the inputs and labels given by `dl`."


class ClassificationMetric(Metric):
    def __init__(self, metric_fn=f1_score):
        self.metric_fn = metric_fn

    def __call__(self, model:nn.Module, dl:DataLoader, **predict_kwargs) -> float:
        with torch.no_grad():
            preds, y = predict_dl(model, dl, **predict_kwargs)
            return self.metric_fn(preds, y)

    @property
    def lower_is_better(self) -> bool:
        return metric_lower_is_better(self.metric_fn)


class LanguageModelMetric(Metric):
    def __init__(self, metric_fn=accuracy_score, pad_idx=0):
        self.metric_fn = metric_fn
        self.pad_idx = pad_idx

    def __call__(self, model:nn.Module, dl:DataLoader, **predict_kwargs) -> float:
        with torch.no_grad():
            preds, y = predict_dl(model, dl, predict=predict_multiclass, **predict_kwargs)
            mask = y.view(-1) != self.pad_idx
            return self.metric_fn(preds.view(-1)[mask], y.view(-1)[mask])

    @property
    def lower_is_better(self) -> bool:
        return metric_lower_is_better(self.metric_fn)
