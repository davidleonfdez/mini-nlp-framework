from abc import ABC, abstractmethod
from dataclasses import dataclass
from mini_nlp_framework.data import DataLoaders
from mini_nlp_framework.losses import LossFunction
from mini_nlp_framework.metrics import Metric
from mini_nlp_framework.torch_utils import get_best_available_device
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Iterable, List, Optional, Union


@dataclass
class EpochTrainingStats:
    avg_train_loss:float
    train_metric:float
    valid_metric:float
    epoch_idx:int


@dataclass
class TrainingStats:
    train_loss_history:np.ndarray
    train_metric_history:np.ndarray
    valid_metric_history:np.ndarray
    n_epochs:int
    n_steps:int


@dataclass
class ClipGradOptions:
    params:Union[torch.Tensor, Iterable[torch.Tensor]]
    max_norm:float


class TrainLength(ABC):
    "Define a criterion to determine if a training run must finish."
    @abstractmethod
    def must_stop(self, stats:EpochTrainingStats):
        pass


class TrainLengthNEpochs(TrainLength):
    "Training must stop after `n_epochs`."
    def __init__(self, n_epochs:int):
        self.n_epochs = n_epochs

    def must_stop(self, stats:EpochTrainingStats):
        return stats.epoch_idx >= self.n_epochs


class TrainLengthBestMetricEpochsAgo(TrainLength):
    """Training must stop when `n_epochs` have passed since the best metric was obtained.
    
    Args
        n_epochs: number of epochs that must pass since the best metric was achieved to make `must_stop` return `True`.
        lower_is_better: if True, a new metric is considered an improvement when its value is strictly lower than the best;
            if `lower_is_better==False`, when its value is strictly higher than the best.
        use_valid: if True, look at valid metric; else, look at train metric
    """
    def __init__(self, n_epochs:int, lower_is_better=True, use_valid=True):
        assert n_epochs >= 1
        self.n_epochs = n_epochs
        self.best_metric = float("inf") if lower_is_better else float("-inf")
        self.n_epochs_after_best = 0
        self.lower_is_better = lower_is_better
        self.use_valid = use_valid

    def _improved_best_metric(self, new_metric:float):
        return ((new_metric < self.best_metric) if self.lower_is_better else (new_metric > self.best_metric))

    def must_stop(self, stats:EpochTrainingStats):
        metric_to_look_at = stats.valid_metric if self.use_valid else stats.train_metric
        if self._improved_best_metric(metric_to_look_at):
            self.n_epochs_after_best = 0
            self.best_metric = metric_to_look_at
            return False
        else:
            self.n_epochs_after_best += 1
            return self.n_epochs_after_best >= self.n_epochs


class TrainLengthMetricWorseDuringEpochs(TrainLength):
    """Training must stop when the chosen metric has gotten worse during `n_epochs` consecutive epochs.
    
    Args
        n_epochs: number of consecutive epochs that must pass with the new metric being worse than the previous one, 
            to make `must_stop` return `True`.
        lower_is_better: if True, a new metric is considered an improvement when its value is strictly lower than the 
            previous one; if `lower_is_better==False`, a new metric is considered an improvement when its value is 
            strictly higher than the previous one.
        use_valid: if True, look at valid metric; else, look at train metric
    """
    def __init__(self, n_epochs:int, lower_is_better=True, use_valid=True):
        assert n_epochs >= 1
        self.n_epochs = n_epochs
        self.previous_metric = float("inf") if lower_is_better else float("-inf")
        self.n_epochs_getting_worse = 0
        self.lower_is_better = lower_is_better
        self.use_valid = use_valid

    def _improved_previous_metric(self, new_metric:float):
        return ((new_metric < self.previous_metric) if self.lower_is_better else (new_metric > self.previous_metric))

    def must_stop(self, stats:EpochTrainingStats):
        metric_to_look_at = stats.valid_metric if self.use_valid else stats.train_metric
        improved_previous_metric = self._improved_previous_metric(metric_to_look_at)
        self.previous_metric = metric_to_look_at
        if improved_previous_metric:
            self.n_epochs_getting_worse = 0
            return False
        else:
            self.n_epochs_getting_worse += 1
            return self.n_epochs_getting_worse >= self.n_epochs   


class TrainLengthOr(TrainLength):
    """Training must stop when any member of `criterion_list` says it must."""
    def __init__(self, criterion_list:List[TrainLength]):
        self.criterion_list = criterion_list

    def must_stop(self, stats:EpochTrainingStats):
        must_stop = False
        for criterion in self.criterion_list:
            if criterion.must_stop(stats):
                # We can't return `True` here because some criterions could have state and need to update it
                must_stop = True
        return must_stop


class TrainingCallback(ABC):
    def on_step_end(self, tr_loss:torch.Tensor, model:nn.Module, opt:torch.optim.Optimizer):
        pass

    def on_epoch_end(self, stats:EpochTrainingStats, model:nn.Module, opt:torch.optim.Optimizer):
        pass


def train(
    train_length:Union[int, TrainLength], model:nn.Module, dls:DataLoaders, loss_func:LossFunction, 
    opt:torch.optim.Optimizer, sched=None, metric:Optional[Metric]=None, 
    device=None, clip_grad:ClipGradOptions=None, callbacks:List[TrainingCallback]=None
) -> TrainingStats:
    """
    Args
        train_length: if it's an int, number of training epochs; if it's a TrainLength's subclass instance, training
            won't stop until `train_length.must_stop(...)`, which is called at the end of each epoch, returns `True`.
        model: module to train.
        dls: dataloaders that iterates over the training and validation data. If you don't want to evaluate `model`
            using a validation set, `dls.valid` can be `None`.
        train_dl: dataloader that iterates over the training data.
        valid_dl: dataloader that iterates over the validation data.
        loss_func: loss function to minimize. We assume that this loss function applies reduction over the batch, i.e., 
            it only returns one value.
        opt: Pytorch optimizer
        sched: scheduler with a method `step` that will be executed once per step.
        metric: function that receives a model, a DataLoader `dl` and a `metric_fn` function, computes the metric 
            `metric_fn` for every batch of `dl` and returns the average.
        device: device, in Pytorch format, where the model and data should be placed to train and calculate metrics.
        clip_grad: if not None, the gradients of `clip_grad` are clipped to be at most `clip_grad.max_norm` right 
            before each optimizer step.
        callbacks: list of callbacks that must be called every time an event (end of step, end of epoch, ...) occurs.

    Returns: statistics of the training run, like a history of the losses/metrics by epoch
    """
    if isinstance(train_length, int):
        train_length = TrainLengthNEpochs(train_length)
    assert dls.train is not None
    if device is None: device = get_best_available_device()
    if callbacks is None: callbacks = []
    n_steps = 0
    n_epochs_completed = 0
    train_loss_history = []
    train_metric_history = []
    valid_metric_history = []

    while (True):
        model.train()
        train_losses_epoch = None
        n_examples_epoch = 0
        for x, y, *extra_xs in dls.train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            preds = model(x, *extra_xs)
            loss = loss_func(preds, y)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(clip_grad.params, clip_grad.max_norm)
            opt.step()
            n_steps += 1
            if sched is not None: sched.step()

            with torch.no_grad():
                actual_bs = x.shape[0]
                n_examples_epoch += actual_bs
                detached_loss = loss.detach()[None] * actual_bs
                train_losses_epoch = (
                    detached_loss if train_losses_epoch is None else torch.cat((train_losses_epoch, detached_loss))
                )

            for cb in callbacks:
                cb.on_step_end(loss, model, opt)
            #losses.append(loss.detach().cpu().item())
            #print('Train loss = ', loss.detach())

        #print('Epoch completed')
        model.eval()

        train_metric, valid_metric = None, None
        if metric is not None:
            train_metric = metric(model, dls.train, device=device)            
            train_metric_history.append(train_metric)
            if dls.valid is not None:
                valid_metric = metric(model, dls.valid, device=device)
                valid_metric_history.append(valid_metric)
        avg_train_loss = ((train_losses_epoch.sum()) / n_examples_epoch).item()
        train_loss_history.append(avg_train_loss)
        
        n_epochs_completed += 1

        epoch_stats = EpochTrainingStats(avg_train_loss, train_metric, valid_metric, n_epochs_completed)

        for cb in callbacks:
            cb.on_epoch_end(epoch_stats, model, opt)

        if train_length.must_stop(epoch_stats):
            break

        #valid_metric_str = f'{valid_metric:.4f}' if dls.valid is not None else 'N/A'
        #last_iter_train_loss = loss.detach().item()
        #print(f'Avg train loss = {avg_train_loss:.4f}, Last iter train loss = {last_iter_train_loss:.4f}')
        #print(f'Train metric (f1) = {train_metric}')
        #print(f'Valid metric (f1) = {valid_metric}')

    return TrainingStats(
        np.array(train_loss_history), 
        np.array(train_metric_history), 
        np.array(valid_metric_history), 
        n_epochs_completed, 
        n_steps,
    )


class BaseTrainer(ABC):
    @abstractmethod
    def train(
        self, 
        train_length:Union[int, TrainLength], 
        model:nn.Module, 
        dls:DataLoaders, 
        loss_func:LossFunction, 
        opt:torch.optim.Optimizer, 
        sched=None, 
        metric:Optional[Metric]=None,
        device=None, 
        clip_grad:ClipGradOptions=None, 
        callbacks:List[TrainingCallback]=None
    ) -> TrainingStats:
        pass


class DefaultTrainer(BaseTrainer):
    def train(
        self, 
        train_length:Union[int, TrainLength], 
        model:nn.Module, 
        dls:DataLoaders, 
        loss_func:LossFunction, 
        opt:torch.optim.Optimizer, 
        sched=None, 
        metric:Optional[Metric]=None,
        device=None, 
        clip_grad:ClipGradOptions=None, 
        callbacks:List[TrainingCallback]=None
    ) -> TrainingStats:
        return train(
            train_length, model, dls, loss_func, opt, sched=sched, metric=metric, device=device, clip_grad=clip_grad,
            callbacks=callbacks
        )


class MetricsPrinter(TrainingCallback):
    def on_epoch_end(self, stats: EpochTrainingStats, model:nn.Module, opt:torch.optim.Optimizer):
        print(f'*** Epoch {stats.epoch_idx} stats ***')
        print(f'Avg train loss = {stats.avg_train_loss}')
        print(f'Train metric (f1) = {stats.train_metric}')
        if stats.valid_metric is not None:
            print(f'Valid metric (f1) = {stats.valid_metric}')
