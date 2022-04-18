import gc
from mini_nlp_framework.data import DataLoaders, DEFAULT_BS, get_dl_from_tensors, get_kfolds
from mini_nlp_framework.metrics import Metric
from mini_nlp_framework.models import BaseModelProvider
from mini_nlp_framework.train import DefaultTrainer, TrainLength, TrainingStats
import numpy as np
import os
import torch
from typing import List, Optional, Tuple, Union


class CrossValidationStats:
    def __init__(self, stats_by_fold:List[TrainingStats], metric_lower_is_better=False):
        self.stats_by_fold = stats_by_fold
        self.metric_lower_is_better = metric_lower_is_better
        self._best_metric_and_epoch_by_fold = None

    def _best_metric_and_epoch(self, history:np.ndarray) -> Tuple[float, int]:
        epoch = history.argmin() if self.metric_lower_is_better else history.argmax()
        return history[epoch], epoch

    @property
    def best_metric_and_epoch_by_fold(self) -> List[Tuple[float, int]]:
        if self._best_metric_and_epoch_by_fold is None:
            self._best_metric_and_epoch_by_fold = [
                self._best_metric_and_epoch(fold_stats.valid_metric_history) for fold_stats in self.stats_by_fold
            ]
        return self._best_metric_and_epoch_by_fold

    @property
    def avg_best_metric_and_epoch(self) -> Tuple[float, float]:
        """Calculate the average between the best metric/epoch of each fold"""
        return tuple(np.array(self.best_metric_and_epoch_by_fold).mean(axis=0))

    def __str__(self) -> str:
        avg_best_metric, avg_best_epoch = self.avg_best_metric_and_epoch
        linesep = os.linesep
        best_metric_and_epoch_by_fold_str = linesep.join([
            f"    Fold {i}: {value} @ epoch {epoch}"
            for i, (value, epoch) in enumerate(self.best_metric_and_epoch_by_fold)
        ])
        return (
            f"--- Cross validation statistics ---{linesep}"
            f"Average of best metric by fold: {avg_best_metric}{linesep}"
            f"Average epoch of best metric by fold: {avg_best_epoch}{linesep}"
            f"Best metric by fold:{linesep}"
            f"{best_metric_and_epoch_by_fold_str}"
        )


def cross_validate(
    model_provider:BaseModelProvider, X:np.ndarray, y:np.ndarray, seq_lengths:Optional[List[int]]=None, nfolds:int=3, 
    train_length:Union[int, TrainLength]=3, bs:int=DEFAULT_BS, trainer=None, metric:Metric=None, hp=None, device=None, 
    **train_params
) -> CrossValidationStats:
    stats_by_fold = []
    if trainer is None: trainer = DefaultTrainer()

    for X_train, X_test, y_train, y_test, sl_train, sl_test in get_kfolds(X, y, seq_lengths=seq_lengths, n=nfolds):
        train_arrays = [X_train, y_train] if seq_lengths is None else [X_train, y_train, sl_train] 
        valid_arrays = [X_test, y_test] if seq_lengths is None else [X_test, y_test, sl_test]
        train_tensors = [torch.tensor(t) for t in train_arrays]
        valid_tensors = [torch.tensor(t) for t in valid_arrays]
        dls = DataLoaders(get_dl_from_tensors(*train_tensors, bs=bs), get_dl_from_tensors(*valid_tensors, bs=bs))
        model, opt, loss_func, clip_grad = model_provider.create(hp=hp, device=device)
        stats = trainer.train(
            train_length, model, dls, loss_func, opt, metric=metric, device=device, clip_grad=clip_grad, **train_params
        )
        stats_by_fold.append(stats)        
        model = None
        opt = None
        gc.collect()

    return CrossValidationStats(stats_by_fold, metric.lower_is_better if metric is not None else False)
