import gc
from mini_nlp_framework.data import DataLoaders, DEFAULT_BS, get_dl_from_tensors, get_kfolds
from mini_nlp_framework.models import BaseModelProvider
from mini_nlp_framework.train import DefaultTrainer, TrainLength, TrainingStats
import numpy as np
import torch
from typing import List, Optional, Union


def cross_validate(
    model_provider:BaseModelProvider, X:np.array, y:np.array, seq_lengths:Optional[List[int]]=None, nfolds:int=3, 
    train_length:Union[int, TrainLength]=3, bs:int=DEFAULT_BS, trainer=None, metric=None, hp=None, device=None, 
    **train_params
) -> List[TrainingStats]:
    stats_by_fold = []
    if trainer is None: trainer = DefaultTrainer()
    for X_train, X_test, y_train, y_test, sl_train, sl_test in get_kfolds(X, y, seq_lengths=seq_lengths, n=nfolds):
        train_arrays = [X_train, y_train] if seq_lengths is None else [X_train, y_train, sl_train] 
        valid_arrays = [X_test, y_test] if seq_lengths is None else [X_test, y_test, sl_test]
        train_tensors = [torch.Tensor(t) for t in train_arrays]
        valid_tensors = [torch.Tensor(t) for t in valid_arrays]
        dls = DataLoaders(get_dl_from_tensors(*train_tensors, bs=bs), get_dl_from_tensors(*valid_tensors, bs=bs))
        model, opt, loss_func = model_provider.create(hp=hp, device=device)
        #print('Model = ', model)
        stats = trainer.train(train_length, model, dls, loss_func, opt, metric=metric, device=device, **train_params)
        stats_by_fold.append(stats)        
        model = None
        opt = None
        gc.collect()
    return stats_by_fold
