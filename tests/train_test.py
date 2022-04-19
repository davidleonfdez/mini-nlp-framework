import math
from mini_nlp_framework.data import DataLoaders, get_dl_from_tensors
from mini_nlp_framework.metrics import Metric
from mini_nlp_framework.predict import predict_dl
from mini_nlp_framework.train import (
    ClipGradOptions,
    EpochTrainingStats,
    train,
    TrainLengthBestMetricEpochsAgo, 
    TrainLengthMetricWorseDuringEpochs,
    TrainLengthNEpochs,
    TrainLengthOr,
    TrainingCallback,
)
from sklearn.metrics import mean_absolute_error
from testing_utils import BiasModel
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_train_lengths():
    tl_3_epochs = TrainLengthNEpochs(3)
    tl_2_epochs_after_best = TrainLengthBestMetricEpochsAgo(2)
    tl_2_epochs_after_best_higher_better = TrainLengthBestMetricEpochsAgo(2, lower_is_better=False)
    tl_2_epochs_after_best_train = TrainLengthBestMetricEpochsAgo(2, use_valid=False)
    tl_2_epochs_worse = TrainLengthMetricWorseDuringEpochs(2)
    tl_2_epochs_worse_higher_better = TrainLengthMetricWorseDuringEpochs(2, lower_is_better=False)
    tl_2_epochs_worse_train = TrainLengthMetricWorseDuringEpochs(2, use_valid=False)
    tl_2_epochs_after_best_or_5_epochs = TrainLengthOr([TrainLengthNEpochs(5), TrainLengthBestMetricEpochsAgo(2)])
    
    assert not tl_3_epochs.must_stop(EpochTrainingStats(0.5, 0.5, 0.5, 1))
    assert not tl_3_epochs.must_stop(EpochTrainingStats(0.5, 0.5, 0.5, 2))
    assert tl_3_epochs.must_stop(EpochTrainingStats(0.5, 0.5, 0.5, 3))
    assert not tl_3_epochs.must_stop(EpochTrainingStats(0.5, 0.5, 0.5, 2))

    assert not tl_2_epochs_after_best.must_stop(EpochTrainingStats(0, 0, 0.5, 1))
    assert not tl_2_epochs_after_best.must_stop(EpochTrainingStats(0, 0, 0.6, 2))
    assert not tl_2_epochs_after_best.must_stop(EpochTrainingStats(0, 0, 0.45, 3))
    assert not tl_2_epochs_after_best.must_stop(EpochTrainingStats(0, 0, 0.45, 4))
    assert tl_2_epochs_after_best.must_stop(EpochTrainingStats(0, 0, 0.48, 5))

    assert not tl_2_epochs_after_best_higher_better.must_stop(EpochTrainingStats(0, 0, 0.5, 1))
    assert not tl_2_epochs_after_best_higher_better.must_stop(EpochTrainingStats(0, 0, 0.6, 2))
    assert not tl_2_epochs_after_best_higher_better.must_stop(EpochTrainingStats(0, 0, 0.7, 3))
    assert not tl_2_epochs_after_best_higher_better.must_stop(EpochTrainingStats(0, 0, 0.45, 4))
    assert not tl_2_epochs_after_best_higher_better.must_stop(EpochTrainingStats(0, 0, 0.72, 5))
    assert not tl_2_epochs_after_best_higher_better.must_stop(EpochTrainingStats(0, 0, 0.5, 6))
    assert tl_2_epochs_after_best_higher_better.must_stop(EpochTrainingStats(0, 0, 0.6, 7))

    assert not tl_2_epochs_after_best_train.must_stop(EpochTrainingStats(0, 1., 0.5, 1))
    assert not tl_2_epochs_after_best_train.must_stop(EpochTrainingStats(0, 0.9, 0.6, 2))
    assert not tl_2_epochs_after_best_train.must_stop(EpochTrainingStats(0, 0.8, 0.7, 3))
    assert not tl_2_epochs_after_best_train.must_stop(EpochTrainingStats(0, 0.9, 0.8, 4))
    assert not tl_2_epochs_after_best_train.must_stop(EpochTrainingStats(0, 0.75, 0.9, 5))
    assert not tl_2_epochs_after_best_train.must_stop(EpochTrainingStats(0, 0.77, 1, 6))
    assert tl_2_epochs_after_best_train.must_stop(EpochTrainingStats(0, 0.75, 1.1, 7))

    assert not tl_2_epochs_worse.must_stop(EpochTrainingStats(0, 0, 0.5, 1))
    assert not tl_2_epochs_worse.must_stop(EpochTrainingStats(0, 0, 0.6, 2))
    assert not tl_2_epochs_worse.must_stop(EpochTrainingStats(0, 0, 0.55, 3))
    assert not tl_2_epochs_worse.must_stop(EpochTrainingStats(0, 0, 0.58, 4))
    assert tl_2_epochs_worse.must_stop(EpochTrainingStats(0, 0, 0.59, 5))

    assert not tl_2_epochs_worse_higher_better.must_stop(EpochTrainingStats(0, 0, 0.5, 1))
    assert not tl_2_epochs_worse_higher_better.must_stop(EpochTrainingStats(0, 0, 0.6, 2))
    assert not tl_2_epochs_worse_higher_better.must_stop(EpochTrainingStats(0, 0, 0.7, 3))
    assert not tl_2_epochs_worse_higher_better.must_stop(EpochTrainingStats(0, 0, 0.65, 4))
    assert not tl_2_epochs_worse_higher_better.must_stop(EpochTrainingStats(0, 0, 0.67, 5))
    assert not tl_2_epochs_worse_higher_better.must_stop(EpochTrainingStats(0, 0, 0.65, 6))
    assert tl_2_epochs_worse_higher_better.must_stop(EpochTrainingStats(0, 0, 0.65, 7))

    assert not tl_2_epochs_worse_train.must_stop(EpochTrainingStats(0, 1., 0.5, 1))
    assert not tl_2_epochs_worse_train.must_stop(EpochTrainingStats(0, 0.9, 0.6, 2))
    assert not tl_2_epochs_worse_train.must_stop(EpochTrainingStats(0, 0.8, 0.7, 3))
    assert not tl_2_epochs_worse_train.must_stop(EpochTrainingStats(0, 0.9, 0.8, 4))
    assert not tl_2_epochs_worse_train.must_stop(EpochTrainingStats(0, 0.75, 0.9, 5))
    assert not tl_2_epochs_worse_train.must_stop(EpochTrainingStats(0, 0.77, 1, 6))
    assert tl_2_epochs_worse_train.must_stop(EpochTrainingStats(0, 0.79, 1.1, 7))

    assert not tl_2_epochs_after_best_or_5_epochs.must_stop(EpochTrainingStats(0, 0, 1., 1))
    assert not tl_2_epochs_after_best_or_5_epochs.must_stop(EpochTrainingStats(0, 0, 1.1, 2))
    assert tl_2_epochs_after_best_or_5_epochs.must_stop(EpochTrainingStats(0, 0, 1.2, 3))

    # Reinit because `TrainLengthBestMetricEpochsAgo` has state
    tl_2_epochs_after_best_or_5_epochs = TrainLengthOr([TrainLengthNEpochs(5), TrainLengthBestMetricEpochsAgo(2)])
    assert not tl_2_epochs_after_best_or_5_epochs.must_stop(EpochTrainingStats(0, 0, 1., 1))
    assert not tl_2_epochs_after_best_or_5_epochs.must_stop(EpochTrainingStats(0, 0, 0.9, 2))
    assert not tl_2_epochs_after_best_or_5_epochs.must_stop(EpochTrainingStats(0, 0, 0.8, 3))
    assert not tl_2_epochs_after_best_or_5_epochs.must_stop(EpochTrainingStats(0, 0.7, 1., 4))
    assert tl_2_epochs_after_best_or_5_epochs.must_stop(EpochTrainingStats(0, 0, 0.6, 5))


class RegressionMetric(Metric):
    def __init__(self, metric_fn=mean_absolute_error):
        self.metric_fn = metric_fn

    def __call__(self, model:nn.Module, dl, **predict_kwargs) -> float:
        with torch.no_grad():
            preds, y = predict_dl(model, dl, predict=lambda model, x, **kwargs: model(x).cpu(), **predict_kwargs)
            return self.metric_fn(preds, y)

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self.metric_fn.__name__


class CountCallsCallback(TrainingCallback):
    def __init__(self):
        self.n_on_step_end_calls = 0
        self.n_on_epoch_end_calls = 0

    def on_step_end(self, tr_loss: torch.Tensor, model: nn.Module, opt: torch.optim.Optimizer):
        self.n_on_step_end_calls += 1

    def on_epoch_end(self, stats: EpochTrainingStats, model: nn.Module, opt: torch.optim.Optimizer):
        self.n_on_epoch_end_calls += 1


def test_train():
    def _create_args(a_param_init, clip_norm=1.):
        train_length = TrainLengthNEpochs(3)
        model = BiasModel(a_param_init)

        with torch.no_grad():
            # Create x and y so that the param `a` should converge to -1
            x = torch.rand(10)
            y = x - 1
        dls = DataLoaders(
            train=get_dl_from_tensors(x[:7], y[:7], bs=2),
            valid=get_dl_from_tensors(x[7:], y[7:], bs=2),
        )
        loss_func = nn.MSELoss()
        # Use small lr to ensure loss always goes down
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = None #torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)
        metric = RegressionMetric()
        clip_grad = ClipGradOptions(model.parameters(), clip_norm)
        cbs = [CountCallsCallback()]

        args = [train_length, model, dls, loss_func, opt]
        kwargs = dict(sched=sched, metric=metric, clip_grad=clip_grad, callbacks=cbs)
        return args, kwargs

    args, kwargs = _create_args(10)
    stats = train(*args, **kwargs)
    model_a_10 = args[1]
    cbs = kwargs['callbacks']

    args, kwargs = _create_args(-5)
    train(*args, **kwargs)
    model_a_m5 = args[1]

    args, kwargs = _create_args(-5, clip_norm=1e-7)
    train(*args, **kwargs)
    model_a_m5_clip_small = args[1]

    # At the beginning, it behaves like an identity layer
    model_to_test_loss = BiasModel(a_init=0)
    x_to_test_loss = torch.Tensor([1., 1., 1., 5.])
    y_to_test_loss = torch.Tensor([0.]*4)
    cbs_to_test_loss = [CountCallsCallback()]
    stats_to_test_loss = train( 
        TrainLengthNEpochs(1), 
        model_to_test_loss,
        DataLoaders(train=get_dl_from_tensors(x_to_test_loss, y_to_test_loss, bs=3)),
        nn.MSELoss(),
        torch.optim.Adam(model_to_test_loss.parameters()),
        callbacks=cbs_to_test_loss,
    )

    expected_n_epochs = 3

    assert model_a_10.a < 10    
    assert model_a_m5.a > -5
    assert model_a_m5.a > model_a_m5_clip_small.a > -5
    assert len(stats.train_loss_history) == expected_n_epochs
    loss_decreases_every_epoch = all(
        loss < stats.train_loss_history[i] for i, loss in enumerate(stats.train_loss_history[1:])
    )
    assert loss_decreases_every_epoch
    train_metric_decreases_every_epoch = all(
        metric < stats.train_metric_history[i] for i, metric in enumerate(stats.train_metric_history[1:])
    )
    assert train_metric_decreases_every_epoch
    valid_metric_decreases_every_epoch = all(
        metric < stats.valid_metric_history[i] for i, metric in enumerate(stats.valid_metric_history[1:])
    )
    assert valid_metric_decreases_every_epoch
    assert cbs[0].n_on_step_end_calls == 12
    assert cbs[0].n_on_epoch_end_calls == 3

    # Check that loss calculation takes into account that batches can be asymmetric
    assert math.isclose(stats_to_test_loss.train_loss_history[0], F.mse_loss(x_to_test_loss, y_to_test_loss), abs_tol=1e-2)
    assert cbs_to_test_loss [0].n_on_step_end_calls == 2
    assert cbs_to_test_loss[0].n_on_epoch_end_calls == 1
