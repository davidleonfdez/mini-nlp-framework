from mini_nlp_framework.data import DataLoaders
from mini_nlp_framework.train import TrainingStats
from mini_nlp_framework.validate import CrossValidationStats, cross_validate
import numpy as np
import os
from testing_utils import BiasHyperParameters, BiasModel, BiasModelProvider, FakeMetric, FakeTrainer
import torch
import torch.nn as nn


def _dls_to_array(dls:DataLoaders):
    x_train, y_train, *sl_train = zip(*dls.train.dataset)
    x_valid, y_valid, *sl_valid = zip(*dls.valid.dataset)
    result_tuples = [x_train, x_valid, y_train, y_valid]
    if len(sl_train) > 0:
        result_tuples.extend([sl_train[0], sl_valid[0]])
    # `stack` converts from:
    # Tuple(Example 1, Example 2, ...) -> np.array([Example 1, Example 2, ...])
    return tuple(np.stack(arr) for arr in result_tuples)


def test_cross_validate():
    n_folds = 4
    X = np.random.randint(0, 10, (16, 30))
    y = np.random.randint(0, 2, (16,))
    seq_lengths = np.random.randint(1, 31, (16,))
    train_length = 2
    bs = 2
    bias_init = -19
    model_provider = BiasModelProvider(bias_init=bias_init)
    fake_metric = FakeMetric()
    hp = BiasHyperParameters(lr=5e-5, wd=0.1)
    device='cpu'
    fake_train_return_args = [
        # The values don't make sense, it's not needed as we are not testing the trainer
        TrainingStats(np.array([3]), np.array([0]), np.array([11]), 3, 9),
        TrainingStats(np.array([1]), np.array([8]), np.array([14]), 2, 10),
        TrainingStats(np.array([6]), np.array([4]), np.array([19]), 3, 10),
        TrainingStats(np.array([9]), np.array([5]), np.array([16]), 5, 9),
    ]
    def update_params(model:BiasModel):
        with torch.no_grad(): model.a -= 1
    trainer = FakeTrainer(fake_train_return_args, update_params)

    _, _, expected_loss_func = BiasModelProvider(bias_init=bias_init).create()
    expected_model_param = bias_init - 1

    expected_x_train_fold_0 = X[4:]
    expected_y_train_fold_0 = y[4:]
    expected_sl_train_fold_0 = seq_lengths[4:]
    expected_x_valid_fold_0 = X[:4]
    expected_y_valid_fold_0 = y[:4]
    expected_sl_valid_fold_0 = seq_lengths[:4]

    expected_x_train_fold_1 = np.concatenate([X[:4], X[8:]])
    expected_y_train_fold_1 = np.concatenate([y[:4], y[8:]])
    expected_sl_train_fold_1 = np.concatenate([seq_lengths[:4], seq_lengths[8:]])
    expected_x_valid_fold_1 = X[4:8]
    expected_y_valid_fold_1 = y[4:8]
    expected_sl_valid_fold_1 = seq_lengths[4:8]

    expected_x_train_fold_2 = np.concatenate([X[:8], X[12:]])
    expected_y_train_fold_2 = np.concatenate([y[:8], y[12:]])
    expected_sl_train_fold_2 = np.concatenate([seq_lengths[:8], seq_lengths[12:]])
    expected_x_valid_fold_2 = X[8:12]
    expected_y_valid_fold_2 = y[8:12]
    expected_sl_valid_fold_2 = seq_lengths[8:12]

    expected_x_train_fold_3 = X[:12]
    expected_y_train_fold_3 = y[:12]
    expected_sl_train_fold_3 = seq_lengths[:12]
    expected_x_valid_fold_3 = X[12:]
    expected_y_valid_fold_3 = y[12:]
    expected_sl_valid_fold_3 = seq_lengths[12:]


    ##################################
    # A) test without sequence lengths
    ##################################
    cv_stats = cross_validate(
        model_provider, X, y, None, nfolds=n_folds, train_length=train_length, bs=bs, trainer=trainer, 
        metric=fake_metric, hp=hp, device=device,
    )
    stats_by_fold = cv_stats.stats_by_fold
    call_args_train_len, call_args_model, call_args_dls, call_args_loss_func, call_args_opt = zip(*trainer.call_args)
    x_train_fold_0, x_valid_fold_0, y_train_fold_0, y_valid_fold_0 = _dls_to_array(call_args_dls[0])
    x_train_fold_1, x_valid_fold_1, y_train_fold_1, y_valid_fold_1 = _dls_to_array(call_args_dls[1])
    x_train_fold_2, x_valid_fold_2, y_train_fold_2, y_valid_fold_2 = _dls_to_array(call_args_dls[2])
    x_train_fold_3, x_valid_fold_3, y_train_fold_3, y_valid_fold_3 = _dls_to_array(call_args_dls[3])    

    assert stats_by_fold == fake_train_return_args
    assert len(trainer.call_args) == len(trainer.call_kwargs) == n_folds
    assert all(train_len_fold_i == train_length for train_len_fold_i in call_args_train_len)
    model_is_reinit_every_fold = all(model_fold_i.a == expected_model_param for model_fold_i in call_args_model)
    assert model_is_reinit_every_fold
    assert all(isinstance(loss_func_fold_i, type(expected_loss_func)) for loss_func_fold_i in call_args_loss_func)
    assert all(call_kwargs['metric'] == fake_metric for call_kwargs in trainer.call_kwargs)
    assert all(call_kwargs['device'] == device for call_kwargs in trainer.call_kwargs)
    assert all(create_args[0] == hp for create_args in model_provider.create_call_args)
    assert all(
        (dls_fold_i.train.batch_size, dls_fold_i.valid.batch_size) == (bs, bs) for dls_fold_i in call_args_dls
    )

    assert x_train_fold_0.dtype == expected_x_train_fold_0.dtype
    assert np.array_equal(x_train_fold_0, expected_x_train_fold_0)
    assert y_train_fold_0.dtype == expected_y_train_fold_0.dtype
    assert np.array_equal(y_train_fold_0, expected_y_train_fold_0)
    assert x_valid_fold_0.dtype == expected_x_valid_fold_0.dtype
    assert np.array_equal(x_valid_fold_0, expected_x_valid_fold_0)
    assert y_valid_fold_0.dtype == expected_y_valid_fold_0.dtype
    assert np.array_equal(y_valid_fold_0, expected_y_valid_fold_0)

    assert np.array_equal(x_train_fold_1, expected_x_train_fold_1)
    assert np.array_equal(y_train_fold_1, expected_y_train_fold_1)
    assert np.array_equal(x_valid_fold_1, expected_x_valid_fold_1)
    assert np.array_equal(y_valid_fold_1, expected_y_valid_fold_1)

    assert np.array_equal(x_train_fold_2, expected_x_train_fold_2)
    assert np.array_equal(y_train_fold_2, expected_y_train_fold_2)
    assert np.array_equal(x_valid_fold_2, expected_x_valid_fold_2)
    assert np.array_equal(y_valid_fold_2, expected_y_valid_fold_2)

    assert np.array_equal(x_train_fold_3, expected_x_train_fold_3)
    assert np.array_equal(y_train_fold_3, expected_y_train_fold_3)
    assert np.array_equal(x_valid_fold_3, expected_x_valid_fold_3)
    assert np.array_equal(y_valid_fold_3, expected_y_valid_fold_3)


    ###############################
    # B) Test with sequence lengths
    ###############################
    trainer = FakeTrainer(fake_train_return_args, update_params)
    model_provider = BiasModelProvider()
    cv_stats = cross_validate(
        model_provider, X, y, seq_lengths, nfolds=n_folds, train_length=train_length, bs=bs, trainer=trainer, 
        metric=fake_metric, hp=hp, device=device,
    )
    stats_by_fold = cv_stats.stats_by_fold
    call_args_dls = [call_args_fold_i[2] for call_args_fold_i in trainer.call_args]
    x_train_fold_0, x_valid_fold_0, y_train_fold_0, y_valid_fold_0, sl_train_fold_0, sl_valid_fold_0 = _dls_to_array(call_args_dls[0])
    x_train_fold_1, x_valid_fold_1, y_train_fold_1, y_valid_fold_1, sl_train_fold_1, sl_valid_fold_1 = _dls_to_array(call_args_dls[1])
    x_train_fold_2, x_valid_fold_2, y_train_fold_2, y_valid_fold_2, sl_train_fold_2, sl_valid_fold_2 = _dls_to_array(call_args_dls[2])
    x_train_fold_3, x_valid_fold_3, y_train_fold_3, y_valid_fold_3, sl_train_fold_3, sl_valid_fold_3 = _dls_to_array(call_args_dls[3])    

    assert x_train_fold_0.dtype == expected_x_train_fold_0.dtype
    assert np.array_equal(x_train_fold_0, expected_x_train_fold_0)
    assert y_train_fold_0.dtype == expected_y_train_fold_0.dtype
    assert np.array_equal(y_train_fold_0, expected_y_train_fold_0)
    assert sl_train_fold_0.dtype == expected_sl_train_fold_0.dtype
    assert np.array_equal(sl_train_fold_0, expected_sl_train_fold_0)
    assert x_valid_fold_0.dtype == expected_x_valid_fold_0.dtype
    assert np.array_equal(x_valid_fold_0, expected_x_valid_fold_0)
    assert y_valid_fold_0.dtype == expected_y_valid_fold_0.dtype
    assert np.array_equal(y_valid_fold_0, expected_y_valid_fold_0)
    assert sl_valid_fold_0.dtype == expected_sl_valid_fold_0.dtype
    assert np.array_equal(sl_valid_fold_0, expected_sl_valid_fold_0)

    assert np.array_equal(x_train_fold_1, expected_x_train_fold_1)
    assert np.array_equal(y_train_fold_1, expected_y_train_fold_1)
    assert np.array_equal(sl_train_fold_1, expected_sl_train_fold_1)
    assert np.array_equal(x_valid_fold_1, expected_x_valid_fold_1)
    assert np.array_equal(y_valid_fold_1, expected_y_valid_fold_1)
    assert np.array_equal(sl_valid_fold_1, expected_sl_valid_fold_1)

    assert np.array_equal(x_train_fold_2, expected_x_train_fold_2)
    assert np.array_equal(y_train_fold_2, expected_y_train_fold_2)
    assert np.array_equal(sl_train_fold_2, expected_sl_train_fold_2)
    assert np.array_equal(x_valid_fold_2, expected_x_valid_fold_2)
    assert np.array_equal(y_valid_fold_2, expected_y_valid_fold_2)
    assert np.array_equal(sl_valid_fold_2, expected_sl_valid_fold_2)

    assert np.array_equal(x_train_fold_3, expected_x_train_fold_3)
    assert np.array_equal(y_train_fold_3, expected_y_train_fold_3)
    assert np.array_equal(sl_train_fold_3, expected_sl_train_fold_3)
    assert np.array_equal(x_valid_fold_3, expected_x_valid_fold_3)
    assert np.array_equal(y_valid_fold_3, expected_y_valid_fold_3)
    assert np.array_equal(sl_valid_fold_3, expected_sl_valid_fold_3)


def test_cross_validation_stats():
    stats_by_fold = [
        TrainingStats(
            np.random.rand(5),
            np.random.rand(5),
            np.array([1., 1.5, 0.5, 2.5, 0.75]),
            5,
            10
        ),
        TrainingStats(
            np.random.rand(5),
            np.random.rand(5),
            np.array([5., 6., 7., 8., 9.]),
            5,
            10
        ),
        TrainingStats(
            np.random.rand(5),
            np.random.rand(5),
            np.array([1.5, 2., 3.5, 1., 0.5]),
            5,
            10
        ),
    ]
    stats_lower_better = CrossValidationStats(stats_by_fold, metric_lower_is_better=True)
    stats_higher_better = CrossValidationStats(stats_by_fold, metric_lower_is_better=False)

    assert stats_lower_better.best_metric_and_epoch_by_fold == [(0.5, 2), (5., 0), (0.5, 4)]
    assert stats_lower_better.avg_best_metric_and_epoch == (2., 2.)
    assert stats_higher_better.best_metric_and_epoch_by_fold == [(2.5, 3), (9, 4), (3.5, 2)]
    assert stats_higher_better.avg_best_metric_and_epoch == (5., 3.)
    expected_str = os.linesep.join([
        "--- Cross validation statistics ---",
        "Average of best metric by fold: 5.0",
        "Average epoch of best metric by fold: 3.0",
        "Best metric by fold:",
        "    Fold 0: 2.5 @ epoch 3",
        "    Fold 1: 9.0 @ epoch 4",
        "    Fold 2: 3.5 @ epoch 2"
    ])
    assert str(stats_higher_better) == expected_str
