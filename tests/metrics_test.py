from mini_nlp_framework.data import get_dl_from_tensors
from mini_nlp_framework.layers import Lambda
from mini_nlp_framework.metrics import ClassificationMetric, LanguageModelMetric
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from testing_utils import ModClassifier
import torch


def test_classification_metric():
    model = ModClassifier(n_classes=2)
    X = torch.tensor([
        [0, 1, 4, 5],
        [2, 2, 2, 3],
        [5, 6, 7, 9],
        [5, 6, 7, 9],
        [0, 0, 1, 0]
    ])
    y = torch.tensor([0., 0, 1, 0, 1])
    dl = get_dl_from_tensors(X, y, bs=2)

    metric = ClassificationMetric()
    metric_value = metric(model, dl)

    expected_preds = np.array([0, 1, 1, 1, 1])
    expected = f1_score(y, expected_preds)
    assert metric_value == expected
    assert not metric.lower_is_better


def test_language_model_metric():
    model = Lambda(lambda x: -x)
    X = torch.tensor([
        [[0., 1, 2], [1, 1, 0], [4, 0, 1], [0, 0, -5]], 
        [[-3, -4, -1], [-1, 1, 1], [4, 2, 1], [1, 5, 3]], 
        [[0., 1, 2], [1, 1, 0], [4, 0, 1], [0, 0, -5]], 
        [[0, 0, -2], [2, 1, 2], [2, 3, 4], [7, 5, 6]], 
        [[0., 1, 2], [1, 1, 0], [4, 0, 1], [0, 0, -5]], 
    ])
    y = torch.tensor([
        [0, 1, 2, 2],
        [2, 1, 0, 0],
        [1, 1, 1, 1],
        [2, 1, 0, 2],
        [0, 0, 0, 0]
    ])
    dl = get_dl_from_tensors(X, y, bs=2)

    metric = LanguageModelMetric(pad_idx=-1)
    metric_value = metric(model, dl)
    expected_preds = np.array([
        [0, 2, 1, 2],
        [1, 0, 2, 0],
        [0, 2, 1, 2],
        [2, 1, 0, 1],
        [0, 2, 1, 2],
    ])
    expected = accuracy_score(y.view(-1), expected_preds.reshape(-1))
    assert metric_value == expected
    assert not metric.lower_is_better

    metric_pad = LanguageModelMetric(pad_idx=2)
    metric_value_pad = metric_pad(model, dl)
    expected_preds = np.array(
        [0, 2]
        + [0, 2, 0]
        + [0, 2, 1, 2]
        + [1, 0]
        + [0, 2, 1, 2]
    )
    expected_y_no_pad = np.array(
        [0, 1]
        + [1, 0, 0]
        + [1, 1, 1, 1]
        + [1, 0]
        + [0, 0, 0, 0]
    )
    expected = accuracy_score(expected_y_no_pad, expected_preds)
    assert metric_value_pad == expected
