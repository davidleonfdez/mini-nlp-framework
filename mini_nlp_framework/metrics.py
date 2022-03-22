from mini_nlp_framework.predict import predict, predict_dl, predict_multiclass
from sklearn.metrics import accuracy_score, f1_score
import torch


def clf_metric(model, dl, metric_fn=f1_score, **predict_kwargs):
    with torch.no_grad():
        preds, y = predict_dl(model, dl, **predict_kwargs)
        return metric_fn(preds, y)


def lm_metric(model, dl, metric_fn=accuracy_score, **predict_kwargs):
    with torch.no_grad():
        preds, y = predict_dl(model, dl, predict=predict_multiclass, **predict_kwargs)
        mask = y.view(-1) != 0
        return metric_fn(preds.view(-1)[mask], y.view(-1)[mask])
