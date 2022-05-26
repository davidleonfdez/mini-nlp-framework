import torch


def predict_binary(model, x, thresh=0.5):
    """
    Predict a class (0, 1) for each input (first dimension of `x`) using the classifier  `model`.

    Args:
        model: binary classifier module that outputs a logit for every input sequence.
        x: input tensor.
        thresh: value of logits that separates positive (above thresh -> 1) and negative (below thresh -> 0) classes.
    Returns
        Integer tensor that contains the predicted classes (0 or 1). Its shape is the same as the output of the model.
    """
    preds = model(x).cpu()
    pred_classes = (preds > thresh).int()
    return pred_classes


def predict_multiclass(model, x, *extra_xs):
    """
    Predict a class (int in [0...(n classes - 1)]) for each input (first dim of `x`) using the classifier `model`.

    Args:
        model: multiclass classifier module that outputs "n classes" logits for every input sequence.
        x: input tensor.
        extra_xs: additional inputs of `model`.
    Returns
        Integer tensor of shape `(x.size(0),)` that contains the predicted classes.
    """
    preds = model(x, *extra_xs).cpu()
    pred_classes = preds.argmax(dim=-1)
    return pred_classes


def predict_dl(model, dl, predict=predict_binary, device=None, **predict_kwargs):
    """
    Collect the predictions of `model` for every input given by `dl` according to predictor method `predict`.

    The unused kwargs are forwarded to `predict`.
    """
    preds = None
    y = []
    preds = []
    for xb, yb, *extra_xs in dl:
        if device is not None: xb = xb.to(device)
        new_preds = predict(model, xb, *extra_xs, **predict_kwargs)
        preds.append(new_preds)
        y.append(yb)
    y = torch.cat(y)
    preds = torch.cat(preds)
    return preds, y
