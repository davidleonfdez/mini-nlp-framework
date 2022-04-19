import torch


def predict_binary(model, x, thresh=0.5):
    preds = model(x).cpu()
    pred_classes = (preds > thresh).int()
    return pred_classes


def predict_multiclass(model, x, *extra_xs):
    preds = model(x, *extra_xs).cpu()
    pred_classes = preds.argmax(dim=-1)
    return pred_classes


def predict_dl(model, dl, predict=predict_binary, device=None, **predict_kwargs):
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
