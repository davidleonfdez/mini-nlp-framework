import torch
import torch.nn.functional as F
from typing import Callable, List


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def flat_cross_entropy_loss(preds, target, ignore_index=-100):
    """Cross entropy loss that accepts `preds` with rank > 2 and `target` with rank > 1.
    
    It assumes the size of the last dimension of `preds` corresponds to the number of classes and the other
    dimensions can be mixed by flattening.

    Args:
        preds: logits of size (*, n classes)
        target: labels. Its flattened shape must be equal to the flattened shape of all but the last dimension of
            `preds`.
    """
    target = target.long()
    return F.cross_entropy(preds.view(-1, preds.shape[-1]), target.view(-1),
                           ignore_index=ignore_index)


def flat_binary_cross_entropy_loss(preds, target):
    """Binary cross entropy loss that accepts `preds` and `target` with rank greater than 1.
    
    Args:
        preds: logits. Its flattened shape must be equal to the flattened shape of `labels`.
        target: labels with value between 0 and 1. Its flattened shape must be equal to the flattened shape of `preds`.
    """
    target = target.float()
    return F.binary_cross_entropy_with_logits(preds.view(-1), target.view(-1))


class ComposedLoss:
    """Callable loss that returns the sum of the results of its wrapped `losses`."""
    def __init__(self, losses:List[LossFunction]): 
        self.losses = losses
        
    def __call__(self, preds, target):
        return torch.stack([l(preds, target) for l in self.losses]).sum()
