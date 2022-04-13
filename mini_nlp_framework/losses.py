import torch
import torch.nn.functional as F
from typing import Callable, List


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def flat_cross_entropy_loss(preds, target, ignore_index=-100):
    #return F.cross_entropy(preds.view(-1, len(vocab.idx_to_word)), target.view(-1),
    return F.cross_entropy(preds.view(-1, preds.shape[-1]), target.view(-1),
                           ignore_index=ignore_index)


def flat_binary_cross_entropy_loss(preds, target):
    return F.binary_cross_entropy_with_logits(preds.view(-1), target.view(-1))


class ComposedLoss:
    def __init__(self, losses:List[LossFunction]): 
        self.losses = losses
        
    def __call__(self, preds, target):
        return torch.stack([l(preds, target) for l in self.losses]).sum()
