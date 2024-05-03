import torch
from torch import nn


class AbsLoss(nn.Module):
    def __init__(self):
        super(AbsLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = torch.abs(targets - outputs) / targets
        return loss.mean()
