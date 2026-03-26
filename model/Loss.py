import torch
import torch.nn as nn

from typing import Optional

class MarginMSELoss(nn.Module):
    def __init__(self, margin):
        super(MarginMSELoss, self).__init__()

        self.margin = margin
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss =  self.loss(input, target + self.margin)
        if weight is not None:
            loss = weight * loss

        return loss.mean()
        
