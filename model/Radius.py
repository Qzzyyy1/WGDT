import torch
import torch.nn as nn
from typing import Optional

from model.Loss import MarginMSELoss

class Radius(nn.Module):
    def __init__(self, init_radius, margin, radius_type):
        super(Radius, self).__init__()

        dic = {
            'MarginMSELoss': {
                'loss': lambda: MarginMSELoss(margin),
                'forward': self.margin_mse_loss
            }
        }

        self.radius = nn.Parameter(torch.tensor([init_radius,]), requires_grad=True)
        self.loss = dic[radius_type]['loss']()
        self.forward = dic[radius_type]['forward']
    
    def margin_ranking_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.loss(self.radius, x, torch.ones_like(x, device=x.device))
    
    def margin_mse_loss(self, x: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.loss(self.radius, x, weight)
