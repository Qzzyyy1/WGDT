import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

class Anchor(nn.Module):
    def __init__(self, num_classes, anchor_weight=10.0, alpha=1.0):
        super(Anchor, self).__init__()

        self.num_classes = num_classes
        self.anchor_weight = anchor_weight
        self.alpha = alpha

        self.anchor = nn.Parameter(torch.eye(self.num_classes) * self.anchor_weight, requires_grad=False)

    def distance_classifier(self, x):

        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d)
        anchor = self.anchor.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x - anchor, 2, 2)

        return dists

    def CACLoss(self, distances, gt):
        device = self.anchor.device
        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
        non_gt = torch.Tensor([[i for i in range(self.num_classes) if gt[x] != i] for x in range(len(distances))]).long().to(device)
        others = torch.gather(distances, 1, non_gt)

        anchor = torch.mean(true)

        tuplet = torch.exp(- others + true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

        total = self.alpha * anchor + tuplet

        return {
            'loss': total,
            'loss_anchor': anchor,
            'loss_tuplet': tuplet
        }

    def forward(self, logits: torch.Tensor, y: Optional[torch.Tensor] = None):
        distance = self.distance_classifier(logits)

        y_hat = F.softmin(distance, -1)
        prediction = y_hat.argmax(1)
        gamma = distance * (1 - F.softmin(distance, -1))

        dic = {
            'logits': logits,
            'distance': distance,
            'gamma': gamma,
            'y_hat': y_hat,
            'prediction': prediction
        }

        if y is not None:

            dic = {
                **dic,
                **self.CACLoss(distance, y)
            }

        return dic
