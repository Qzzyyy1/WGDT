import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.99, normalize=True, warmup_epochs=5, temperature=1.0):
        super().__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.normalize = normalize
        self.warmup_epochs = warmup_epochs
        self.temperature = temperature

        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('initialized', torch.zeros(num_classes, dtype=torch.bool))

    def reset(self):
        self.prototypes.zero_()
        self.initialized.zero_()

    def _normalize(self, x):
        if not self.normalize:
            return x
        return F.normalize(x, p=2, dim=-1)

    def get_momentum(self, epoch=None):
        if epoch is not None and epoch < self.warmup_epochs:
            return 0.0
        return self.momentum

    @torch.no_grad()
    def initialize(self, prototypes):
        prototypes = self._normalize(prototypes.detach())
        self.prototypes.copy_(prototypes)
        self.initialized.fill_(True)

    @torch.no_grad()
    def update(self, features, labels, epoch=None):
        features = self._normalize(features.detach())
        momentum = self.get_momentum(epoch)

        for class_index in range(self.num_classes):
            class_mask = labels == class_index
            if class_mask.sum() == 0:
                continue

            class_mean = features[class_mask].mean(dim=0)
            class_mean = self._normalize(class_mean.unsqueeze(0)).squeeze(0)

            if not self.initialized[class_index] or momentum == 0.0:
                self.prototypes[class_index] = class_mean
            else:
                prototype = momentum * self.prototypes[class_index] + (1.0 - momentum) * class_mean
                self.prototypes[class_index] = self._normalize(prototype.unsqueeze(0)).squeeze(0)
            self.initialized[class_index] = True

    def get_prototypes(self):
        return self._normalize(self.prototypes)

    def compute_distance(self, features):
        features = self._normalize(features)
        prototypes = self.get_prototypes()
        return torch.cdist(features, prototypes)

    def compute_logits(self, features):
        distance = self.compute_distance(features)
        return -distance / self.temperature

    def predict(self, features):
        logits = self.compute_logits(features)
        prediction = logits.argmax(dim=1)
        return {
            'logits': logits,
            'prediction': prediction,
            'distance': -logits * self.temperature,
        }

    def compactness_loss(self, features, labels):
        distance = self.compute_distance(features)
        positive_distance = torch.gather(distance, 1, labels.view(-1, 1)).view(-1)
        return positive_distance.mean()
