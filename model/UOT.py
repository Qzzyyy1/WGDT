import torch
import torch.nn as nn
import torch.nn.functional as F


class UOTSolver(nn.Module):
    def __init__(self, epsilon=0.05, tau_source=0.95, tau_target=0.95, max_iter=30,
                 metric='euclidean', dustbin_quantile=0.8, dustbin_beta=0.95,
                 dustbin_mass_prior=0.05, use_no_grad=True, eps=1e-8):
        super().__init__()

        self.epsilon = epsilon
        self.tau_source = tau_source
        self.tau_target = tau_target
        self.max_iter = max_iter
        self.metric = metric
        self.dustbin_quantile = dustbin_quantile
        self.dustbin_beta = dustbin_beta
        self.dustbin_mass_prior = dustbin_mass_prior
        self.use_no_grad = use_no_grad
        self.eps = eps

        self.register_buffer('dustbin_cost_ema', torch.tensor(0.0))
        self.register_buffer('dustbin_initialized', torch.tensor(False, dtype=torch.bool))

    def distance_unknown_score(self, min_distance, dustbin_cost):
        distance_margin = min_distance - dustbin_cost
        temperature = max(self.epsilon, self.eps)
        return torch.sigmoid(distance_margin / temperature)

    def pairwise_cost(self, source_proto, target_feat):
        if self.metric == 'cosine':
            source_proto = F.normalize(source_proto, p=2, dim=-1)
            target_feat = F.normalize(target_feat, p=2, dim=-1)
            similarity = torch.matmul(source_proto, target_feat.transpose(0, 1))
            return 1.0 - similarity
        return torch.cdist(source_proto, target_feat)

    @torch.no_grad()
    def estimate_dustbin_cost(self, cost):
        q_batch = torch.quantile(cost.detach().reshape(-1), self.dustbin_quantile)
        if not self.dustbin_initialized:
            self.dustbin_cost_ema.copy_(q_batch)
            self.dustbin_initialized.fill_(True)
        else:
            ema_cost = self.dustbin_beta * self.dustbin_cost_ema + (1.0 - self.dustbin_beta) * q_batch
            self.dustbin_cost_ema.copy_(ema_cost)
        return self.dustbin_beta * self.dustbin_cost_ema + (1.0 - self.dustbin_beta) * q_batch

    def append_dustbin(self, cost, dustbin_cost):
        dustbin_row = torch.full((1, cost.size(1)), float(dustbin_cost), device=cost.device, dtype=cost.dtype)
        return torch.cat([cost, dustbin_row], dim=0)

    def build_mass(self, row_num, col_num, device, dtype):
        known_rows = row_num - 1
        dustbin_mass = min(max(self.dustbin_mass_prior, self.eps), 1.0 - self.eps)
        known_mass = max(1.0 - dustbin_mass, self.eps)

        source_mass = torch.full((row_num,), 0.0, device=device, dtype=dtype)
        source_mass[:-1] = known_mass / max(known_rows, 1)
        source_mass[-1] = dustbin_mass
        target_mass = torch.full((col_num,), 1.0 / col_num, device=device, dtype=dtype)
        return source_mass, target_mass

    def sinkhorn_unbalanced(self, cost, source_mass, target_mass):
        kernel = torch.exp(-cost / max(self.epsilon, self.eps)).clamp_min(self.eps)
        u = torch.ones_like(source_mass)
        v = torch.ones_like(target_mass)

        for _ in range(self.max_iter):
            kv = torch.matmul(kernel, v).clamp_min(self.eps)
            u = torch.pow(source_mass / kv, self.tau_source)
            ktu = torch.matmul(kernel.transpose(0, 1), u).clamp_min(self.eps)
            v = torch.pow(target_mass / ktu, self.tau_target)

        return u.unsqueeze(1) * kernel * v.unsqueeze(0)

    def extract_scores(self, transport_plan):
        column_mass = transport_plan.sum(dim=0).clamp_min(self.eps)
        class_scores = transport_plan[:-1].transpose(0, 1)
        class_scores = class_scores / class_scores.sum(dim=1, keepdim=True).clamp_min(self.eps)
        dustbin_scores = transport_plan[-1] / column_mass
        best_known_mass = class_scores.max(dim=1)[0]
        mass_unknown_score = dustbin_scores / (dustbin_scores + best_known_mass + self.eps)
        assignment = class_scores.argmax(dim=1)
        return {
            'class_scores': class_scores,
            'dustbin_scores': dustbin_scores,
            'best_known_mass': best_known_mass,
            'mass_unknown_score': mass_unknown_score,
            'assignment': assignment,
        }

    def forward(self, source_proto, target_feat):
        cost = self.pairwise_cost(source_proto, target_feat)
        dustbin_cost = self.estimate_dustbin_cost(cost)
        min_distance = cost.min(dim=0)[0]
        distance_unknown_score = self.distance_unknown_score(min_distance, dustbin_cost)
        cost_ext = self.append_dustbin(cost, dustbin_cost)
        source_mass, target_mass = self.build_mass(cost_ext.size(0), cost_ext.size(1), cost_ext.device, cost_ext.dtype)

        if self.use_no_grad:
            with torch.no_grad():
                transport_plan = self.sinkhorn_unbalanced(cost_ext.detach(), source_mass, target_mass)
        else:
            transport_plan = self.sinkhorn_unbalanced(cost_ext, source_mass, target_mass)

        loss = (transport_plan.detach() * cost_ext).sum()
        scores = self.extract_scores(transport_plan.detach())
        target_dustbin_ratio = transport_plan[-1].sum() / transport_plan.sum().clamp_min(self.eps)
        return {
            'cost': cost,
            'cost_ext': cost_ext,
            'min_distance': min_distance.detach(),
            'transport_plan': transport_plan.detach(),
            'loss': loss,
            'dustbin_cost': dustbin_cost.detach(),
            'dustbin_mass_prior': torch.tensor(self.dustbin_mass_prior, device=cost.device, dtype=cost.dtype),
            'target_dustbin_ratio': target_dustbin_ratio.detach(),
            'distance_unknown_score': distance_unknown_score.detach(),
            **scores,
        }
