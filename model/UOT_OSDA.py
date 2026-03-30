import copy
import os
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from utils.file import check_path
from utils.meter import OpensetDomainMetric, computeOpensetDomainResult
from utils.Trainer import Trainer
from utils.dataLoader import CombinedLoader
from utils.open_set import predict_open_set, predict_open_set_transport, fuse_unknown_score
from utils.utils import mergeArgs, getCliOverrideKeys
from utils.pyExt import dataToDevice
from .DCRN import DCRN
from .Prototype import PrototypeMemory
from .UOT import UOTSolver
from .Anchor import Anchor
from .Radius import Radius
from utils.dann import DomainDiscriminator, DomainAdversarialLoss


class Model(nn.Module):
    def __init__(self, args, source_info, target_info, device, in_channels, patch, known_num_classes):
        super().__init__()

        self.args = args
        self.source_info = source_info
        self.target_info = target_info
        self.device = device
        self.in_channels = in_channels
        self.patch = patch
        self.known_num_classes = known_num_classes

        self.feature_encoder = DCRN(in_channels, patch, known_num_classes)
        self.source_classifier = nn.Linear(288, known_num_classes)
        self.anchor = Anchor(
            known_num_classes,
            anchor_weight=args.anchor_weight,
            alpha=args.alpha,
        )
        self.radius = Radius(
            args.learnable_radius_init,
            args.learnable_radius_margin,
            'MarginMSELoss',
        )
        self.disc_encoder = DomainDiscriminator(in_feature=288, hidden_size=args.dann_hidden_size)
        self.domain_adv = DomainAdversarialLoss(self.disc_encoder)
        self.prototype_memory = PrototypeMemory(
            num_classes=known_num_classes,
            feat_dim=288,
            momentum=args.prototype_momentum,
            normalize=True,
            warmup_epochs=args.prototype_warmup_epochs,
            temperature=args.prototype_temperature,
        )
        self.uot_solver = UOTSolver(
            epsilon=args.uot_epsilon,
            tau_source=args.uot_tau_source,
            tau_target=args.uot_tau_target,
            max_iter=args.uot_max_iter,
            metric=args.uot_metric,
            dustbin_quantile=args.dustbin_quantile,
            dustbin_beta=args.dustbin_beta,
            dustbin_mass_prior=args.dustbin_mass_prior,
            dustbin_cost_mode=args.dustbin_cost_mode,
            dustbin_cost_value=args.dustbin_cost_value,
            dustbin_source_quantile=args.dustbin_source_quantile,
            dustbin_source_margin=args.dustbin_source_margin,
            use_no_grad=args.uot_no_grad == 'True',
        )

        self.source_oa = Accuracy()
        self.metric = OpensetDomainMetric(self.known_num_classes, self.args)
        self.prediciton_all = []

        self.register_buffer('running_threshold', torch.tensor(float(args.unknown_threshold)))
        self.register_buffer('threshold_initialized', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('class_radius', torch.zeros(known_num_classes))
        self.register_buffer('class_radius_initialized', torch.zeros(known_num_classes, dtype=torch.bool))

        self.source_dustbin_score_list = []
        self.target_dustbin_ratio_list = []
        self.best_oracle_hscore = float('-inf')
        self.oracle_loader = None
        self.uot_warmup_notice_printed = False
        self.dann_warmup_notice_printed = False
        self.dann_stop_notice_printed = False

        self.use_ema_teacher = args.use_ema_teacher == 'True'
        self.teacher_initialized = False
        if self.use_ema_teacher:
            self.teacher_feature_encoder = copy.deepcopy(self.feature_encoder)
            self.teacher_prototype_memory = copy.deepcopy(self.prototype_memory)
            self._freeze_teacher_modules()
            self.sync_teacher_from_student()
        else:
            self.teacher_feature_encoder = None
            self.teacher_prototype_memory = None

        self.use_eval_ema = args.use_eval_ema == 'True'
        self.eval_ema_initialized = False
        self.eval_ema_state = {}
        self.eval_ema_backup = None

    def get_unknown_threshold(self):
        return float(self.running_threshold.item())

    def compute_feature_prototype_distance(self, features):
        prototypes = self.prototype_memory.get_prototypes().detach()
        return self.uot_solver.pairwise_cost(prototypes, features).transpose(0, 1)

    def is_prototype_update_active(self, epoch):
        if epoch is None:
            return True
        return self.args.proto_update_stop_epoch < 0 or epoch < self.args.proto_update_stop_epoch

    def get_source_decay_factor(self, epoch):
        if epoch is None:
            return 1.0
        if self.args.source_decay_epoch < 0 or epoch < self.args.source_decay_epoch:
            return 1.0
        return self.args.source_decay_factor

    def _freeze_teacher_modules(self):
        if not self.use_ema_teacher:
            return
        for parameter in self.teacher_feature_encoder.parameters():
            parameter.requires_grad = False
        self.teacher_feature_encoder.eval()

    def _iter_eval_ema_tensors(self):
        tracked_prefixes = (
            'feature_encoder.',
            'source_classifier.',
            'anchor.',
            'radius.',
            'prototype_memory.',
            'uot_solver.',
        )
        tracked_buffers = {
            'running_threshold',
            'threshold_initialized',
            'class_radius',
            'class_radius_initialized',
        }
        for name, parameter in self.named_parameters():
            if name.startswith('teacher_'):
                continue
            if name.startswith(tracked_prefixes):
                yield name, parameter
        for name, buffer in self.named_buffers():
            if name.startswith('teacher_') or name.startswith('source_oa.') or name.startswith('metric.'):
                continue
            if name in tracked_buffers or name.startswith(tracked_prefixes):
                yield name, buffer

    @torch.no_grad()
    def init_eval_ema_state(self):
        self.eval_ema_state = {}
        for name, tensor in self._iter_eval_ema_tensors():
            self.eval_ema_state[name] = tensor.detach().clone()
        self.eval_ema_initialized = True

    @torch.no_grad()
    def update_eval_ema_state(self):
        if not self.use_eval_ema:
            return
        if not self.eval_ema_initialized:
            self.init_eval_ema_state()
            return
        decay = float(self.args.eval_ema_decay)
        for name, tensor in self._iter_eval_ema_tensors():
            if name not in self.eval_ema_state:
                self.eval_ema_state[name] = tensor.detach().clone()
                continue
            if torch.is_floating_point(tensor):
                self.eval_ema_state[name].mul_(decay).add_(tensor.detach(), alpha=1.0 - decay)
            else:
                self.eval_ema_state[name].copy_(tensor.detach())

    @torch.no_grad()
    def apply_eval_ema_state(self):
        if not self.use_eval_ema or not self.eval_ema_initialized or self.eval_ema_backup is not None:
            return
        self.eval_ema_backup = {}
        for name, tensor in self._iter_eval_ema_tensors():
            self.eval_ema_backup[name] = tensor.detach().clone()
            tensor.copy_(self.eval_ema_state[name])

    @torch.no_grad()
    def restore_eval_ema_state(self):
        if self.eval_ema_backup is None:
            return
        for name, tensor in self._iter_eval_ema_tensors():
            tensor.copy_(self.eval_ema_backup[name])
        self.eval_ema_backup = None

    @torch.no_grad()
    def sync_teacher_from_student(self):
        if not self.use_ema_teacher:
            return
        self.teacher_feature_encoder.load_state_dict(self.feature_encoder.state_dict())
        self.teacher_prototype_memory.load_state_dict(self.prototype_memory.state_dict())
        self.teacher_feature_encoder.eval()
        self.teacher_initialized = True

    @torch.no_grad()
    def update_ema_teacher(self):
        if not self.use_ema_teacher:
            return
        if not self.teacher_initialized:
            self.sync_teacher_from_student()
            return

        momentum = float(self.args.teacher_momentum)
        for teacher_param, student_param in zip(self.teacher_feature_encoder.parameters(), self.feature_encoder.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_buffer, student_buffer in zip(self.teacher_feature_encoder.buffers(), self.feature_encoder.buffers()):
            if torch.is_floating_point(teacher_buffer):
                teacher_buffer.data.mul_(momentum).add_(student_buffer.data, alpha=1.0 - momentum)
            else:
                teacher_buffer.data.copy_(student_buffer.data)

        if bool(self.teacher_prototype_memory.initialized.any().item()):
            self.teacher_prototype_memory.prototypes.mul_(momentum).add_(self.prototype_memory.prototypes, alpha=1.0 - momentum)
        else:
            self.teacher_prototype_memory.prototypes.copy_(self.prototype_memory.prototypes)
        self.teacher_prototype_memory.initialized.copy_(self.teacher_prototype_memory.initialized | self.prototype_memory.initialized)
        self.teacher_feature_encoder.eval()

    def compute_soft_dann_target_weights(self, out):
        distance_matrix = self.compute_feature_prototype_distance(out['features'])
        top_dists, top_indices = distance_matrix.topk(2, dim=1, largest=False)
        d1 = top_dists[:, 0]
        d2 = top_dists[:, 1]
        top1_class = top_indices[:, 0]

        candidate_radius = self.class_radius[top1_class] * self.args.radius_margin
        candidate_radius_initialized = self.class_radius_initialized[top1_class]

        tau_close = max(float(self.args.tau_close), 1e-6)
        tau_margin = max(float(self.args.tau_margin), 1e-6)

        with torch.no_grad():
            w_open = 1.0 - out['unknown_score'].detach()
            w_close_raw = torch.sigmoid((candidate_radius.detach() - d1.detach()) / tau_close)
            w_close = torch.where(
                candidate_radius_initialized,
                w_close_raw,
                torch.ones_like(w_close_raw),
            )
            margin_gap = d2.detach() - d1.detach()
            w_margin = torch.sigmoid((margin_gap - self.args.tgt_margin_value) / tau_margin)
            w_target = (w_open * w_close * w_margin).pow(1.0 / 3.0)

        return {
            'distance_matrix': distance_matrix,
            'd1': d1,
            'd2': d2,
            'top1_class': top1_class,
            'candidate_radius': candidate_radius,
            'candidate_radius_initialized': candidate_radius_initialized,
            'margin_gap': d2 - d1,
            'w_open': w_open,
            'w_close': w_close,
            'w_margin': w_margin,
            'w_target': w_target,
        }

    def compute_wgdt_style_target_state(self, out):
        eps = 1e-8
        mode = getattr(self.args, 'radius_score_mode', 'prototype_distance')
        if mode == 'anchor_gamma':
            distance = out['distance']
            gamma = out['gamma']
            min_distance, distance_prediction = distance.min(dim=1)
            min_gamma, gamma_prediction = gamma.min(dim=1)
            score = min_gamma
            prediction = gamma_prediction
            distance_matrix = distance
            gamma_matrix = gamma
            score_mean = min_gamma.detach().mean()
            score_std = min_gamma.detach().std()
            class_scale = torch.ones_like(min_distance)
            class_scale_initialized = torch.ones_like(distance_prediction, dtype=torch.bool)
        elif mode == 'prototype_distance_classwise':
            distance_matrix = self.compute_feature_prototype_distance(out['features'])
            class_prediction = out['prediction'].detach().long()
            class_prediction = class_prediction.clamp(min=0, max=self.known_num_classes - 1)
            class_distance = distance_matrix.gather(1, class_prediction.unsqueeze(1)).squeeze(1)
            gamma_matrix = None
            raw_class_scale = self.class_radius[class_prediction].detach()
            class_scale_initialized = self.class_radius_initialized[class_prediction]
            if self.class_radius_initialized.any():
                fallback_scale = self.class_radius[self.class_radius_initialized].mean().detach()
            else:
                fallback_scale = class_distance.detach().mean()
            class_scale = torch.where(
                class_scale_initialized,
                raw_class_scale,
                torch.full_like(raw_class_scale, fallback_scale),
            ).clamp_min(eps)
            score = class_distance / class_scale
            prediction = class_prediction
            score_mean = score.detach().mean()
            score_std = score.detach().std()
            min_distance = class_distance
            distance_prediction = class_prediction
        else:
            distance_matrix = self.compute_feature_prototype_distance(out['features'])
            min_distance, distance_prediction = distance_matrix.min(dim=1)
            gamma_matrix = None
            score = min_distance
            prediction = distance_prediction
            score_mean = min_distance.detach().mean()
            score_std = min_distance.detach().std()
            class_scale = torch.ones_like(min_distance)
            class_scale_initialized = torch.ones_like(distance_prediction, dtype=torch.bool)

        score_detached = score.detach()
        score_max = score_detached.max()
        score_min = score_detached.min()
        score_range = (score_max - score_min).clamp_min(eps)
        weight = (score_max - score_detached) / score_range
        if torch.allclose(score_max, score_min):
            weight = torch.ones_like(score_detached)

        distance_detached = min_distance.detach()

        return {
            'mode': mode,
            'distance_matrix': distance_matrix,
            'gamma_matrix': gamma_matrix,
            'min_distance': min_distance,
            'score': score,
            'distance_prediction': distance_prediction,
            'prediction': prediction,
            'weight': weight,
            'score_mean': score_mean,
            'score_std': score_std,
            'distance_mean': distance_detached.mean(),
            'distance_std': distance_detached.std(),
            'class_scale_mean': class_scale.mean().detach(),
            'class_scale_std': class_scale.std().detach(),
            'class_scale_initialized_ratio': class_scale_initialized.float().mean(),
            'weight_mean': weight.mean(),
            'weight_std': weight.std(),
        }

    def weighted_mean(self, value, weight=None):
        if value.numel() == 0:
            return self.radius.radius.new_zeros(())
        if weight is None:
            return value.mean()
        weight = weight.to(value.dtype)
        return (value * weight).sum() / weight.sum().clamp_min(1e-8)

    def compute_dual_boundary_radius_state(self, wgdt_target_state):
        score = wgdt_target_state['score'].detach()
        weight = wgdt_target_state['weight'].detach().clamp(0.0, 1.0)

        positive_quantile = max(float(self.args.radius_positive_quantile), float(self.args.radius_negative_quantile))
        negative_quantile = min(float(self.args.radius_positive_quantile), float(self.args.radius_negative_quantile))
        positive_cutoff = torch.quantile(weight.float(), positive_quantile)
        negative_cutoff = torch.quantile(weight.float(), negative_quantile)

        positive_mask = weight >= positive_cutoff
        negative_mask = weight <= negative_cutoff
        if positive_mask.sum() == 0:
            positive_mask[torch.argmax(weight)] = True
        if negative_mask.sum() == 0:
            negative_mask[torch.argmin(weight)] = True

        radius_value = self.radius.radius.squeeze(0)
        positive_scores = score[positive_mask]
        negative_scores = score[negative_mask]
        positive_weight = weight[positive_mask].clamp_min(1e-8)
        negative_weight = (1.0 - weight[negative_mask]).clamp_min(1e-8)

        positive_margin = float(self.args.radius_positive_margin)
        negative_margin = float(self.args.radius_negative_margin)
        positive_violation = F.relu(positive_scores - (radius_value - positive_margin))
        negative_violation = F.relu((radius_value + negative_margin) - negative_scores)

        if int(self.args.radius_boundary_power) == 2:
            positive_term = positive_violation.square()
            negative_term = negative_violation.square()
        else:
            positive_term = positive_violation
            negative_term = negative_violation

        positive_loss = self.weighted_mean(positive_term, positive_weight)
        negative_loss = self.weighted_mean(negative_term, negative_weight)
        loss_raw = (
            positive_loss * float(self.args.radius_positive_loss_weight)
            + negative_loss * float(self.args.radius_negative_loss_weight)
        )

        return {
            'loss_raw': loss_raw,
            'positive_loss': positive_loss.detach(),
            'negative_loss': negative_loss.detach(),
            'positive_ratio': positive_mask.float().mean(),
            'negative_ratio': negative_mask.float().mean(),
            'positive_score_mean': positive_scores.mean().detach(),
            'negative_score_mean': negative_scores.mean().detach(),
            'positive_cutoff': positive_cutoff.detach(),
            'negative_cutoff': negative_cutoff.detach(),
            'positive_weight_mean': positive_weight.mean().detach(),
            'negative_weight_mean': negative_weight.mean().detach(),
        }

    def compute_transport_barycenter_state(self, out, teacher_out=None):
        eps = 1e-8
        feature_norm = F.normalize(out['features'], p=2, dim=1)
        reference_out = teacher_out if teacher_out is not None else out

        with torch.no_grad():
            if teacher_out is not None and self.use_ema_teacher and self.teacher_prototype_memory is not None:
                prototypes = self.teacher_prototype_memory.get_prototypes().detach()
            else:
                prototypes = self.prototype_memory.get_prototypes().detach()
            known_transport = reference_out['transport_plan'][:-1].transpose(0, 1)
            known_mass = known_transport.sum(dim=1)
            class_posterior = known_transport / known_mass.unsqueeze(1).clamp_min(eps)
            sharpen_t = max(float(self.args.barycenter_sharpen_t), 1e-6)
            sharp_posterior = class_posterior.pow(1.0 / sharpen_t)
            sharp_posterior = sharp_posterior / sharp_posterior.sum(dim=1, keepdim=True).clamp_min(eps)
            barycenter_raw = torch.matmul(sharp_posterior, prototypes)
            barycenter_raw_norm = barycenter_raw.norm(p=2, dim=1)
            barycenter = F.normalize(barycenter_raw, p=2, dim=1, eps=eps)
            posterior_confidence = class_posterior.max(dim=1)[0]
            clean_weight = known_mass.detach() * posterior_confidence.detach()
            open_set_confidence = 1.0 - reference_out['unknown_score'].detach()
            soft_weight = clean_weight * open_set_confidence

            teacher_hard_mask = (
                (posterior_confidence >= float(self.args.teacher_conf_threshold))
                & (open_set_confidence >= float(self.args.teacher_open_threshold))
            )
            residual_weight = soft_weight * (~teacher_hard_mask).float() * float(self.args.barycenter_residual_weight)
            selected_weight = soft_weight * teacher_hard_mask.float()
            ultimate_weight = selected_weight + residual_weight

        barycenter_distance = 0.5 * (feature_norm - barycenter.detach()).pow(2).sum(dim=1)
        loss_raw = (
            ultimate_weight * barycenter_distance
        ).sum() / ultimate_weight.sum().clamp_min(eps)

        return {
            'known_transport': known_transport,
            'known_mass': known_mass,
            'clean_weight': clean_weight,
            'soft_weight': soft_weight,
            'selected_weight': selected_weight,
            'residual_weight': residual_weight,
            'ultimate_weight': ultimate_weight,
            'teacher_hard_mask': teacher_hard_mask,
            'open_set_confidence': open_set_confidence,
            'posterior_confidence': posterior_confidence,
            'class_posterior': class_posterior,
            'sharp_posterior': sharp_posterior,
            'feature_norm': feature_norm,
            'barycenter': barycenter,
            'barycenter_raw_norm': barycenter_raw_norm,
            'barycenter_distance': barycenter_distance,
            'loss_raw': loss_raw,
            'teacher_used': float(teacher_out is not None),
            'reference_unknown_score_mean': reference_out['unknown_score'].mean().detach(),
        }

    def get_target_gate_state(self, out):
        distance_matrix = self.compute_feature_prototype_distance(out['features'])
        candidate_preds = out['class_scores'].argmax(dim=1)
        candidate_distance = distance_matrix.gather(1, candidate_preds.unsqueeze(1)).squeeze(1)
        candidate_radius = self.class_radius[candidate_preds] * self.args.radius_margin
        candidate_radius_initialized = self.class_radius_initialized[candidate_preds]

        alternative_distance_matrix = distance_matrix.clone()
        alternative_distance_matrix.scatter_(1, candidate_preds.unsqueeze(1), float('inf'))
        alternative_distance, alternative_preds = alternative_distance_matrix.min(dim=1)
        margin_gap = alternative_distance - candidate_distance

        global_unknown = out['unknown_score'] >= self.get_unknown_threshold()
        local_unknown = torch.where(
            candidate_radius_initialized,
            candidate_distance > candidate_radius,
            torch.ones_like(global_unknown, dtype=torch.bool),
        )
        final_unknown = global_unknown & local_unknown
        safe_known = (~global_unknown) & (~local_unknown)

        return {
            'distance_matrix': distance_matrix,
            'candidate_preds': candidate_preds,
            'candidate_distance': candidate_distance,
            'candidate_radius': candidate_radius,
            'candidate_radius_initialized': candidate_radius_initialized,
            'alternative_preds': alternative_preds,
            'alternative_distance': alternative_distance,
            'margin_gap': margin_gap,
            'global_unknown': global_unknown,
            'local_unknown': local_unknown,
            'final_unknown': final_unknown,
            'safe_known': safe_known,
        }

    def predict_target(self, out):
        if self.args.open_set_decision == 'transport':
            return predict_open_set_transport(
                out['transport_plan'],
                self.known_num_classes,
            )
        if self.args.open_set_decision == 'radius':
            wgdt_state = self.compute_wgdt_style_target_state(out)
            prediction = wgdt_state['prediction'].clone()
            prediction[wgdt_state['score'] > self.radius.radius.detach()] = self.known_num_classes
            return prediction
        gate_state = self.get_target_gate_state(out)
        prediction = gate_state['candidate_preds'].clone()
        prediction[gate_state['final_unknown']] = self.known_num_classes
        return prediction

    def get_eval_unknown_score(self, out):
        if self.args.open_set_decision == 'radius':
            return self.compute_wgdt_style_target_state(out)['score'].detach()
        return out['unknown_score']

    def build_unknown_score(self, out):
        out['unknown_score'] = fuse_unknown_score(
            out['mass_unknown_score'],
            out['distance_unknown_score'],
            alpha=self.args.unknown_score_alpha,
        )
        return out

    def update_running_threshold(self):
        if len(self.source_dustbin_score_list) == 0:
            return self.get_unknown_threshold()

        scores = torch.cat(self.source_dustbin_score_list).detach().float()
        current_threshold = torch.quantile(scores, self.args.threshold_quantile)

        if not self.threshold_initialized:
            self.running_threshold.copy_(current_threshold)
            self.threshold_initialized.fill_(True)
        else:
            updated = self.args.threshold_ema * self.running_threshold + (1.0 - self.args.threshold_ema) * current_threshold
            self.running_threshold.copy_(updated)

        self.source_dustbin_score_list = []
        return self.get_unknown_threshold()

    def save_checkpoint(self, filename, extra=None):
        save_dir = f'logs/{self.args.log_name}/checkpoints'
        check_path(save_dir)
        save_path = f'{save_dir}/{filename}'
        save_dict = {
            'epoch': self.progress.epoch if hasattr(self, 'progress') else 0,
            'model_state_dict': self.state_dict(),
            'running_threshold': self.get_unknown_threshold(),
            'args': vars(self.args),
        }
        if extra is not None:
            save_dict.update(extra)
        torch.save(save_dict, save_path)

    def save_state_dict_only(self, filename, extra=None):
        save_dir = f'logs/{self.args.log_name}/checkpoints'
        check_path(save_dir)
        save_path = f'{save_dir}/{filename}'
        save_dict = {
            'model_state_dict': self.state_dict(),
        }
        if extra is not None:
            save_dict.update(extra)
        torch.save(save_dict, save_path)

    def get_best_oracle_checkpoint_path(self):
        return f'logs/{self.args.log_name}/checkpoints/best_oracle_hscore.pth'

    def load_best_oracle_checkpoint(self):
        checkpoint_path = self.get_best_oracle_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            print(f'[Best Checkpoint] Not found: {checkpoint_path}. Final evaluation will use the last epoch weights.')
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])

        best_hscore = checkpoint.get('best_oracle_hscore', None)
        best_epoch = checkpoint.get('epoch', None)
        if best_hscore is not None and best_epoch is not None:
            print(f'[Best Checkpoint] Loaded epoch {int(best_epoch) + 1} with tracked eval H-Score {float(best_hscore) * 100:.2f}% for final evaluation.')
        else:
            print('[Best Checkpoint] Loaded best tracked checkpoint for final evaluation.')
        return True

    def evaluate_oracle(self):
        if self.oracle_loader is None:
            return None

        was_training = self.training
        prediction_list = []
        target_list = []
        unknown_score_list = []

        self.eval()
        self.apply_eval_ema_state()
        with torch.no_grad():
            for data in self.oracle_loader:
                x, y = dataToDevice(data, self.device)
                out = self.forward_target(x)
                prediction = self.predict_target(out)
                prediction_list.append(prediction.detach().cpu())
                target_list.append(y.detach().cpu())
                unknown_score_list.append(self.get_eval_unknown_score(out).detach().cpu())
        self.restore_eval_ema_state()

        if was_training:
            self.train()

        result = computeOpensetDomainResult(
            torch.cat(prediction_list),
            torch.cat(target_list),
            self.known_num_classes,
            torch.cat(unknown_score_list),
        )
        return result

    def encode(self, x):
        return self.feature_encoder(x)['features']

    def forward_source(self, x, y=None, epoch=None, update_prototypes=False, respect_proto_stop=True):
        features = self.encode(x)
        prototype_update_active = update_prototypes and (self.is_prototype_update_active(epoch) if respect_proto_stop else True)
        if y is not None and prototype_update_active:
            self.prototype_memory.update(features, y, epoch=epoch)

        logits = self.source_classifier(features)
        prediction = logits.argmax(dim=1)
        prototype_logits = self.prototype_memory.compute_logits(features)
        anchor_out = self.anchor(logits, y) if y is not None else self.anchor(logits)
        out = {
            'features': features,
            'logits': logits,
            'prediction': prediction,
            'prototype_logits': prototype_logits,
            **anchor_out,
        }

        if y is not None:
            out['loss_cls'] = F.cross_entropy(logits, y)
            out['loss_proto'] = self.prototype_memory.compactness_loss(features, y)
            if prototype_update_active:
                with torch.no_grad():
                    positive_distance = self.compute_feature_prototype_distance(features).gather(1, y.view(-1, 1)).view(-1)
                    for class_index in range(self.known_num_classes):
                        class_mask = y == class_index
                        if class_mask.sum() == 0:
                            continue
                        class_radius = torch.quantile(positive_distance[class_mask].float(), self.args.class_radius_quantile)
                        if not self.class_radius_initialized[class_index]:
                            self.class_radius[class_index] = class_radius
                            self.class_radius_initialized[class_index] = True
                        else:
                            updated_radius = (
                                self.args.class_radius_ema * self.class_radius[class_index]
                                + (1.0 - self.args.class_radius_ema) * class_radius
                            )
                            self.class_radius[class_index] = updated_radius
            out['prototype_update_active'] = float(prototype_update_active)
        return out
    def forward_target(self, x):
        features = self.encode(x)
        classifier_logits = self.source_classifier(features)
        classifier_scores = torch.softmax(classifier_logits, dim=1)
        anchor_out = self.anchor(classifier_logits)
        prototypes = self.prototype_memory.get_prototypes().detach()
        uot_out = self.uot_solver(prototypes, features)
        out = {
            'features': features,
            'classifier_logits': classifier_logits,
            'classifier_scores': classifier_scores,
            **anchor_out,
            **uot_out,
        }
        return self.build_unknown_score(out)

    @torch.no_grad()
    def forward_target_teacher(self, x):
        if not self.use_ema_teacher:
            return None
        if not self.teacher_initialized:
            self.sync_teacher_from_student()
        self.teacher_feature_encoder.eval()
        features = self.teacher_feature_encoder(x)['features']
        classifier_logits = self.source_classifier(features)
        classifier_scores = torch.softmax(classifier_logits, dim=1)
        anchor_out = self.anchor(classifier_logits)
        prototypes = self.teacher_prototype_memory.get_prototypes().detach()
        uot_out = self.uot_solver(prototypes, features)
        out = {
            'features': features,
            'classifier_logits': classifier_logits,
            'classifier_scores': classifier_scores,
            **anchor_out,
            **uot_out,
        }
        return self.build_unknown_score(out)

    def forward_uot_by_features(self, features):
        prototypes = self.prototype_memory.get_prototypes().detach()
        out = self.uot_solver(prototypes, features)
        return self.build_unknown_score(out)

    def pre_train_step(self, batch):
        x, y = batch
        epoch = self.progress.epoch if hasattr(self, 'progress') else 0
        out = self.forward_source(x, y, epoch=epoch, update_prototypes=True, respect_proto_stop=False)
        loss = (
            out['loss_cls']
            + self.args.prototype_loss_weight * out['loss_proto']
            + self.args.anchor_aux_loss_weight * out['loss_anchor'] * self.args.alpha
            + self.args.tuplet_aux_loss_weight * out['loss_tuplet']
        )
        self.source_oa.update(out['prediction'], y)
        return {
            'loss': loss,
            'information': {
                'loss_cls': out['loss_cls'],
                'loss_proto': out['loss_proto'],
                'loss_anchor': out['loss_anchor'] * self.args.alpha,
                'loss_tuplet': out['loss_tuplet'],
            }
        }

    def pre_train_epoch_end(self):
        dic = {
            'source_oa': self.source_oa.compute()
        }
        self.source_oa.reset()
        return dic

    def pre_train_optimizer(self):
        return torch.optim.SGD(
            [
                {'params': self.feature_encoder.parameters()},
                {'params': self.source_classifier.parameters()},
            ],
            lr=0.001,
            momentum=0.9,
            weight_decay=5e-4,
        )
    def train_step(self, batch):
        [source_x, source_y], [target_x, target_y] = batch
        epoch = self.progress.epoch if hasattr(self, 'progress') else 0

        source_out = self.forward_source(source_x, source_y, epoch=epoch, update_prototypes=True)
        if self.use_ema_teacher:
            self.update_ema_teacher()
        self.uot_solver.update_source_calibration(
            self.prototype_memory.get_prototypes().detach(),
            source_out['features'].detach(),
            source_y,
        )
        source_uot_out = self.forward_uot_by_features(source_out['features'].detach())
        target_out = self.forward_target(target_x)
        target_teacher_out = self.forward_target_teacher(target_x) if self.use_ema_teacher else None

        self.source_dustbin_score_list.append(source_uot_out['unknown_score'].detach().cpu())
        self.target_dustbin_ratio_list.append(target_out['target_dustbin_ratio'].detach().cpu())

        uot_active = epoch >= self.args.uot_warmup_epochs
        loss_uot = target_out['loss'] * self.args.uot_loss_weight if uot_active else target_out['loss'] * 0.0

        dann_stop_reached = self.args.dann_stop_epochs > 0 and epoch >= self.args.dann_stop_epochs
        dann_active = epoch >= self.args.dann_warmup_epochs and not dann_stop_reached
        soft_dann_state = self.compute_soft_dann_target_weights(target_out)
        wgdt_target_state = self.compute_wgdt_style_target_state(target_out)
        target_weight = wgdt_target_state['weight'] if self.args.open_set_decision == 'radius' else soft_dann_state['w_target']
        loss_disc = self.domain_adv(
            source_out['features'],
            target_out['features'],
            w_t=target_weight.unsqueeze(1),
        ) * self.args.domain_loss_weight if dann_active else target_out['loss'] * 0.0

        barycenter_state = self.compute_transport_barycenter_state(target_out, teacher_out=target_teacher_out)
        barycenter_active = epoch >= self.args.barycenter_warmup_epochs
        loss_bary_raw = barycenter_state['loss_raw']
        loss_bary = loss_bary_raw * self.args.barycenter_loss_weight if barycenter_active else target_out['loss'] * 0.0
        radius_active = epoch >= self.args.radius_warmup_epochs
        radius_score = wgdt_target_state['score'] if self.args.open_set_decision == 'radius' else target_out['unknown_score']
        radius_weight = wgdt_target_state['weight'] if self.args.open_set_decision == 'radius' else barycenter_state['ultimate_weight']
        radius_boundary_state = None
        if self.args.open_set_decision == 'radius' and self.args.radius_loss_form == 'dual_boundary':
            radius_boundary_state = self.compute_dual_boundary_radius_state(wgdt_target_state)
            loss_radius_raw = radius_boundary_state['loss_raw']
        else:
            loss_radius_raw = self.radius(
                radius_score.detach(),
                weight=radius_weight.detach(),
            )
        loss_radius = loss_radius_raw * self.args.radius_loss_weight if radius_active else target_out['loss'] * 0.0

        source_decay_factor = self.get_source_decay_factor(epoch)

        gate_state = self.get_target_gate_state(target_out)
        safe_known = gate_state['safe_known']
        safe_known_count = int(safe_known.sum().item())
        safe_known_ratio = safe_known.float().mean()
        tgt_proto_active = epoch >= self.args.tgt_proto_warmup_epochs
        if safe_known_count > 0:
            loss_tgt_proto_raw = gate_state['candidate_distance'][safe_known].mean()
            loss_tgt_margin_raw = torch.relu(self.args.tgt_margin_value - gate_state['margin_gap'][safe_known]).mean()
            safe_margin_gap_mean = gate_state['margin_gap'][safe_known].mean()
            safe_alt_distance_mean = gate_state['alternative_distance'][safe_known].mean()
        else:
            loss_tgt_proto_raw = target_out['loss'] * 0.0
            loss_tgt_margin_raw = target_out['loss'] * 0.0
            safe_margin_gap_mean = target_out['loss'] * 0.0
            safe_alt_distance_mean = target_out['loss'] * 0.0
        loss_tgt_proto = loss_tgt_proto_raw * self.args.tgt_proto_loss_weight if tgt_proto_active else target_out['loss'] * 0.0
        loss_tgt_margin = loss_tgt_margin_raw * self.args.tgt_margin_loss_weight if tgt_proto_active else target_out['loss'] * 0.0

        loss_dic = {
            'loss_cls': source_out['loss_cls'] * self.args.adapt_cls_loss_weight * source_decay_factor,
            'loss_proto': source_out['loss_proto'] * self.args.prototype_loss_weight * self.args.adapt_proto_loss_weight * source_decay_factor,
            'loss_anchor': source_out['loss_anchor'] * self.args.alpha * self.args.anchor_aux_loss_weight * self.args.adapt_anchor_loss_weight * source_decay_factor,
            'loss_tuplet': source_out['loss_tuplet'] * self.args.tuplet_aux_loss_weight * self.args.adapt_tuplet_loss_weight * source_decay_factor,
            'loss_uot': loss_uot,
            'loss_disc': loss_disc,
            'loss_bary': loss_bary,
            'loss_radius': loss_radius,
            'loss_tgt_proto': loss_tgt_proto,
            'loss_tgt_margin': loss_tgt_margin,
        }

        self.source_oa.update(source_out['prediction'], source_y)

        information = {
            **loss_dic,
            'source_oa_running': self.source_oa.compute(),
            'dustbin_mean': target_out['dustbin_scores'].mean(),
            'unknown_score_mean': target_out['unknown_score'].mean(),
            'dustbin_cost': target_out['dustbin_cost'],
            'dustbin_mass_prior': target_out['dustbin_mass_prior'],
            'target_dustbin_ratio': target_out['target_dustbin_ratio'],
            'source_threshold_batch': torch.quantile(source_uot_out['unknown_score'].detach(), self.args.threshold_quantile),
            'uot_active': float(uot_active),
            'loss_disc_raw': self.domain_adv.domain_discriminator_accuracy if dann_active else 0.0,
            'dann_active': float(dann_active),
            'target_known_weight_mean': target_weight.mean(),
            'radius_score_mode': {
                'prototype_distance': 0.0,
                'anchor_gamma': 1.0,
                'prototype_distance_classwise': 2.0,
            }.get(wgdt_target_state['mode'], -1.0),
            'wgdt_score_mean': wgdt_target_state['score_mean'],
            'wgdt_score_std': wgdt_target_state['score_std'],
            'wgdt_distance_mean': wgdt_target_state['distance_mean'],
            'wgdt_distance_std': wgdt_target_state['distance_std'],
            'wgdt_class_scale_mean': wgdt_target_state['class_scale_mean'],
            'wgdt_class_scale_std': wgdt_target_state['class_scale_std'],
            'wgdt_class_scale_initialized_ratio': wgdt_target_state['class_scale_initialized_ratio'],
            'wgdt_weight_mean': wgdt_target_state['weight_mean'],
            'wgdt_weight_std': wgdt_target_state['weight_std'],
            'w_open_mean': soft_dann_state['w_open'].mean(),
            'w_close_mean': soft_dann_state['w_close'].mean(),
            'w_margin_mean': soft_dann_state['w_margin'].mean(),
            'w_target_mean': soft_dann_state['w_target'].mean(),
            'w_target_std': soft_dann_state['w_target'].std(),
            'ema_teacher_active': float(target_teacher_out is not None),
            'teacher_unknown_score_mean': target_teacher_out['unknown_score'].mean() if target_teacher_out is not None else target_out['unknown_score'].new_zeros(()),
            'teacher_student_unknown_gap': (target_teacher_out['unknown_score'] - target_out['unknown_score'].detach()).abs().mean() if target_teacher_out is not None else target_out['unknown_score'].new_zeros(()),
            'barycenter_active': float(barycenter_active),
            'loss_bary_raw': loss_bary_raw.detach(),
            'radius_active': float(radius_active),
            'loss_radius_raw': loss_radius_raw.detach(),
            'learnable_radius': self.radius.radius.detach(),
            'radius_loss_form': 1.0 if self.args.radius_loss_form == 'dual_boundary' else 0.0,
            'radius_positive_ratio': radius_boundary_state['positive_ratio'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_negative_ratio': radius_boundary_state['negative_ratio'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_positive_loss_raw': radius_boundary_state['positive_loss'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_negative_loss_raw': radius_boundary_state['negative_loss'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_positive_score_mean': radius_boundary_state['positive_score_mean'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_negative_score_mean': radius_boundary_state['negative_score_mean'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_positive_weight_mean': radius_boundary_state['positive_weight_mean'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_negative_weight_mean': radius_boundary_state['negative_weight_mean'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_positive_cutoff': radius_boundary_state['positive_cutoff'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'radius_negative_cutoff': radius_boundary_state['negative_cutoff'] if radius_boundary_state is not None else target_out['loss'].new_zeros(()),
            'known_mass_mean': barycenter_state['known_mass'].mean(),
            'known_mass_std': barycenter_state['known_mass'].std(),
            'clean_weight_mean': barycenter_state['clean_weight'].mean(),
            'soft_weight_mean': barycenter_state['soft_weight'].mean(),
            'selected_weight_mean': barycenter_state['selected_weight'].mean(),
            'residual_weight_mean': barycenter_state['residual_weight'].mean(),
            'teacher_hard_mask_ratio': barycenter_state['teacher_hard_mask'].float().mean(),
            'open_set_conf_mean': barycenter_state['open_set_confidence'].mean(),
            'ultimate_weight_mean': barycenter_state['ultimate_weight'].mean(),
            'posterior_confidence_mean': barycenter_state['posterior_confidence'].mean(),
            'posterior_confidence_std': barycenter_state['posterior_confidence'].std(),
            'sharp_posterior_max_mean': barycenter_state['sharp_posterior'].max(dim=1)[0].mean(),
            'barycenter_raw_norm_mean': barycenter_state['barycenter_raw_norm'].mean(),
            'barycenter_dist_mean': barycenter_state['barycenter_distance'].mean().detach(),
            'teacher_used_for_barycenter': barycenter_state['teacher_used'],
            'reference_unknown_score_mean': barycenter_state['reference_unknown_score_mean'],
            'adapt_cls_loss_weight': float(self.args.adapt_cls_loss_weight),
            'adapt_proto_loss_weight': float(self.args.adapt_proto_loss_weight),
            'adapt_anchor_loss_weight': float(self.args.adapt_anchor_loss_weight),
            'adapt_tuplet_loss_weight': float(self.args.adapt_tuplet_loss_weight),
            'tgt_proto_active': float(tgt_proto_active),
            'source_decay_factor': float(source_decay_factor),
            'prototype_update_active': source_out.get('prototype_update_active', 0.0),
            'loss_tgt_proto_raw': loss_tgt_proto_raw.detach(),
            'loss_tgt_margin_raw': loss_tgt_margin_raw.detach(),
            'safe_known_ratio': safe_known_ratio,
            'safe_known_count': float(safe_known_count),
            'safe_margin_gap_mean': safe_margin_gap_mean.detach(),
            'safe_alt_distance_mean': safe_alt_distance_mean.detach(),
        }
        return {
            'loss': sum(value for value in loss_dic.values()),
            'information': information,
        }

    def train_epoch_end(self):
        if self.args.uot_warmup_epochs > 0 and not self.uot_warmup_notice_printed:
            print(f'[UOT Warmup] `loss_uot=0.0` for the first {self.args.uot_warmup_epochs} epochs.')
            self.uot_warmup_notice_printed = True
        if self.args.uot_warmup_epochs > 0 and (self.progress.epoch + 1) == self.args.uot_warmup_epochs:
            print('[UOT Warmup] Warmup finished. UOT alignment will start next epoch.')
        if self.args.dann_warmup_epochs > 0 and not self.dann_warmup_notice_printed:
            print(f'[DANN Warmup] `loss_disc=0.0` for the first {self.args.dann_warmup_epochs} epochs.')
            self.dann_warmup_notice_printed = True
        if self.args.dann_warmup_epochs > 0 and (self.progress.epoch + 1) == self.args.dann_warmup_epochs:
            if self.args.dann_stop_epochs > 0 and self.args.dann_stop_epochs <= self.args.dann_warmup_epochs:
                print('[DANN Warmup] Warmup finished, but the DANN active window is empty. `loss_disc` will remain 0.0.')
            else:
                print('[DANN Warmup] Warmup finished. Confidence-weighted DANN will start next epoch.')
        if self.args.dann_stop_epochs > 0 and not self.dann_stop_notice_printed and (self.progress.epoch + 1) == self.args.dann_stop_epochs:
            print('[DANN Stop] Confidence-weighted DANN window finished. `loss_disc` will stay 0.0 from next epoch.')
            self.dann_stop_notice_printed = True

        threshold = self.update_running_threshold()
        target_dustbin_ratio = None
        if len(self.target_dustbin_ratio_list) > 0:
            target_dustbin_ratio = torch.stack(self.target_dustbin_ratio_list).mean()
            self.target_dustbin_ratio_list = []

        dic = {
            'source_oa': self.source_oa.compute(),
            'running_threshold': threshold,
        }
        if self.use_eval_ema and self.eval_ema_initialized and 'radius.radius' in self.eval_ema_state:
            dic['eval_ema_radius'] = float(self.eval_ema_state['radius.radius'].item())
        if target_dustbin_ratio is not None:
            dic['target_dustbin_ratio'] = target_dustbin_ratio

        summary_parts = [f"source_oa={float(dic['source_oa']):.4f}"]
        if 'eval_ema_radius' in dic:
            summary_parts.append(f"eval_ema_radius={float(dic['eval_ema_radius']):.4f}")
        summary_parts.append(f"running_threshold={float(dic['running_threshold']):.4f}")
        print('[Train Summary] ' + ', '.join(summary_parts))

        oracle_result = self.evaluate_oracle()
        if oracle_result is not None:
            oracle_hscore = float(oracle_result['hos'])
            dic['oracle_hos'] = oracle_hscore
            dic['oracle_unknown'] = oracle_result['unknown']
            dic['oracle_aa_known'] = oracle_result['aa_known']
            print(f"[Tracked Eval] Target H-Score: {oracle_hscore * 100:.2f}%")

            if self.args.save_best_oracle_checkpoint == 'True' and oracle_hscore > self.best_oracle_hscore:
                self.best_oracle_hscore = oracle_hscore
                self.save_state_dict_only(
                    'best_oracle_hscore.pth',
                    {
                        'best_oracle_hscore': oracle_hscore,
                        'epoch': self.progress.epoch if hasattr(self, 'progress') else 0,
                    }
                )

        self.source_oa.reset()
        if self.args.save_last_checkpoint == 'True':
            self.save_checkpoint('last_checkpoint.pth', {'target_dustbin_ratio': float(target_dustbin_ratio) if target_dustbin_ratio is not None else None})
        return dic

    def train_step_end(self):
        if self.use_eval_ema and (self.progress.epoch if hasattr(self, 'progress') else 0) >= self.args.eval_ema_start_epoch:
            self.update_eval_ema_state()

    def test_start(self):
        self.apply_eval_ema_state()

    def test_finish(self):
        self.restore_eval_ema_state()

    def prediction_start(self):
        self.apply_eval_ema_state()

    def prediction_finish(self):
        self.restore_eval_ema_state()

    def train_optimizer(self):
        train_lr_encoder = self.args.train_lr_encoder if self.args.train_lr_encoder > 0 else self.args.lr_encoder
        default_domain_lr = self.args.lr_domain if hasattr(self.args, 'lr_domain') else self.args.lr_encoder
        train_lr_domain = self.args.train_lr_domain if self.args.train_lr_domain > 0 else default_domain_lr
        train_lr_radius = self.args.train_lr_radius if self.args.train_lr_radius > 0 else 1e-4
        return torch.optim.SGD(
            [
                {'params': self.feature_encoder.parameters(), 'lr': train_lr_encoder},
                {'params': self.source_classifier.parameters(), 'lr': train_lr_encoder},
                {'params': self.disc_encoder.parameters(), 'lr': train_lr_domain},
                {'params': self.radius.parameters(), 'lr': train_lr_radius},
            ],
            momentum=0.9,
            weight_decay=5e-4,
        )
    def test_step(self, batch):
        x, y = batch
        out = self.forward_target(x)
        prediction = self.predict_target(out)
        self.metric.update(prediction, y, self.get_eval_unknown_score(out))

    def test_end(self):
        self.metric.finish()

    def prediction_step(self, batch):
        x = batch
        out = self.forward_target(x)
        prediction = self.predict_target(out)
        self.prediciton_all.append(prediction)

    def prediction_end(self):
        from utils.draw import drawPredictionMap

        drawPredictionMap(
            self.prediciton_all,
            f'{self.args.log_name} {self.args.target_dataset}',
            self.target_info,
            known_classes=self.args.target_known_classes,
            unknown_classes=self.args.target_unknown_classes,
            draw_background=False,
        )


def run_model(model: Model, data_loader: dict):
    trainer = Trainer(model, model.device)
    model.oracle_loader = data_loader['target']['test']

    if model.args.pre_train == 'True':
        trainer.train('pre_train', data_loader['source']['train'], model.args.pre_train_epochs)
    trainer.train('train', CombinedLoader([data_loader['source']['train'], data_loader['target']['train']]), model.args.epochs)

    if model.args.save_best_oracle_checkpoint == 'True':
        model.load_best_oracle_checkpoint()
    else:
        print('[Best Checkpoint] Disabled. Final evaluation will use the last epoch weights.')

    trainer.test('test', data_loader['target']['test'])

    if model.args.draw == 'True':
        trainer.test('prediction', data_loader['target']['all'])


def get_model(args, source_info, target_info):
    from utils.utils import getDevice

    model_args = {
        'args': args,
        'source_info': source_info,
        'target_info': target_info,
        'device': getDevice(args.device),
        'in_channels': args.pca if hasattr(args, 'pca') and args.pca > 0 else source_info.bands_num,
        'patch': args.patch,
        'known_num_classes': len(args.source_known_classes),
    }
    return Model(**model_args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['WGDT', 'UOT_OSDA'], default='UOT_OSDA')
    parser.add_argument('--log_name', type=str, default='UOT_OSDA')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_num', type=int, default=180)
    parser.add_argument('--few_train_num', type=int, default=150)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--patch', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--source_dataset', choices=['Houston13_7gt', 'PaviaU_7gt', 'HyRank_source', 'Yancheng_ZY'], default='Houston13_7gt')
    parser.add_argument('--target_dataset', choices=['Houston18_OS', 'PaviaC_OS', 'HyRank_target', 'Yancheng_GF'], default='Houston18_OS')

    parser.add_argument('--pre_train', type=str, default='True')
    parser.add_argument('--pre_train_epochs', type=int, default=20)
    parser.add_argument('--draw', type=str, default='False')

    parser.add_argument('--prototype_momentum', type=float, default=0.99)
    parser.add_argument('--prototype_temperature', type=float, default=1.0)
    parser.add_argument('--prototype_warmup_epochs', type=int, default=0)
    parser.add_argument('--prototype_loss_weight', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--anchor_weight', type=float, default=10.0)
    parser.add_argument('--anchor_aux_loss_weight', type=float, default=1.0)
    parser.add_argument('--tuplet_aux_loss_weight', type=float, default=1.0)

    parser.add_argument('--uot_epsilon', type=float, default=0.05)
    parser.add_argument('--uot_tau_source', type=float, default=0.95)
    parser.add_argument('--uot_tau_target', type=float, default=0.95)
    parser.add_argument('--uot_max_iter', type=int, default=30)
    parser.add_argument('--uot_metric', type=str, choices=['euclidean', 'cosine'], default='euclidean')
    parser.add_argument('--uot_loss_weight', type=float, default=1.0)
    parser.add_argument('--uot_no_grad', type=str, default='True')
    parser.add_argument('--uot_warmup_epochs', type=int, default=0)
    parser.add_argument('--domain_loss_weight', type=float, default=0.1)
    parser.add_argument('--dann_hidden_size', type=int, default=64)
    parser.add_argument('--dann_warmup_epochs', type=int, default=5)
    parser.add_argument('--dann_stop_epochs', type=int, default=-1)
    parser.add_argument('--dann_weight_low', type=float, default=0.05)
    parser.add_argument('--dann_weight_high', type=float, default=0.95)
    parser.add_argument('--class_radius_quantile', type=float, default=0.9)
    parser.add_argument('--class_radius_ema', type=float, default=0.9)
    parser.add_argument('--radius_margin', type=float, default=1.2)
    parser.add_argument('--tgt_proto_warmup_epochs', type=int, default=10)
    parser.add_argument('--tgt_proto_loss_weight', type=float, default=0.02)
    parser.add_argument('--tgt_margin_loss_weight', type=float, default=0.05)
    parser.add_argument('--tgt_margin_value', type=float, default=0.1)
    parser.add_argument('--barycenter_warmup_epochs', type=int, default=5)
    parser.add_argument('--barycenter_loss_weight', type=float, default=0.0)
    parser.add_argument('--radius_warmup_epochs', type=int, default=5)
    parser.add_argument('--radius_loss_weight', type=float, default=0.0)
    parser.add_argument('--learnable_radius_init', type=float, default=0.0)
    parser.add_argument('--learnable_radius_margin', type=float, default=0.1)
    parser.add_argument('--radius_score_mode', type=str, choices=['prototype_distance', 'prototype_distance_classwise', 'anchor_gamma'], default='prototype_distance')
    parser.add_argument('--radius_loss_form', type=str, choices=['margin_mse', 'dual_boundary'], default='dual_boundary')
    parser.add_argument('--radius_positive_quantile', type=float, default=0.7)
    parser.add_argument('--radius_negative_quantile', type=float, default=0.3)
    parser.add_argument('--radius_positive_margin', type=float, default=0.0)
    parser.add_argument('--radius_negative_margin', type=float, default=0.0)
    parser.add_argument('--radius_positive_loss_weight', type=float, default=1.0)
    parser.add_argument('--radius_negative_loss_weight', type=float, default=1.0)
    parser.add_argument('--radius_boundary_power', type=int, choices=[1, 2], default=2)
    parser.add_argument('--barycenter_sharpen_t', type=float, default=0.5)
    parser.add_argument('--adapt_cls_loss_weight', type=float, default=1.0)
    parser.add_argument('--adapt_proto_loss_weight', type=float, default=1.0)
    parser.add_argument('--adapt_anchor_loss_weight', type=float, default=1.0)
    parser.add_argument('--adapt_tuplet_loss_weight', type=float, default=1.0)
    parser.add_argument('--train_lr_encoder', type=float, default=-1.0)
    parser.add_argument('--train_lr_domain', type=float, default=-1.0)
    parser.add_argument('--train_lr_radius', type=float, default=-1.0)
    parser.add_argument('--use_ema_teacher', type=str, default='False')
    parser.add_argument('--teacher_momentum', type=float, default=0.999)
    parser.add_argument('--use_eval_ema', type=str, default='False')
    parser.add_argument('--eval_ema_decay', type=float, default=0.999)
    parser.add_argument('--eval_ema_start_epoch', type=int, default=0)
    parser.add_argument('--teacher_conf_threshold', type=float, default=0.7)
    parser.add_argument('--teacher_open_threshold', type=float, default=0.6)
    parser.add_argument('--barycenter_residual_weight', type=float, default=0.0)
    parser.add_argument('--tau_close', type=float, default=0.05)
    parser.add_argument('--tau_margin', type=float, default=0.05)
    parser.add_argument('--proto_update_stop_epoch', type=int, default=10)
    parser.add_argument('--source_decay_epoch', type=int, default=10)
    parser.add_argument('--source_decay_factor', type=float, default=0.05)

    parser.add_argument('--dustbin_quantile', type=float, default=0.8)
    parser.add_argument('--dustbin_beta', type=float, default=0.95)
    parser.add_argument('--dustbin_mass_prior', type=float, default=0.05)
    parser.add_argument('--dustbin_cost_mode', type=str, choices=['source_quantile', 'absolute', 'target_quantile'], default='source_quantile')
    parser.add_argument('--dustbin_cost_value', type=float, default=0.6)
    parser.add_argument('--dustbin_source_quantile', type=float, default=0.95)
    parser.add_argument('--dustbin_source_margin', type=float, default=0.1)
    parser.add_argument('--unknown_threshold', type=float, default=0.5)
    parser.add_argument('--threshold_quantile', type=float, default=0.95)
    parser.add_argument('--threshold_ema', type=float, default=0.9)
    parser.add_argument('--save_last_checkpoint', type=str, default='True')
    parser.add_argument('--save_best_oracle_checkpoint', type=str, default='True')
    parser.add_argument('--unknown_score_alpha', type=float, default=0.5)
    parser.add_argument('--open_set_decision', type=str, choices=['threshold', 'transport', 'radius'], default='threshold')

    args = parser.parse_args()
    mergeArgs(args, args.target_dataset, getCliOverrideKeys())
    return args



