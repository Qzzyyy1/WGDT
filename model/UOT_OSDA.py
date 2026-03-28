import os
import os
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
        gate_state = self.get_target_gate_state(out)
        prediction = gate_state['candidate_preds'].clone()
        prediction[gate_state['final_unknown']] = self.known_num_classes
        return prediction

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
        with torch.no_grad():
            for data in self.oracle_loader:
                x, y = dataToDevice(data, self.device)
                out = self.forward_target(x)
                prediction = self.predict_target(out)
                prediction_list.append(prediction.detach().cpu())
                target_list.append(y.detach().cpu())
                unknown_score_list.append(out['unknown_score'].detach().cpu())

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

    def forward_source(self, x, y=None, epoch=None, update_prototypes=False):
        features = self.encode(x)
        prototype_update_active = update_prototypes and self.is_prototype_update_active(epoch)
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
        prototypes = self.prototype_memory.get_prototypes().detach()
        uot_out = self.uot_solver(prototypes, features)
        out = {
            'features': features,
            'classifier_logits': classifier_logits,
            'classifier_scores': classifier_scores,
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
        out = self.forward_source(x, y, epoch=epoch, update_prototypes=True)
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
        self.uot_solver.update_source_calibration(
            self.prototype_memory.get_prototypes().detach(),
            source_out['features'].detach(),
            source_y,
        )
        source_uot_out = self.forward_uot_by_features(source_out['features'].detach())
        target_out = self.forward_target(target_x)

        self.source_dustbin_score_list.append(source_uot_out['unknown_score'].detach().cpu())
        self.target_dustbin_ratio_list.append(target_out['target_dustbin_ratio'].detach().cpu())

        uot_active = epoch >= self.args.uot_warmup_epochs
        loss_uot = target_out['loss'] * self.args.uot_loss_weight if uot_active else target_out['loss'] * 0.0

        dann_stop_reached = self.args.dann_stop_epochs > 0 and epoch >= self.args.dann_stop_epochs
        dann_active = epoch >= self.args.dann_warmup_epochs and not dann_stop_reached
        target_weight = torch.clamp(1.0 - target_out['unknown_score'].detach(), min=self.args.dann_weight_low, max=self.args.dann_weight_high)
        loss_disc = self.domain_adv(
            source_out['features'],
            target_out['features'],
            w_t=target_weight.unsqueeze(1),
        ) * self.args.domain_loss_weight if dann_active else target_out['loss'] * 0.0

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
            'loss_cls': source_out['loss_cls'] * source_decay_factor,
            'loss_proto': source_out['loss_proto'] * self.args.prototype_loss_weight * source_decay_factor,
            'loss_anchor': source_out['loss_anchor'] * self.args.alpha * self.args.anchor_aux_loss_weight * source_decay_factor,
            'loss_tuplet': source_out['loss_tuplet'] * self.args.tuplet_aux_loss_weight * source_decay_factor,
            'loss_uot': loss_uot,
            'loss_disc': loss_disc,
            'loss_tgt_proto': loss_tgt_proto,
            'loss_tgt_margin': loss_tgt_margin,
        }

        self.source_oa.update(source_out['prediction'], source_y)

        information = {
            **loss_dic,
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
        if target_dustbin_ratio is not None:
            dic['target_dustbin_ratio'] = target_dustbin_ratio

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

    def train_optimizer(self):
        return torch.optim.SGD(
            [
                {'params': self.feature_encoder.parameters(), 'lr': self.args.lr_encoder},
                {'params': self.source_classifier.parameters(), 'lr': self.args.lr_encoder},
                {'params': self.disc_encoder.parameters(), 'lr': self.args.lr_domain if hasattr(self.args, 'lr_domain') else self.args.lr_encoder},
            ],
            momentum=0.9,
            weight_decay=5e-4,
        )
    def test_step(self, batch):
        x, y = batch
        out = self.forward_target(x)
        prediction = self.predict_target(out)
        self.metric.update(prediction, y, out['unknown_score'])

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
    parser.add_argument('--prototype_warmup_epochs', type=int, default=5)
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
    parser.add_argument('--open_set_decision', type=str, choices=['threshold', 'transport'], default='threshold')

    args = parser.parse_args()
    mergeArgs(args, args.target_dataset, getCliOverrideKeys())
    return args

