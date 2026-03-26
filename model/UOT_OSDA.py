import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from utils.meter import OpensetDomainMetric
from utils.Trainer import Trainer
from utils.dataLoader import CombinedLoader
from utils.open_set import predict_open_set
from utils.utils import mergeArgs
from .DCRN import DCRN
from .Prototype import PrototypeMemory
from .UOT import UOTSolver


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
            use_no_grad=args.uot_no_grad == 'True',
        )

        self.source_oa = Accuracy()
        self.metric = OpensetDomainMetric(self.known_num_classes, self.args)
        self.prediciton_all = []

    def encode(self, x):
        return self.feature_encoder(x)['features']

    def forward_source(self, x, y=None, epoch=None, update_prototypes=False):
        features = self.encode(x)
        if y is not None and update_prototypes:
            self.prototype_memory.update(features, y, epoch=epoch)

        logits = self.prototype_memory.compute_logits(features)
        prediction = logits.argmax(dim=1)
        out = {
            'features': features,
            'logits': logits,
            'prediction': prediction,
        }

        if y is not None:
            out['loss_cls'] = F.cross_entropy(logits, y)
            out['loss_proto'] = self.prototype_memory.compactness_loss(features, y)
        return out

    def forward_target(self, x):
        features = self.encode(x)
        prototypes = self.prototype_memory.get_prototypes().detach()
        uot_out = self.uot_solver(prototypes, features)
        return {
            'features': features,
            **uot_out,
        }

    def pre_train_step(self, batch):
        x, y = batch
        epoch = self.progress.epoch if hasattr(self, 'progress') else 0
        out = self.forward_source(x, y, epoch=epoch, update_prototypes=True)
        loss = out['loss_cls'] + self.args.prototype_loss_weight * out['loss_proto']
        self.source_oa.update(out['prediction'], y)
        return {
            'loss': loss,
            'information': {
                'loss_cls': out['loss_cls'],
                'loss_proto': out['loss_proto'],
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
            self.feature_encoder.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=5e-4,
        )

    def train_step(self, batch):
        [source_x, source_y], [target_x, target_y] = batch
        epoch = self.progress.epoch if hasattr(self, 'progress') else 0

        source_out = self.forward_source(source_x, source_y, epoch=epoch, update_prototypes=True)
        target_out = self.forward_target(target_x)

        loss_dic = {
            'loss_cls': source_out['loss_cls'],
            'loss_proto': source_out['loss_proto'] * self.args.prototype_loss_weight,
            'loss_uot': target_out['loss'] * self.args.uot_loss_weight,
        }

        self.source_oa.update(source_out['prediction'], source_y)

        information = {
            **loss_dic,
            'dustbin_mean': target_out['dustbin_scores'].mean(),
            'dustbin_cost': target_out['dustbin_cost'],
        }
        return {
            'loss': sum(value for value in loss_dic.values()),
            'information': information,
        }

    def train_epoch_end(self):
        dic = {
            'source_oa': self.source_oa.compute()
        }
        self.source_oa.reset()
        return dic

    def train_optimizer(self):
        return torch.optim.SGD(
            self.feature_encoder.parameters(),
            lr=self.args.lr_encoder,
            momentum=0.9,
            weight_decay=5e-4,
        )

    def test_step(self, batch):
        x, y = batch
        out = self.forward_target(x)
        prediction = predict_open_set(
            out['class_scores'],
            out['dustbin_scores'],
            self.args.unknown_threshold,
            self.known_num_classes,
        )
        self.metric.update(prediction, y, out['dustbin_scores'])

    def test_end(self):
        self.metric.finish()

    def prediction_step(self, batch):
        x = batch
        out = self.forward_target(x)
        prediction = predict_open_set(
            out['class_scores'],
            out['dustbin_scores'],
            self.args.unknown_threshold,
            self.known_num_classes,
        )
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

    if model.args.pre_train == 'True':
        trainer.train('pre_train', data_loader['source']['train'], model.args.pre_train_epochs)
    trainer.train('train', CombinedLoader([data_loader['source']['train'], data_loader['target']['train']]), model.args.epochs)
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

    parser.add_argument('--uot_epsilon', type=float, default=0.05)
    parser.add_argument('--uot_tau_source', type=float, default=1.0)
    parser.add_argument('--uot_tau_target', type=float, default=1.0)
    parser.add_argument('--uot_max_iter', type=int, default=30)
    parser.add_argument('--uot_metric', type=str, choices=['euclidean', 'cosine'], default='euclidean')
    parser.add_argument('--uot_loss_weight', type=float, default=1.0)
    parser.add_argument('--uot_no_grad', type=str, default='True')

    parser.add_argument('--dustbin_quantile', type=float, default=0.8)
    parser.add_argument('--dustbin_beta', type=float, default=0.95)
    parser.add_argument('--unknown_threshold', type=float, default=0.5)

    args = parser.parse_args()
    mergeArgs(args, args.target_dataset)
    return args
