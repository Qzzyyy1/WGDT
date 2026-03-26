import torch
import torch.nn as nn
from torchmetrics import Accuracy

from utils.dann import DomainDiscriminator, DomainAdversarialLoss
from utils.meter import OpensetDomainMetric
from utils.Trainer import Trainer
from utils.dataLoader import CombinedLoader
from utils.utils import mergeArgs
from .DCRN import DCRN
from .Anchor import Anchor
from .Radius import Radius

class Model(nn.Module):
    def __init__(self, args, source_info, target_info, device, in_channels, patch, known_num_classes, anchor_type, radius_loss_type, radius_init, radius_margin, alpha, domain_loss_weight, radius_loss_weight):
        super().__init__()

        self.args = args
        self.source_info = source_info
        self.target_info = target_info
        self.device = device

        self.in_channels = in_channels
        self.patch = patch
        self.known_num_classes = known_num_classes
        self.alpha = alpha
        self.domain_loss_weight = domain_loss_weight
        self.radius_loss_weight = radius_loss_weight

        self.feature_encoder = DCRN(in_channels, patch, known_num_classes)
        self.classifier = nn.Linear(288, known_num_classes)
        self.disc_encoder = DomainDiscriminator(in_feature=288, hidden_size=64)
        self.domain_adv = DomainAdversarialLoss(self.disc_encoder)
        self.anchor = {
            'anchor': lambda: Anchor(known_num_classes, anchor_weight=args.anchor_weight, alpha=self.alpha),
        }[anchor_type]()
        self.radius = Radius(radius_init, radius_margin, radius_loss_type)

        self.source_oa = Accuracy()
        self.metric = OpensetDomainMetric(self.known_num_classes, self.args)
        self.prediciton_all = []

    # 预训练
    def pre_train_step(self, batch):
        x, y = batch
        loss = self(x, y)['loss']

        return loss

    # 预训练优化器
    def pre_train_optimizer(self):
        optimizer = torch.optim.SGD([
            {'params': self.feature_encoder.parameters()},
            {'params': self.classifier.parameters()},
            {'params': self.anchor.parameters()}
        ], lr=0.001, momentum=0.9, weight_decay=5e-4)

        return optimizer

    def train_step(self, batch):
        [source_x, source_y], [target_x, target_y] = batch

        source_out = self(source_x, source_y)
        target_out = self(target_x)

        gamma = target_out['gamma']
        min_gamma = gamma.min(1)[0]
        distance = target_out['distance']
        min_distance = distance.min(1)[0].detach()
        weight = (min_distance.max() - min_distance) / (min_distance.max() - min_distance.min())

        loss_disc = self.domain_adv(source_out['features'], target_out['features'], w_t=weight)
        loss_radius = self.radius(min_gamma.detach(), weight=weight)

        loss_dic = dict(
            loss_anchor = source_out['loss_anchor'] * self.alpha,
            loss_tuplet = source_out['loss_tuplet'],
            loss_disc = loss_disc * self.domain_loss_weight,
            loss_radius = loss_radius * self.radius_loss_weight
        )

        self.source_oa.update(source_out['prediction'], source_y)

        return {
            'loss': sum(value for value in loss_dic.values()),
            'information': loss_dic
        }
    
    def train_epoch_end(self):
        dic = {
            'source_oa': self.source_oa.compute()
        }
        self.source_oa.reset()

        return dic

    def train_optimizer(self):
        momentum = 0.9
        l2_decay = 5e-4
        optimizer = torch.optim.SGD([
            {'params': self.feature_encoder.parameters()},
            {'params': self.classifier.parameters()},
            {'params': self.anchor.parameters()}
        ], lr=self.args.lr_encoder, momentum=momentum, weight_decay=l2_decay)
        optimizer_critic = torch.optim.SGD(self.disc_encoder.parameters(), lr=self.args.lr_domain, momentum=momentum, weight_decay=l2_decay)
        optimizer_raiuds = torch.optim.SGD(self.radius.parameters(), lr=1e-4, momentum=momentum, weight_decay=l2_decay)

        return optimizer, optimizer_critic, optimizer_raiuds

    def test_step(self, batch):
        x, y = batch

        out = self(x)
        prediction = out['prediction']
        gamma = out['gamma']

        score, prediction = torch.min(gamma, 1)
        prediction[score > self.radius.radius] = self.known_num_classes

        self.metric.update(prediction, y)
    
    def test_end(self):
        self.metric.finish()

    def prediction_step(self, batch):
        x = batch

        out = self(x)
        gamma = out['gamma']
        score, prediction = torch.min(gamma, 1)

        prediction[score > self.radius.radius] = self.known_num_classes
        self.prediciton_all.append(prediction)

    def prediction_end(self):
        from utils.draw import drawPredictionMap

        drawPredictionMap(self.prediciton_all, f'{self.args.log_name} {self.args.target_dataset}', self.target_info, known_classes=self.args.target_known_classes, unknown_classes=self.args.target_unknown_classes, draw_background=False)

    def forward(self, x, y=None):
        features = self.feature_encoder(x)['features']
        logits = self.classifier(features)
        anchor_out = self.anchor(logits, y)

        return {
            'features': features,
            **anchor_out
        }


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
        'anchor_type': args.anchor_type,
        'radius_loss_type': args.radius_loss_type,
        'radius_init': args.radius_init,
        'radius_margin': args.radius_margin,
        'alpha': args.alpha,
        'domain_loss_weight': args.domain_loss_weight,
        'radius_loss_weight': args.radius_loss_weight,
    }
    return Model(**model_args)

def parse_args():
    # 配置和解析程序运行时的超参数和设置
    import argparse

    # seed
    # PaviaU_7gt -> PaviaC_OS [7, 43, 45, 46, 66, 67, 77, 81, 88, 98]
    # Houston13_7gt -> Houston18_OS [1, 23, 30, 35, 52, 64, 68, 72, 73, 91]
    # HyRank_source -> HyRank_target [6, 13, 20, 40, 49, 58, 60, 67, 81, 91]
    # Yancheng_ZY ->'Yancheng_GF [15, 18, 19, 24, 37, 50, 51, 62, 77, 80]

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default='WGDT')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_num', type=int, default=180)
    parser.add_argument('--few_train_num', type=int, default=150)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--patch', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--source_dataset', choices=['Houston13_7gt', 'PaviaU_7gt', 'HyRank_source', 'Yancheng_ZY'], default='Houston13_7gt')
    parser.add_argument('--target_dataset', choices=['Houston18_OS', 'PaviaC_OS', 'HyRank_target', 'Yancheng_GF'], default='Houston18_OS')
    parser.add_argument('--anchor_type', type=str, choices=['anchor'], default='anchor'),
    parser.add_argument('--radius_loss_type', type=str, choices=['MarginMSELoss'], default='MarginMSELoss')
    parser.add_argument('--radius_init', type=float, default=0.0)

    parser.add_argument('--pre_train', type=str, default='True')
    parser.add_argument('--pre_train_epochs', type=int, default=20)

    parser.add_argument('--draw', type=str, default='False')

    args = parser.parse_args()

    mergeArgs(args, args.target_dataset)

    return args