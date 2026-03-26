import torch
import json

from utils.file import saveJSONFile
from utils.typing import MatrixSequence
from utils.pyExt import dictTensorItem

def computeOpensetDomainResult(prediction: MatrixSequence, label: MatrixSequence, known_num_classes: int):
    from torchmetrics import Accuracy
    from torchmetrics.classification import MulticlassAccuracy

    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)

    known_mask = label < known_num_classes
    unknown_mask = label == known_num_classes

    device = label.device

    oa_meter = Accuracy().to(device)
    aa_meter = MulticlassAccuracy(known_num_classes + 1, average=None).to(device)
    known_meter = Accuracy().to(device)
    unknown_meter = Accuracy().to(device)

    oa = oa_meter(prediction, label)
    aa = aa_meter(prediction, label).mean()
    classes_acc = aa_meter(prediction, label)
    oa_known = known_meter(prediction[known_mask], label[known_mask])
    aa_known = classes_acc[:-1].mean()
    unknown = unknown_meter(prediction[unknown_mask], label[unknown_mask])
    hos = (2 * aa_known * unknown) / (aa_known + unknown)

    return dictTensorItem({
        'oa': oa,
        'aa': aa,
        'classes_acc': classes_acc,
        'oa_known': oa_known,
        'aa_known': aa_known,
        'unknown': unknown,
        'hos': hos
    })

class PredictionTargetGather:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prediction_list = []
        self.target_list = []

    def update(self, prediction, target):
        assert prediction.shape == target.shape, 'Error: The prediction and target shapes are different.'

        self.prediction_list.append(prediction)
        self.target_list.append(target)

    def get(self):
        return torch.cat(self.prediction_list), torch.cat(self.target_list)

class OpensetDomainMetric:
    def __init__(self, known_num_classes, args):

        self.known_num_classes = known_num_classes
        self.save_path = f'logs/{args.log_name}/{args.log_name} {args.source_dataset}-{args.target_dataset} seed={args.seed}.json'

        self.reset()

    def reset(self):
        self.gather = PredictionTargetGather()
        self.save_dict = None

    def update(self, prediction, target):
        self.gather.update(prediction, target)

    def compute(self):
        self.save_dict = computeOpensetDomainResult(*self.gather.get(), self.known_num_classes)
        return self.save_dict
    
    def save(self, a=False):
        saveJSONFile(self.save_path, self.save_dict, a=a)

    def print(self):
        print(json.dumps(self.save_dict, indent=4))

    def finish(self, a=False):
        self.compute()
        self.print()
        self.save(a=a)
        self.reset()
