import json
import torch
import random
import os
import numpy as np

from utils.file import read_strategy

from .pyExt import Dict2Obj

def mergeArgs(args, dataset):
    with open("datasets/dataset_params.json", "r", encoding="utf-8") as f:
        info = json.load(f)[dataset]
    
    for key, value in info.items():
        setattr(args, key, value)

def getDatasetInfo(dataset):
    with open("datasets/dataset_config.json", "r", encoding="utf-8") as f:
        info = json.load(f)[dataset]

    return Dict2Obj(info)

def getDataByInfo(info):

    path = os.path.join('./datasets', info.path, info.file_name)
    data = read_strategy[info.type](path, info.mat_name)

    return data.astype(np.float32)

def getGTByInfo(info):

    path = os.path.join('./datasets', info.path, info.gt_file_name)
    gt = read_strategy[info.type](path, info.gt_mat_name)
    
    return gt.astype(np.int64)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getDevice(device=None):
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device == -1:
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{device}')
