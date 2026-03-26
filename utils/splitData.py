import numpy as np
import torch
import math
from sklearn.model_selection import train_test_split

from utils.utils import getDataByInfo, getGTByInfo
from utils.augment import applyPCA
from utils.typing import Matrix

def transformGT(args, info, known_classes, unknown_classes):
    gt_mat = getGTByInfo(info)

    unknown_class_index = len(known_classes) + 1

    gt_transform = np.zeros_like(gt_mat)

    for index, cls in enumerate(unknown_classes):
        gt_transform[np.where(gt_mat == cls)] = unknown_class_index

    for index, cls in enumerate(known_classes):
        gt_transform[np.where(gt_mat == cls)] = index + 1

    return gt_transform

def paddingData(dataset, patch_size):
    distance = patch_size // 2
    C, H, W = dataset.shape
    # center
    data = torch.zeros(size=(C, H + 2 * distance, W + 2 * distance), dtype=torch.float)
    data[:, distance:distance+H, distance:distance+W] = dataset
    # flip
    data[:, 0:distance, :] = data[:, distance:distance+distance, :].flip(1) # top
    data[:, -distance:, :] = data[:, -distance*2:-distance, :].flip(1) # bottom
    data[:, :, 0:distance] = data[:, :, distance:distance*2].flip(2) # left
    data[:, :, -distance:] = data[:, :, -distance*2:-distance].flip(2) # right
    return data

def getSourceTrainIndex(gt: Matrix, args, info):
    train_index_list = []

    for cls in range(1, gt.max() + 1):
        x, y = np.where(gt == cls)
        locations = x * gt.shape[1] + y

        sample_num = x.shape[0]
        train_num = args.train_num
        if train_num > sample_num:
            if hasattr(args, 'few_train_num'):
                train_num = args.few_train_num
            else:
                raise ValueError('Error: train_num > sample_num')
        if train_num == 0:
            train_num = math.ceil(sample_num * args.train_rate)

        if train_num == sample_num:
            train_index_list.extend(locations)
        else:
            train_index, test_index = train_test_split(locations, train_size=train_num, random_state=args.seed)
            train_index_list.extend(train_index)

    print(f'source train samples: {len(train_index_list)}')

    return train_index_list

def getTargetIndex(gt: Matrix, args, info):

    x, y = np.where(np.logical_and(gt > 0, gt < gt.max()))
    locations = x * gt.shape[1] + y
    known_index_list = locations.tolist()

    x, y = np.where(gt == gt.max())
    locations = x * gt.shape[1] + y
    unknown_index_list = locations.tolist()

    print(f'target samples: {len(known_index_list)} + {len(unknown_index_list)} = {len(known_index_list) + len(unknown_index_list)}')

    return {
        'all_index_list': [*known_index_list, *unknown_index_list],
        'known_index_list': known_index_list,
        'unknown_index_list': unknown_index_list
    }

def initData(args, info):
    assert args.patch % 2, 'Error: patch size'

    data = getDataByInfo(info)
    data = torch.from_numpy(data).permute(2,0,1)
    if hasattr(args, 'pca') and args.pca > 0:
        data = applyPCA(data, args.pca)
    if info.norm is not True:
        data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    data = paddingData(data, args.patch)

    return data

def initDataset(args, info, known_classes=[], unknown_classes=[]):
    
    data = initData(args, info)
    gt = transformGT(args, info, known_classes, unknown_classes)

    return {
        'data': data,
        'gt': torch.from_numpy(gt)
    }