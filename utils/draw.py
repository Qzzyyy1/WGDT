import os

import numpy as np
import torch
from typing import List

from utils.file import saveImage
from utils.typing import Matrix

def getColors():
    return np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
            [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
            [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
            [0, 0, 128]])

def getClassificationMap(label: Matrix, unknown=[], unknown_color=[255, 255, 255]):
    colors = getColors()
    image = np.zeros((*label.shape, 3), dtype='uint8')
    for cls in range(1, label.max() + 1):
        image[np.where(label == cls)] = colors[cls - 1]

    for cls in unknown:
        image[np.where(label == cls)] = unknown_color

    return image

def clearBackground(info, image, known_classes=None, unknown_classes=None):
    from utils.splitData import transformGT

    gt = transformGT(None, info, known_classes, unknown_classes)
    image[np.where(gt == 0)] = [0, 0, 0]

    return image

def parsePredictionLabel(label: List[torch.Tensor], H):
    label = torch.cat(label).cpu() + 1
    return label.reshape(H, -1)

def drawPredictionMap(label: List[torch.Tensor], name, info, known_classes=[], unknown_classes=[], draw_background=True):
    label = parsePredictionLabel(label, info.image_width)
    image = getClassificationMap(label, unknown=[len(known_classes) + 1])
    if draw_background is False:
        image = clearBackground(info, image, known_classes, unknown_classes)
    saveImage(image, name, 'map')

def drawGTMap(dataset: str, path='map', known_classes=[], unknown_classes=[], unknown_color=[255, 255, 255]):
    from utils.utils import getDatasetInfo
    from utils.splitData import transformGT
    info = getDatasetInfo(dataset)
    gt = transformGT(None, info, known_classes, unknown_classes)
    image = getClassificationMap(gt, unknown=[len(known_classes) + 1], unknown_color=unknown_color)
    saveImage(image, dataset, path)

# http://www.spectralpython.net/
def drawDatasetImage(dataset: str, rgb=[60, 30, 10], path='map'):
    import spectral
    from utils.utils import getDatasetInfo, getDataByInfo

    info = getDatasetInfo(dataset)
    data = getDataByInfo(info)

    if not os.path.exists(path):
        os.makedirs(path)

    filename = f'{dataset} rgb'
    spectral.save_rgb(f'{os.path.join(path, filename)}.jpg', data, rgb)

def drawColorBanner(height=30, width=50, border=2, path='map'):
    colors = getColors()
    image = np.zeros((height + 2 * border, width + 2 * border, 3), dtype='uint8')

    saveImage(image, 'class=background color=[0, 0, 0]', path)

    for cls, color in enumerate(colors):
        image[border : -border, border : -border] = color
        saveImage(image, f'class={cls} color={color}', path)

    image[border : -border, border : -border] = [255, 255, 255]
    saveImage(image, 'class=unknown color=[255, 255, 255]', path)
