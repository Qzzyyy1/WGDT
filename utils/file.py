import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from scipy.io import loadmat

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def saveFile(path, content):
    file_path, file_name = os.path.split(path)
    check_path(file_path)

    with open(path, 'w') as f:
        f.write(content)

def saveJSONFile(path, save_dict, a=False):
    if a:
        with open(path, 'r') as f:
            ever = json.loads(f.read())
            save_dict = {**ever, **save_dict}

    saveFile(path, json.dumps(save_dict, indent=4))

def saveFig(filename, path='map', dpi=300):
    check_path(path)

    plt.savefig(f'{os.path.join(path, filename)}.png', dpi=dpi)


def saveImage(image, name, path='map'):
    check_path(path)

    plt.imsave(f'{os.path.join(path, name)}.png', image)

read_strategy = {
    None: lambda path, key: loadmat(path)[key],
    'npy': lambda path, key: np.load(path),
    'tif': lambda path, key: tf.imread(path)
}
