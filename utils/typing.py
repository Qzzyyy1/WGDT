from typing import Union
import torch
from torch.utils.data import DataLoader
import numpy as np

Sequence = Union[list, tuple]
Matrix = Union[torch.Tensor, np.ndarray]
MatrixSequence = Union[Sequence, Matrix]

Collecter = Union[Sequence, dict]

Loader = DataLoader
