import torch
from typing import Callable

from utils.typing import Sequence, Collecter

class Dict2Obj(dict):
    def __getattr__(self, key):
        if key not in self:
            return None
        else:
            value = self[key]
            if isinstance(value,dict):
                value = Dict2Obj(value)
            return value
        
def applyFuncForCollector(data: Collecter, func: Callable):
    dataType = type(data)

    if dataType in [int, float]:
        return data

    if dataType == torch.Tensor:
        return func(data)

    if dataType in [list, tuple]:
        collector = []
        for value in data:
            collector.append(applyFuncForCollector(value, func))
        return collector
    elif dataType == dict:
        collector = {}
        for key, value in data.items():
            collector[key] = applyFuncForCollector(value, func)
        return collector
    else:
        raise TypeError('Invalid type')


def dictTensorItem(dic: dict) -> dict:

    return applyFuncForCollector(dic, lambda x: x.tolist())

def dataToDevice(data: Collecter, device: torch.device) -> Collecter:

    return applyFuncForCollector(data, lambda x: x.to(device))

def getFunc(_class: object, name: str) -> Callable:

    if hasattr(_class, name):
        return getattr(_class, name)
    else:
        return lambda: None

def find_min_length(li: Sequence) -> int:

    min_length = len(li[0])

    for item in li:
        min_length = min(min_length, len(item))

    return min_length
