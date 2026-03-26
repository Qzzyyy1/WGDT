import torch
from torch import nn
from typing import Callable
from tqdm import tqdm

from utils.Optimizer import OptimizerManager
from utils.pyExt import dataToDevice, getFunc
from utils.logger import ProgressLogger
from utils.typing import Sequence, Collecter, Loader

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def train(self, hook: str, dataloader: Loader, epochs: int):
        self.model.train()
        progress = ProgressLogger(epochs)
        self.model.progress = progress

        optimizer = getattr(self.model, f'{hook}_optimizer')()
        if type(optimizer) not in [list, tuple]:
            optimizer = [optimizer]
        loop_step: Callable = getattr(self.model, f'{hook}_step')
        epoch_end = getFunc(self.model, f'{hook}_epoch_end')

        for epoch in range(epochs):
            
            for data in dataloader:
                data = dataToDevice(data, self.device)
                step_out = loop_step(data)
                loss, information = parseTrainStepOut(step_out)

                with OptimizerManager(optimizer):
                    loss.backward()

                progress.add_information(information)

            epoch_out: dict = epoch_end()
            progress.update(epoch_out)
        
        progress.close()

    def test(self, hook: str, dataloader: Loader):
        self.model.eval()

        loop_step: Callable = getattr(self.model, f'{hook}_step')
        test_end = getFunc(self.model, f'{hook}_end')

        with torch.no_grad():
            for data in tqdm(dataloader):
                data = dataToDevice(data, self.device)
                loop_step(data)

            test_end()

def parseTrainStepOut(step_out: Collecter) -> Sequence:
    out_type = type(step_out)

    if out_type == dict:
        loss = step_out['loss']
        information = step_out['information']
    elif out_type == list or out_type == tuple:
        loss = step_out[0]
        information = step_out[1]
    else:
        loss = step_out
        information = dict(loss=loss)

    return loss, information
