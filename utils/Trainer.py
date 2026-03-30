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
        step_end = getFunc(self.model, f'{hook}_step_end')

        for epoch in range(epochs):
            
            for data in dataloader:
                data = dataToDevice(data, self.device)
                step_out = loop_step(data)
                loss, information = parseTrainStepOut(step_out)

                with OptimizerManager(optimizer):
                    loss.backward()

                if step_end is not None:
                    step_end()
                progress.add_information(information)

            epoch_out: dict = epoch_end()
            progress.update(epoch_out)
        
        progress.close()

    def test(self, hook: str, dataloader: Loader):
        self.model.eval()

        loop_step: Callable = getattr(self.model, f'{hook}_step')
        test_start = getFunc(self.model, f'{hook}_start')
        test_end = getFunc(self.model, f'{hook}_end')
        test_finish = getFunc(self.model, f'{hook}_finish')

        with torch.no_grad():
            if test_start is not None:
                test_start()
            for data in tqdm(dataloader):
                data = dataToDevice(data, self.device)
                loop_step(data)

            test_end()
            if test_finish is not None:
                test_finish()

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
