from tqdm import tqdm
from typing import Optional

from utils.pyExt import dictTensorItem

class ProgressLogger:
    def __init__(self, epochs: int):

        self.epoch = 0

        self.progress_bar = tqdm(range(1, epochs + 1))

        self.information = {}

    def update(self, dic: Optional[dict] = None):
        self.progress_bar.update()
        self.progress_bar.set_description(f'epoch: {self.epoch + 1}')
        if dic is not None:
            self.add_information(dic)

        self.epoch += 1

    def add_information(self, dic: dict):
        dic = dictTensorItem(dic)
        self.information.update(dic)
        self.progress_bar.set_postfix(self.information)

    def close(self):
        self.progress_bar.close()
