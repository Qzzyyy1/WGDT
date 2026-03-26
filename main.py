import argparse
import importlib

from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo, seed_torch


def get_model_module():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, choices=['WGDT', 'UOT_OSDA'], default='WGDT')
    args, _ = parser.parse_known_args()
    return importlib.import_module(f'model.{args.model_name}')


if __name__ == '__main__':
    model_module = get_model_module()
    args = model_module.parse_args()
    seed_torch(args.seed)
    source_info = getDatasetInfo(args.source_dataset)
    target_info = getDatasetInfo(args.target_dataset)
    data_loader: dict = getDataLoader(args, source_info, target_info, drop_last=True)
    model = model_module.get_model(args, source_info, target_info)
    model_module.run_model(model, data_loader)
