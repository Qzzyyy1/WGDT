from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo, seed_torch

from model.WGDT import get_model, parse_args, run_model

if __name__ == '__main__':
    # 解析命令行参数和配置文件
    args = parse_args()
    # 固定随机数种子
    seed_torch(args.seed)
    # 获取数据集信息
    source_info = getDatasetInfo(args.source_dataset)
    target_info = getDatasetInfo(args.target_dataset)
    # 数据加载
    data_loader: dict = getDataLoader(args, source_info, target_info, drop_last=True)
    # 初始化模型
    model = get_model(args, source_info, target_info)
    # 启动模型
    run_model(model, data_loader)
