https://github.com/Li-ZK/WGDT.

This is a code demo for the paper "Open-Set Domain Adaptation for Hyperspectral Image Classification Based on Weighted Generative Adversarial Networks and Dynamic Thresholding".

[1]Ke Bi, Zhaokui Li, Yushi Chen, Qian Du, Li Ma, Yan Wang, Zhuoqun Fang, Mingtai Qi, “Open-Set Domain Adaptation for Hyperspectral Image Classification Based on Weighted Generative Adversarial Networks and Dynamic Thresholding,” IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2025.3549951.

## Requirements

python = 3.7.15

torchmetrics = 0.10.3

pytorch = 1.12.1

scikit-learn = 1.0.2

scipy = 1.7.3

You can download the source and target datasets mentioned above at https://pan.baidu.com/s/1BY0EqAWe1BOherY7kZypHQ?pwd=vkde, and move to folder datasets.
An example datasets folder has the following structure:

```
datasets
├── Pavia_7gt
├── PaviaC
├── Houston_7gt
├── Houston18
├── HyRANK
└── Yancheng
```

## Usage:

run main.py

### Batch seeds for HOS

You can run all supported dataset pairs with the fixed seed lists below and summarize HOS / class accuracy by:

- `PaviaU_7gt -> PaviaC_OS`: `7, 43, 45, 46, 66, 67, 77, 81, 88, 98`
- `Houston13_7gt -> Houston18_OS`: `1, 23, 30, 35, 52, 64, 68, 72, 73, 91`
- `HyRank_source -> HyRank_target`: `6, 13, 20, 40, 49, 58, 60, 67, 81, 91`
- `Yancheng_ZY -> Yancheng_GF`: `15, 18, 19, 24, 37, 50, 51, 62, 77, 80`

```bash
python run_hos_seeds.py --device 0
```

On a multi-GPU server, you can dispatch jobs to 4 GPUs in parallel by:

```bash
python run_hos_seeds.py --gpus 0 1 2 3
```

Each subprocess is isolated with `CUDA_VISIBLE_DEVICES`, so every task only occupies one physical GPU.

Results will be saved under `logs/WGDT/`, including a JSON summary file, a CSV summary file, and a Markdown table file.
- `HOS` is summarized as `mean ± std`
- `class_acc_*` columns are the mean accuracy of each class across seeds
- the Markdown file is ready to paste into notes or a paper draft

If you want to skip finished runs and only aggregate existing results:

```bash
python run_hos_seeds.py --gpus 0 1 2 3 --skip_existing
```
