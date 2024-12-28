# UniMatch V2 for Remote Sensing Change Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-change-detection-on-levir-cd)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-change-detection-on-levir-cd-1)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd-1?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-change-detection-on-levir-cd-2)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd-2?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-change-detection-on-whu-20)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-20?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-change-detection-on-whu-40)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-40?p=unimatch-v2-pushing-the-limit-of-semi)

This codebase contains the official PyTorch implementation of <b>UniMatch V2</b> in **semi-supervised remote sensing change detection**:

> **[UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2410.10777)**</br>
> Lihe Yang, Zhen Zhao, Hengshuang Zhao</br>
> Preprint, 2024

## Results

**We provide the [training log of each reported value](https://github.com/LiheYoung/UniMatch-V2/blob/main/training-logs). You can refer to them during reproducing. We also provide all the [checkpoints](https://huggingface.co/LiheYoung/UniMatch-V2/tree/main) of our core experiments.**


### LEVIR-CD

The two numbers in each cell denote the **changed-class IoU** and **overall accuracy**, respectively.

| Method                      | 5%        | 10%       | 20%       | 40%       |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: |
| [SemiCD](https://github.com/wgcban/SemiCD)                      | 72.5 / 98.47      | 75.5 / 98.63      | 76.2 / 98.68      | 77.2 / 98.72      |
| [UniMatch V1](https://github.com/LiheYoung/UniMatch)   | 80.7 / 98.95  | 82.0 / 99.02  | 81.7 / 99.02  | 82.1 / 99.03  |
| [SemiCD-VL](https://github.com/likyoo/SemiCD-VL)   | 81.9 / 99.02 | 82.6 / 99.06 | 82.7 / 99.05 | 83.0 / 99.07 |
| **UniMatch V2** | **83.3** / **99.08** | **83.8** / **99.11** | **84.3** / **99.14** | **84.3** / **99.14** |

### WHU-CD

The two numbers in each cell denote the **changed-class IoU** and **overall accuracy**, respectively.

| Method                      | 20%       | 40%       |
| :-------------------------: | :-------: | :-------: |
| [SemiCD](https://github.com/wgcban/SemiCD)             | 74.8 / 98.84  | 77.2 / 98.96  |
| [UniMatch V1](https://github.com/LiheYoung/UniMatch)   | 81.7 / 99.18  | 85.1 / 99.35  |
| [SemiCD-VL](https://github.com/likyoo/SemiCD-VL)       | 84.8 / 99.36  | 85.7 / 99.39  |
| **UniMatch V2** | **87.9** / **99.50** | **88.6** / **99.52** |


## Getting Started

### Pre-trained Encoders

[DINOv2-Small](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | [DINOv2-Base](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)

```
├── ./pretrained
    ├── dinov2_small.pth
    └── dinov2_base.pth
```

### Datasets

- LEVIR-CD: [imageA, imageB, and label](https://huggingface.co/LiheYoung/UniMatch-V2/resolve/main/LEVIR-CD.zip)
- WHU-CD: [imageA, imageB, and label](https://huggingface.co/LiheYoung/UniMatch-V2/resolve/main/WHU-CD.zip)

Please modify your dataset path in configuration files.

```
├── [Your LEVIR-CD/WHU-CD Path]
    ├── A
    ├── B
    └── label
```


## Training

### UniMatch V2

```bash
sh scripts/train.sh <num_gpu> <port>
# to fully reproduce our results, the <num_gpu> should be set as 1 for all settings
# otherwise, you need to adjust the learning rate accordingly
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch-V2/blob/main/remote-sensing/scripts/train.sh).

### Supervised Baseline

Modify the ``method`` from ``'unimatch_v2'`` to ``'supervised'`` in [train.sh](https://github.com/LiheYoung/UniMatch-V2/blob/main/remote-sensing/scripts/train.sh). 


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{unimatchv2,
  title={UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Zhao, Zhen and Zhao, Hengshuang},
  journal={arXiv:2410.10777},
  year={2024}
}
```


## Acknowledgement

The processed LEVIR-CD and WHU-CD datasets are originally processed by [SemiCD](https://github.com/wgcban/SemiCD). However, the downloading links are invalid now. Therefore, we borrow thw downloaded files from [Kaiyu Li](https://likyoo.github.io/). We thank their contributions and help.
