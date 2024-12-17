# BiDNet
This repo is the official implementation of "BiDNet: A Real-Time Semantic Segmentation Network with Anti-Feature Interference and Detail Recovery for Industrial Defects". This repo contains the supported code, configuration files, and datasets to reproduce the semantic segmentation results of BiDNet. The code is based on [MMSegmentaion V1.2.2.](https://github.com/open-mmlab/mmsegmentation/tree/main) All experiments were performed on a single NVIDIA GTX 3090Ti GPU in CUDA 11.7, Python 3.8, and PyTorch 1.13.1.

## Installation
Step 1. Create a conda environment and activate it.
```
conda create --name BIDNet python=3.8 -y
conda activate BIDNet
```
Step 2. Install PyTorch following [official instructions](https://pytorch.org/get-started/previous-versions/), e.g.
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```
Step 3. Install MMCV using MIM.
```
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0rc4
```
Step 4. Install BIDNet.
```
git clone -b main https://github.com/jiaweipan997/BiDNet.git
cd BIDNet
pip install -v -e .
pip install ftfy
pip install regex
```

## Verify the installation
Step 1. We need to download config and checkpoint files. When it is done, you will find two files pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py and pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth in your current folder.
```
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
```
Step 2. Verify the inference demo. You will see a new image result.jpg on your current folder, where segmentation masks are covered on all objects.
```
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```

## Dataset prepare
The preprocessed dataset can be downloaded from this [link](https://pan.baidu.com/s/1yqEHECbgDbgxbJRc-bAZag?pwd=s4hn) with the code s4hn.
Create a new data folder and put the downloaded dataset in it and unzip it. The structure is as follows：
```
BiDNet
├── mmseg
├── tools
├── configs
├── data
│   ├── MSD
│   │   ├── imgs
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── labels
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test

```

## Train
```
# MSD dataset:
bash tools/dist_train.sh configs/BIDNet/bidnet_mscan-t_1xb8-adamw-80k_msd-512x512.py 1
# MTD dataset:
bash tools/dist_train.sh configs/BIDNet/bidnet_mscan-t_1xb4-adamw-40k_mtd-512x512.py 1
# GSD dataset:
bash tools/dist_train.sh configs/BIDNet/bidnet_mscan-t_1xb16-adamw-160k_gsd-512x512.py 1
```

## Test
```
# MSD dataset:
python tools/test.py configs/BIDNet/bidnet_mscan-t_1xb8-adamw-80k_msd-512x512.py  work_dirs/bidnet_mscan-t_1xb8-adamw-80k_msd-512x512/iter_80000.pth
# MTD dataset:
python tools/test.py configs/BIDNet/bidnet_mscan-t_1xb4-adamw-40k_mtd-512x512.py work_dirs/bidnet_mscan-t_1xb4-adamw-40k_mtd-512x512/iter_40000.pth
# GSD dataset:
python tools/test.py configs/BIDNet/bidnet_mscan-t_1xb16-adamw-160k_gsd-512x512.py work_dirs/bidnet_mscan-t_1xb16-adamw-160k_gsd-512x512/iter_160000.pth
```

## Get Params and FLOPs
```
python tools/analysis_tools/get_flops.py configs/BIDNet/bidnet_mscan-t_1xb8-adamw-80k_msd-512x512.py --shape 512
```

## Get FPS
```
python tools/analysis_tools/benchmark.py configs/BIDNet/bidnet_mscan-t_1xb8-adamw-80k_msd-512x512.py work_dirs/bidnet_mscan-t_1xb8-adamw-80k_msd-512x512/iter_80000.pth
```
