# RSML-3：Seal 实验环境与路径

## 服务器

| 项目 | 值 |
|---|---|
| SSH 主机 | `RSML-3` |
| 用户 | `yqwang` |
| 系统 | Ubuntu 22.04 |
| GPU | 2 × NVIDIA GeForce RTX 5090，各约 32 GB |
| 项目根目录 | `/share_datasets/Seal_Model_Reproduction/` |
| Conda | `/home/yqwang/miniforge3/` |
| 本项目环境 | `/home/yqwang/miniforge3/envs/fct_sam_seal/` |
| LRDNet 环境 | `/home/yqwang/miniforge3/envs/lrdnet_seal/` |

`fct_sam_seal` 克隆自 `cd_base`，并安装 `imageio`。实际使用的主要版本：Python 3.10、PyTorch 2.14.0+cu132、torchvision 0.29.0+cu132、timm 1.0.29、OpenCV 5.0.0、SciPy 1.15.2、einops 0.8.2、imageio 2.38.0。

## 数据与代码

| 用途 | 服务器路径 |
|---|---|
| 原始压缩包 | `/share_datasets/Seal_Model_Reproduction/Seal.zip` |
| 解压数据 | `/share_datasets/Seal_Model_Reproduction/Seal_data/` |
| 统一评估脚本 | `/share_datasets/Seal_Model_Reproduction/evaluate.py` |
| FCT-SAM 代码 | `/share_datasets/Seal_Model_Reproduction/FCT-SAM-WSCOD-main/` |
| FCT-SAM 权重 | `/share_datasets/Seal_Model_Reproduction/FCT-SAM-WSCOD-main/pretrain/` |
| FCT-SAM 测试输出 | `/share_datasets/Seal_Model_Reproduction/results/FCT-SAM-WSCOD/Seal/test_maps/` |
| LRDNet 代码与实验 | `/share_datasets/Seal_Model_Reproduction/LRDNet-master/` |
| LRDNet ResNet50 骨干 | `/share_datasets/Seal_Model_Reproduction/LRDNet-master/pth/resnet50-0676ba61.pth` |
| LRDNet 框监督测试输出 | `/share_datasets/Seal_Model_Reproduction/LRDNet-master/experiments/Seal_box/test_maps/` |
| LRDNet 框监督评估日志 | `/share_datasets/Seal_Model_Reproduction/LRDNet-master/experiments/Seal_box/evaluate.log` |
| LRDNet 原始训练图像 | `/share_datasets/Seal_Model_Reproduction/LRDNet-master/data_original/TrainDataset/Imgs/` |
| LRDNet 原始涂鸦标注 | `/share_datasets/Seal_Model_Reproduction/LRDNet-master/data_original/Scribble/` |
| LRDNet 原始协议 Seal 预测 | `/share_datasets/Seal_Model_Reproduction/LRDNet-master/experiments/SCOD/test_maps/` |
| LRDNet 原始协议 Seal 评估 | `/share_datasets/Seal_Model_Reproduction/LRDNet-master/experiments/SCOD/Seal_evaluate.log` |
| MIR-Net 代码与公开权重 | `/share_datasets/Seal_Model_Reproduction/MIR-Net-main/` |
| MIR-Net 跨域测试输出 | `/share_datasets/Seal_Model_Reproduction/results/MIR-Net/Seal/published_test_maps/` |
| 全监督数据集（单模态热红外） | `/share_datasets/Seal_Model_Reproduction/Seal_IRSTD/` |
| DNA-Net 代码 | `/share_datasets/Seal_Model_Reproduction/DNANet-main/` |
| MSHNet 代码 | `/share_datasets/Seal_Model_Reproduction/MSHNet-main/` |
| UIU-Net 代码 | `/share_datasets/Seal_Model_Reproduction/UIU-Net-main/` |
| 全监督基线训练日志 | `/share_datasets/Seal_Model_Reproduction/logs/{dnanet,mshnet,uiunet}_train.log` |
| 全监督基线预测与指标 | `/share_datasets/Seal_Model_Reproduction/results/<Method>/Seal/fullsup_*` |

`Seal_IRSTD/` 是全监督 IRSTD 基线使用的单模态副本：`images/<stem>.png`（由 `Seal_data/{split}/T` 的 16 位热红外逐图 min-max 拉伸到 8 位再复制为 3 通道）、`masks/<stem>.png`（GT 二值化）、`trainval.txt`/`val.txt`/`test.txt`（1,500/500/1,983），`dataset_meta.json` 记录转换方式。生成脚本见 `tools/prepare_seal_irstd.py`。

`Seal.zip` 解压后顶层目录为 `Seal/`，服务器上已命名为 `Seal_data/`。每个划分包含 `RGB/`、`T/`、`GT/`、`box/`：

| 划分 | RGB | GT |
|---|---:|---:|
| train | 1,500 | 1,500 |
| val | 500 | 500 |
| test | 1,983 | 1,983 |

## FCT-SAM 实验文档

测试与重训配置、运行命令和结果统一见 [FCT-SAM-WSCOD_Seal实验记录.md](FCT-SAM-WSCOD_Seal实验记录.md)。

## 全监督基线

四个 IRSTD 全监督方法的数据、训练、推理与评估记录：[ACM](ACM_Seal实验记录.md)、[DNA-Net](DNANet_Seal实验记录.md)、[MSHNet](MSHNet_Seal实验记录.md)、[UIU-Net](UIU-Net_Seal实验记录.md)。三个进入主表的方法统一使用 `Seal_IRSTD/`，训练在 GPU 上并行执行，测试预测写入 `results/<Method>/Seal/fullsup_test_maps/`。

统一评估脚本使用 `Ours/evaluate.py` 中固定不变的四个原函数。根目录入口的四个函数与其原文一致；E 使用原函数的连续灰度计算方式。各方法的 S、Fw、E、MAE 均须遵循该口径。
