# 🧠 Model Training and Evaluation

本项目旨在实现一个基于深度学习的视觉模型，用于图像分析与预测任务（如目标检测、语义分割或伪装目标检测）。  
本文档说明了模型的训练、验证、评估与结果展示流程。

---

## 📁 项目结构
```text
project/
│
├── configs/             # 配置文件（如超参数设置）
├── datasets/            # 数据集文件或加载脚本
├── models/              # 模型定义
├── train.py             # 模型训练脚本
├── evaluate.py          # 模型评估脚本
├── utils/               # 工具函数（日志、度量指标、可视化）
├── requirements.txt     # 环境依赖
└── README.md            # 说明文档（本文件）
```

## ⚙️ 环境配置

### 1️⃣ 创建环境并安装依赖
```bash
git clone https://github.com/farewellIamLoser/FCT-SAM-WSCOD.git
cd FCT-SAM-WSCOD

# 可选：使用 conda 环境（环境名建议用下划线）
conda create -n FCT_SAM_WSCOD python=3.10
conda activate FCT_SAM_WSCOD

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 主要依赖包（requirements.txt 示例）

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python
tqdm
numpy
matplotlib
pyyaml
```

---

## 📦 数据准备

项目默认数据结构如下：

```
datasets/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

若使用公开数据集（如 COD10K、CAMO、CHAMELEON 等），请下载后放入对应目录。
如需自定义路径，可在 `configs/train.yaml` 中修改。

---

## 🧩 模型结构

模型整体结构如下（示例）：

```
Input Image → Patch Embedding → Transformer Encoder → Decoder → Prediction Mask
```

你可以在 `models/` 文件夹中找到模型定义文件。
若有额外模块（如 SAM 提示、注意力融合等），请在文档中标注。

---

## 🚀 模型训练

### 基本命令

```bash
python train.py --config configs/train.yaml
```

### 常用参数

|       参数       |    默认值   |    含义    |
| :------------: | :------: | :------: |
|   `--epochs`   |    100   |   训练轮数   |
| `--batch_size` |     8    |   批量大小   |
|     `--lr`     |   1e-4   |    学习率   |
|  `--save_path` | ./output |  模型保存路径  |
|   `--resume`   |   None   | 从检查点恢复训练 |

### 输出

* 模型权重文件保存在 `output/checkpoints/`
* 日志文件记录在 `output/logs/`
* TensorBoard 支持：

  ```bash
  tensorboard --logdir output/logs
  ```

---

## 📊 模型评估

### 评估命令

```bash
python evaluate.py --model output/checkpoints/best.pth --data ./datasets/test
```

### 评估指标

|     指标    |      含义      |
| :-------: | :----------: |
|    MAE    |    平均绝对误差（衡量预测结果与真实掩码的像素差异，越小越好）    |
| adaptive F-measure | 自适应 F 值（基于精确率 Precision 与召回率 Recall 的调和平均，综合衡量检测性能） |
|    mean E-measure    |      平均 E 值（同时考虑像素级与结构级一致性，反映预测掩码与真实掩码的整体相似度）     |
|    structure-measure   |    结构相似度指标（衡量预测掩码与真实掩码在结构和区域层面的相似程度）   |

### 评估结果保存

* 定量指标保存至：`output/eval_metrics.json`
* 可视化预测结果保存至：`output/results/`

---

## 🧪 实验结果

|   数据集  |  MAE  |   Fβ  |  IoU  | 参数量(M) |
| :----: | :---: | :---: | :---: | :----: |
| COD10K | 0.031 | 0.864 | 0.771 |  27.6  |
|  CAMO  | 0.055 | 0.828 | 0.695 |  27.6  |
|  N4CK  | 0.052 | 0.828 | 0.695 |  27.6  |
> 可将你训练所得结果替换表格中的值。
> 若有可视化结果，可将样例图放在 `/results/` 下并附链接。

---

## 💾 模型权重下载
| 模型    | 下载链接 | 说明       |
| :------: | :---------------: | :--: |
| Baseline | [Google Drive](https://drive.google.com/your_file_link) | 原始模型 |
| Ours     | [Baidu Cloud](https://pan.baidu.com/s/1F8SK34brd-Nhqjy4sQ8iCw) (提取码: rnig) | 改进版本 |

---

## 🔍 文件说明

|          文件          |      功能     |
| :------------------: | :---------: |
|      `train.py`      |   模型训练主脚本   |
|     `evaluate.py`    |   模型评估主脚本   |
|       `models/`      |    网络结构定义   |
|       `utils/`       | 日志、指标与可视化工具 |
| `configs/train.yaml` |  训练超参数配置文件  |

---

## 🧠 引用（Citation）

如需引用本项目或论文，请使用以下格式：

```bibtex
@inproceedings{wu2025scribble,
  author    = {Zi-Jie Wu and Rongrong Gao and Tian-Zhu Xiang},
  title     = {Scribble-Based Weakly Supervised Camouflaged Object Detection via {SAM}-Guided Feature Correlation Transformer},
  booktitle = {Proceedings of the 27th European Conference on Artificial Intelligence (ECAI 2025)},
  series    = {Frontiers in Artificial Intelligence and Applications},
  pages     = {122--129},
  year      = {2025},
  publisher = {IOS Press},
  doi       = {10.3233/FAIA250797}
}
```

---

## 🙏 致谢

本项目参考了以下开源代码：

* [PyTorch](https://pytorch.org/)
* [timm](https://github.com/rwightman/pytorch-image-models)
* [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
* [CamoFormer]([https://github.com/rwightman/pytorch-image-models](https://github.com/HVision-NKU/CamoFormer))
---

## 🧰 联系方式

如有问题或建议，请联系：
📧 Email: [1021188717@qq.com](mailto:1021188717@qq.com)
👤 Author: Wuzijie

```
