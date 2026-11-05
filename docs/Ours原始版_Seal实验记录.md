# Ours（原始版，RGB-T + 框提示双解码器）：Seal 训练与测试记录

> **历史版本（2026-11-04，`Ours/Original_Code/`）**。最终代码是 `Ours/CODE/`（CM-LoRA + BISF 融合），对应预测与指标见 [Ours_Seal实验记录.md](Ours_Seal实验记录.md)（1105 批次）。本文件仅作版本演进记录。

代码：`Ours/Original_Code/`（作者原始实现，本仓库只做运行相关修补，见下）。训练、验证、
测试全部使用 `Seal_data/{train,val,test}`，train 1,500 / val 500 / test 1,983。

## 协议

| 项目 | 值 |
|---|---|
| 输入 | Seal `train/RGB`（jpg）+ `train/T`（16 位 TIFF，逐图 min-max → 8 位 → RGB） |
| 监督 | `train/box` 的绝对像素框（`x1 y1 x2 y2`），无 GT mask 参与训练 |
| 网络 | SAM3 tracker 主干；RGB 走 LoRA 分支，T 走 ResNet50 + Bottleneck 温度分支；双 mask decoder（无提示 / 框提示） |
| 损失 | `structure_loss(mask1, sigmoid(mask2))` + `projection_loss` + LSC 平滑 + 框内前景项 |
| 模型选择 | Seal **val** 500 张的 MAE 最优（原始代码的选点规则） |
| 测试 | `test/RGB` + `test/T`，无提示分支 `mask_1`，双线性恢复到 GT 尺寸后逐图 min-max → 8 位 PNG |
| 指标 | 根目录 `evaluate.py` 四原函数 + `evaluate_small_targets.py` 小目标四项，同一批 1,983 张 |

## 相对原始代码的改动（不动网络与损失）

- `utils.py` 新增 `open_thermal`（Seal 16 位 T 的 min-max），`data.py` 的 T 分支改用它；
  原 `open_rgb` 会把 16 位 TIFF 截断到 255。
- `data.py` 的箱目录同时接受 Seal 的 `box/`（原来只认 `bbox/`）。
- `sam3/sam3model.py`：ResNet50 权重路径参数化（`--resnet_ckpt`），缺文件时报错清晰。
- `train.py` / `test.py`：设备由 `SEAL_CUDA` 控制（默认 1），数据与权重默认路径指向本项目。

## 环境与命令

服务器 RSML-3；独立 venv `Ours/venv_ours`（基于 `fct_sam_seal` 的 system-site-packages，
补装 iopath/hydra-core/omegaconf/einops/ftfy/regex/huggingface_hub/psutil/decord/pycocotools）。

```bash
bash /share_datasets/Seal_Model_Reproduction/Ours/run_original.sh 30 1
```

权重（服务器 `/share_datasets/Seal_Model_Reproduction/Ours/weights/`，主干不进本仓库，本地副本已删除）：`sam3.pt`（3,450,062,241 字节，SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`，删除前已与服务器逐字节一致）、`resnet50-19c8e357.pth`。

训练设置：30 epoch、lr 5e-5、batch 1、1008×1008，`--save_name best3.pth`，按 val MAE 选点。

## 冒烟与基准（随机初始化，未含最终权重）

- 前向：1008² 输入 → `image_embed (1,256,72,72)`、`high_res [(1,32,288,288),(1,64,144,144)]`，
  两个 decoder 输出 `(1,1,288,288)`。
- 训练步（fp32，与原始 `train.py` 一致）：约 0.50 s/step，峰值显存 7.9 GiB；损失有限。
  1,500 步 ≈ 12.5 min/epoch，加 val 约 16 min/epoch。

## 结果

训练 30 epoch、batch 1、1008×1008、lr 5e-5，按 val MAE 选点。val MAE 在第 6 epoch 达到最低
0.000166，之后在 0.00047–0.053 间波动且再未改善，因此最终权重是**第 6 epoch** 的 `best3.pth`
（2.2 GB）。这是原始代码自带的选点规则，未按 test 分数改选。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ | IoU↑ | F1↑ | Pd↑ | Fa↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始版，Seal train 框监督，val 最优（ep6），test 1,983 张 | 0.49828 | 0.38774 | 0.57970 | 0.00027912 | 0.24975 | 0.39968 | 0.58838 | 9.7774e-05 |

完整精度：S 0.49827792707208135、Fw 0.38774462403247995、E 0.5796989130894807、
MAE 0.00027911554207094014、IoU 0.24974667188513647、F1 0.39967567428431694、
Pd 0.5883787078423405、Fa 9.777410424472832e-05；像素 TP/FP/FN = 46,582 / 80,805 / 59,130
（TP+FN = 105,712 = GT 前景像素，与全项目同一批 GT）。

横向对照（12 个全监督 IRSTD 基线）：IoU 0.2497 落在基线区间（0.1917–0.3074）中部；
Fw 0.3877 低于 DNA-Net/MSHNet/RPCANet/HDNet/RPCANet++/BGM/L2SKNet（0.499–0.541）；
E 0.5797 明显低于全部基线（0.6641–0.6860）；MAE 0.000279 高于 8 个基线、也高于全背景平凡基线
0.00020553；Pd 0.5884 只高于 ILNet；Fa 9.78e-05 高于 10 个基线。即本方法在 Seal 上
**定位尚可、区域与误检指标偏弱**。

训练后期 val MAE 反复回升（0.0005–0.053）而训练 loss 仍在下降，说明在 1,500 张上很快过拟合；
表中严格保留 val 最优检查点的真实数值，不为了好看改选后期 epoch。

## 产物

- 权重：`results/Ours/Original/best3.pth`；日志 `logs/ours_original_train.log`
- 预测：`results/Ours/Original/infer/best3/test/`（1,983 张）
- 指标：`results/Ours/Original/metrics4.json`、`metrics_small.json`
