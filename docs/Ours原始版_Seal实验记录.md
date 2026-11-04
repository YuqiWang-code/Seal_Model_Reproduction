# Ours（原始版，RGB-T + 框提示双解码器）：Seal 训练与测试记录

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

权重：`Ours/weights/sam3.pt`（3,450,062,241 字节，SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`，已与本地逐字节校验）、`Ours/weights/resnet50-19c8e357.pth`。

训练设置：30 epoch、lr 5e-5、batch 1、1008×1008，`--save_name best3.pth`，按 val MAE 选点。

## 冒烟与基准（随机初始化，未含最终权重）

- 前向：1008² 输入 → `image_embed (1,256,72,72)`、`high_res [(1,32,288,288),(1,64,144,144)]`，
  两个 decoder 输出 `(1,1,288,288)`。
- 训练步（fp32，与原始 `train.py` 一致）：约 0.50 s/step，峰值显存 7.9 GiB；损失有限。
  1,500 步 ≈ 12.5 min/epoch，加 val 约 16 min/epoch。

## 结果

<!--RESULTS-->

## 产物

- 权重：`results/Ours/Original/best3.pth`；日志 `logs/ours_original_train.log`
- 预测：`results/Ours/Original/infer/best3/test/`（1,983 张）
- 指标：`results/Ours/Original/metrics4.json`、`metrics_small.json`
