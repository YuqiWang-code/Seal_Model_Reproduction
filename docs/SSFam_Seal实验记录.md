# SSFam：Seal 实验记录

## 跨域主表协议

主表采用作者公开 RGBT 完整权重在 Seal 上的跨域测试。Seal 同时提供 RGB 与热红外图像，因而采用作者的 RGBT 变体；模型选择与 Seal 测试分数无关。Seal 框监督适配仅作历史实验，不再执行或恢复，不使用 Seal 训练集或验证集选择主表权重。

本实验的四项指标使用 [Ours/evaluate.py](../Ours/evaluate.py) 的 `s_measure_folder`、`weighted_f_measure_folder`、`e_measure_folder`、`mae_folder` 原函数。服务器根目录 `evaluate.py` 的四个函数与该文件逐函数一致；`E` 为原函数对连续灰度预测的 E-measure。

## 方法与权重

- 论文：`SSFam-main/SSFam_Scribble_Supervised_Salient_Object_Detection_Family.pdf`。SSFam 将涂鸦监督的显著目标检测扩展到 RGB、RGB-D、RGB-T 和 V-D-T；测试时不输入涂鸦提示。
- 作者发布的 RGBT 完整权重为 `SSFam-main/模型参数/RGBT/loraH9_Scribble0.023440429411862376best_6.pth`，服务器副本为 `SSFam-main/checkpoint_rgbt.pth`，对应 SAM ViT-L 与双模态编码。推理入口严格完整加载全部参数，作者权重可直接用于 Seal 跨域测试，无需源域重训。
- 输入为 1024×1024。作者测试脚本对模型的无提示分支输出做双线性插值、sigmoid 和逐图 min-max 归一化；`SSFam-main/infer_seal.py` 保持该输出处理，并按 GT 文件名配对、保存灰度 PNG。
- Seal 的 `T` 为 16 位 TIFF。推理前按每张热图的最小值和最大值线性映射到 8 位，再复制为三通道；RGB 使用 ImageNet 均值和标准差归一化，热图不做该归一化。

## 公开权重跨域测试

- 数据：`Seal_data/test/{RGB,T,GT}`，1,983 对；不使用 Seal 的训练集、验证集或框标注。
- 服务器：RSML-3，物理 GPU 0；Python 环境 `/home/yqwang/miniforge3/envs/fct_sam_seal/`。
- 预测目录：`/share_datasets/Seal_Model_Reproduction/results/SSFam/Seal/RGBT_published/`，1,983 张。
- 1,983 张预测与 GT 文件名一一对应，尺寸均与 GT 相同，均为灰度 PNG。作者 RGBT 网络的无提示分支和 ViT-L 权重严格完整加载；RGB 与 T 按文件名配对。
- 固定原函数评估产物：`SSFam-main/experiments/Published_fixed_eval/metrics.json`。已有预测完整，直接复算，无需重新推理。
- 推理命令：

```bash
cd /share_datasets/Seal_Model_Reproduction
mkdir -p SSFam-main/source_data/test
unzip -q SSFam-main/source_test.zip 'Test/VT821/*' -d SSFam-main/source_data
ln -s ../Test/VT821/RGB SSFam-main/source_data/test/RGB
ln -s ../Test/VT821/T SSFam-main/source_data/test/T
ln -s ../Test/VT821/GT_PNG SSFam-main/source_data/test/GT
CUDA_VISIBLE_DEVICES=0 /home/yqwang/miniforge3/envs/fct_sam_seal/bin/python SSFam-main/infer_seal.py \
  --modality RGBT --checkpoint SSFam-main/checkpoint_rgbt.pth \
  --seal-dir Seal_data --output-dir results/SSFam/Seal/RGBT_published
```

```bash
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate.py \
  --pred_root results/SSFam/Seal/RGBT_published \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT
```

| 权重 | S↑ | Fw↑ | E↑ | MAE↓ | IoU↑ | F1↑ | Pd↑ | Fa↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SSFam RGBT 公开权重，跨域测试 | 0.46495 | 0.00562 | 0.27369 | 0.04515 | 0.00099438 | 0.00198679 | 0.28741704 | 0.0402274131 |

新增四项使用根目录 `metrics.py` 的 `AS1000Metrics` 与 `evaluate_small_targets.py`，对同一批 1,983 张预测计算；阈值 >0.5，8 连通，质心距离严格小于 3 像素一对一匹配，Fa 为原始比例。完整精度、计数及协议见[SSFam.json](小目标指标评估_20261103/SSFam.json)。

此前同一预测目录以旧口径记录为 S 0.47151、Fw 0.00570、E 0.29624、MAE 0.04515；该组数字保留为历史记录，不用于最终比较。

## 源域 VT821 核查

作者 RGBT 测试集归档 `data/SSFam/数据集/RGBT/Test.zip` 包含 VT821 的 RGB、T、GT 与 GT_PNG。服务器副本为 `SSFam-main/source_test.zip`；仅解压 VT821 测试划分。使用同一公开 RGBT 完整权重及 `infer_seal.py` 的无提示推理流程，在 `SSFam-main/source_data/Test/VT821/{RGB,T,GT_PNG}` 上生成 821 张预测。该划分只用于核查推理实现，不参与训练或选择权重。源域热图为 JPEG RGB 图像，无需 Seal 16 位 TIFF 映射。

预测保存在 `SSFam-main/experiments/Published_source_VT821/maps/`，与 821 张 GT_PNG 文件名一一对应。使用同一四个固定原函数，得到 S 0.918617828569752、Fw 0.8804909459002501、E 0.9385882353399647、MAE 0.02283204346895218；完整结果在 `SSFam-main/experiments/Published_source_VT821/metrics.json`。这些结果支持权重加载及 RGB/T 推理链路正常。此处的评估函数与论文表格口径可能不同，不直接比较数值。

```bash
cd /share_datasets/Seal_Model_Reproduction
CUDA_VISIBLE_DEVICES=0 /home/yqwang/miniforge3/envs/fct_sam_seal/bin/python SSFam-main/infer_seal.py \
  --modality RGBT --checkpoint SSFam-main/checkpoint_rgbt.pth \
  --seal-dir SSFam-main/source_data \
  --output-dir SSFam-main/experiments/Published_source_VT821/maps
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate.py \
  --pred_root SSFam-main/experiments/Published_source_VT821/maps \
  --gt_dir SSFam-main/source_data/Test/VT821/GT_PNG \
  --json_output SSFam-main/experiments/Published_source_VT821/metrics.json
```

## 历史 Seal 框监督适配

`SSFam-main/train_seal.py` 从上述 RGBT 完整权重初始化，只读取 `Seal_data/train/{RGB,T,box}`。每个框中心作为一个前景点，在框外取五个背景点；这些稀疏点同时提供 SAM 训练提示和点监督。无提示分支通过作者的双解码器结构学习有提示分支的预测。训练保留作者的结构、平滑和局部一致性损失，对前景点与背景点的交叉熵取等权平均。该协议属于 Seal 框监督适配，不等同于论文的原始涂鸦监督。

输入 1024×1024、batch size 1、Adam 学习率 5e-5、30 epoch；冻结 SAM 图像编码器和提示编码器，训练 LoRA 与解码器。每 5 epoch 使用完整 500 张 `Seal_data/val/{RGB,T,GT}` 的原函数 Fw 选择最佳权重；`test/GT` 只用于最终评估。训练、验证和测试均使用相同的 16 位 TIFF 单图 min-max 转 8 位处理。每个划分抽查的热图都映射到 0–255，保留 64～152 个灰度级。

### 既有训练产物

`SSFam-main/experiments/Seal_box/` 的训练在 2026-11-03 09:39 启动，核查时完成第 24 epoch。验证日志记录第 5、10、15、20 epoch 的 Fw 分别为 0.36279、0.43263、0.43973、0.53834。目录只保存第 20 epoch 的 `best.pth`、`best.json` 和被覆盖的 500 张 `val_maps`；其余轮次没有独立权重或预测。

训练代码每轮验证通过子进程读取服务器根目录 `evaluate.py`。该文件于 12:12 在训练期间更新，现有日志与产物不能逐轮确认第 5、10、15 epoch 的评估源码；第 20 epoch 的验证预测写于 12:46 后。缺少完整候选，不能将旧 `best.pth` 作为固定原函数口径下的全程最佳。该训练及其收尾进程已停止，未运行最终测试。

### 固定原函数训练产物

独立目录 `SSFam-main/experiments/Seal_box_original_eval/` 保存了 `config.json`、固定的 `evaluate_source.py`、第 5、10、15 epoch 的独立权重及各 500 张验证预测。原函数验证 Fw 分别为 0.3620349794300652、0.42396283859104894、0.40138596463856885；`validation.jsonl` 和 `selection.json` 记录第 10 epoch 为已保存候选中的最佳。训练于第 17 epoch 开始后停止，没有生成测试预测或测试指标。对应进程及子进程均已结束，自动跟进已暂停；这些产物不进入跨域主表。

## 已知问题

Seal 热图的 16 位到 8 位映射改变了作者测试代码的热图读取方式。历史框监督实验的框中心点未必落在实际前景上，其稀疏点监督也与论文的涂鸦监督不同。

