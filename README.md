<!-- 本文件是 AGENTS.md 的镜像，便于在 GitHub 首页展示；协议与数值以 AGENTS.md 为准。 -->
> 本文件与 [AGENTS.md](AGENTS.md) 内容一致，仅为 GitHub 首页展示而生成。

# Seal Model Reproduction

本项目评估对比方法在 Seal 数据集上的跨域泛化，汇总 `S↑ / Fw↑ / E↑ / MAE↓ / IoU↑ / F1↑ / Pd↑ / Fa↓`。对比方法清单见[腾讯文档](https://docs.qq.com/sheet/DRVFTdFptTkJ4SmxY?tab=BB08J2)；每个方法的训练与测试策略遵循其论文及代码。

## 跨域泛化协议

- 对比方法有可用的作者完整训练权重时，直接在 Seal test 推理与评估。跨域分数低不构成在 Seal 上重训的理由。
- 无可用完整权重时，使用该方法论文或官方代码支持的非 Seal 源数据集及原监督形式训练，再在 Seal test 评估。已有符合该协议的源域训练权重可以复用。主干预训练权重不等同于完整方法权重。
- 不使用 Seal train/val 的图像、GT、box、点提示或伪标签训练、适配、选权重、选阈值或调参。Seal test GT 仅用于最终指标计算；恢复预测原尺寸可以读取尺寸信息。
- 权重采用作者发布的主方法权重，或按源域协议预先确定的训练终点/源域验证规则选择，不依据 Seal 分数更换检查点或后处理。源域测试集仅用于核对实现，不当作训练集。
- 本项目五个对比方法的 Seal 框监督、点监督及 SAM 伪标签适配任务停止执行，历史产物仅在实验文档中存档，不进入跨域主表；相关自动跟进不得重启这些适配任务。
- 本地 `data/TrainDataset.zip`、`data/TestDataset.zip` 为 S-COD 相关训练与测试归档，`data/Scribble-COD.zip` 提供独立涂鸦标注；须按实际内容区分完整 GT 与 0/1/2 稀疏标签。`data/NESS-Net/`、`data/SSFam/` 另有各自数据归档。优先复用服务器已解压源数据，缺失时再上传对应归档。

最终汇总表为 `docs/Seal量化SOTA表.xlsx`：第 3～20 行为弱监督/跨域对比方法，每个方法只填一组指标；第 21 行之后只放全监督 IRSTD 基线（见下一节），每个基线一行，不增加适配、消融或其他补充结果。各方法的多组实验和历史结果记录在对应实验文档中。表内采用的实验协议须明确，不能按测试集分数择优拼选。

该表来自腾讯文档，可能新增列或更新协议。合并下载版本时按方法及列标题对应，采用新版本的布局和明确更新；新版本空白处保留现有有效评估结果，同协议的五位小数保留完整精度。新增 IoU、F1、Pd、Fa 四列后，每个方法仍只保留一行；填写时按标题定位列，不依赖旧列号。

## 数据和评估

- 服务器：`RSML-3`，用户 `yqwang`。项目根目录：`/share_datasets/Seal_Model_Reproduction/`。
- 数据：`Seal_data/{train,val,test}/{RGB,T,GT,box}`。使用原有划分；测试集有 1,983 对 RGB 与 GT。
- 四项指标的唯一计算依据是 `Ours/evaluate.py` 中的 `s_measure_folder`、`weighted_f_measure_folder`、`e_measure_folder`、`mae_folder`。这四个函数及其内部实现禁止修改，所有方法必须使用原函数计算，不得更换公式、数值常量、平滑核、归一化或聚合方式。
- 统一运行入口是项目根目录的 `evaluate.py`，其四个指标函数须与 `Ours/evaluate.py` 原文一致；仅入口参数允许适配路径。预测图放在单独的结果目录，文件名与 `Seal_data/test/GT` 一一对应。服务器运行时显式指定 `--gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT`。
- `E` 使用上述原始 `e_measure_folder` 对连续灰度预测计算，不使用 256 阈值 mean E-measure 替代。已有其他评估口径的结果须明确标记，按指定四个原函数重算后才能作为最终比较结果。
- 新增小目标指标由根目录 `metrics.py` 的 `AS1000Metrics` 与 `evaluate_small_targets.py` 统一计算。固定预测阈值 >0.5、8 连通、质心距离严格小于 3 像素的一对一匹配；IoU/F1 采用全数据集累计像素计数，Pd 为检出目标数/GT目标数，Fa 为未匹配预测连通域面积/总像素数。表中均存 0～1 原始比例，不将 Fa 乘 10^6 后填入未标单位的列。定义、文献与官方代码对照见 [docs/小目标评价指标_定义与代码对照.md](docs/小目标评价指标_定义与代码对照.md)。同行八项指标必须来自同一模型和预测，原有四个指标函数保持不变。
- `docs/小目标评价指标.pdf` 是早期草稿，使用 `pred > 0`、像素假阳 Fa、`距离 <= 3` 与 ×100 百分数，与现行实现不同；差异及数值影响（Ours 的 Fa 相差 31.8 倍）见定义对照文档，两种口径不得混用。`docs/小目标指标评估_20261103/` 的 NESS-Net 行必须是公开 refinement + NAMLab 跨域结果，框适配第 5 epoch 的结果仅存档为 `NESS-Net_Seal框适配_epoch5.json`。
- 跨域指训练源数据与 Seal 目标数据不同，监督形式差异另行注明，不能以 Seal 框适配或 GT 抽点训练代替源域训练。
- LRDNet 的公开权重链接失效，主表采用已完成的原始 S-COD 涂鸦监督训练：4,040 张 S-COD/CAMO 训练图像及公开涂鸦标注，两阶段均取预先确定的第 100 epoch 权重；不使用 Seal 数据选权重。Seal 框适配仅作历史实验存档。结果记录在 [docs/LRDNet_Seal实验记录.md](docs/LRDNet_Seal实验记录.md)。
- 不根据 Seal 测试集指标选择权重。先依据论文配置、代码结构和检查点参数确定主方法权重，再运行完整测试集。
- 所有指标必须如实记录，不能根据 Seal 测试结果挑选、修改或隐去权重及预测。跨域主表不采用 Seal 适配结果。Ours 的 `Ours/test_maps.zip` 含 1,983 张测试预测，使用指定四个原函数评估：S 0.49915、Fw 0.67338、E 0.66731、MAE 0.00015；预测与评估记录见 [docs/Ours_Seal实验记录.md](docs/Ours_Seal实验记录.md)。Ours 的训练来源须单独注明，不能仅凭测试预测认定其属于跨域训练。腾讯表格原有 Ours 数值保留来源记录。

## 全监督 IRSTD 基线（Seal 训练 → Seal 测试）

- 定位：与上面的弱监督跨域表并列的全监督对照。方法为 DNA-Net（TIP22）、MSHNet（CVPR24）、UIU-Net（TIP23），均取自 [awesome-infrared-small-targets](https://github.com/Tianfang-Zhang/awesome-infrared-small-targets) 的 Single-Frame 全监督条目（对照 GrokCV 出版物页），代码为作者官方实现，根目录各一个文件夹：`DNANet-main/`、`MSHNet-main/`、`UIU-Net-main/`。ACM（WACV21）也已完成代码与训练入口（`ACM-main/`），但官方 SoftIoU 在 Seal 的 0.02% 前景上不收敛（val IoU 恒 0，两个随机种子、官方 warm-up 均复现），作为负结果存档，不进入主表，见 [ACM 记录](docs/ACM_Seal实验记录.md)。
- 数据：服务器 `Seal_IRSTD/`。`Seal_data/{split}/T/*.tif`（16 位热红外）逐图 **min-max** 拉伸到 8 位并复制为 3 通道 PNG，`GT` 二值化；`trainval.txt` 1,500 / `val.txt` 500 / `test.txt` 1,983，沿用 Seal 原划分。生成脚本 [tools/prepare_seal_irstd.py](tools/prepare_seal_irstd.py)，指标合并脚本 [tools/merge_seal_metrics.py](tools/merge_seal_metrics.py)，单方法评估脚本 [tools/eval_fullsup_method.sh](tools/eval_fullsup_method.sh)。
- 训练：Seal train 全像素 GT，从零初始化，crop 256×256，batch 8。epoch 与优化器沿用官方默认：DNA-Net 400（官方 1,500）/Adagrad 0.05/cosine/深监督 SoftIoU；MSHNet 400/Adagrad 0.05/SLSIoU；UIU-Net 500/Adam 1e-3/七输出 BCE。不使用 Seal 的 `box`、点提示或伪标签。
- 选择与推理：模型选择只用 Seal **val**（500 张，滑窗像素 IoU@0.5），test 不参与选择、不据此换检查点；测试为原生分辨率 256×256 滑窗、步长 128、重叠区 logits 平均后 sigmoid，输出与 test GT 同名的 8 位 PNG。
- 指标：同行八项仍由根目录 `evaluate.py` 的四个原函数与 `evaluate_small_targets.py` 计算，同一检查点、同一批 1,983 张预测，口径与第 3～20 行完全一致。
- 结果（全部为 test 1,983 张、val 最优检查点）：

  | 方法 | val 最优 epoch | S | Fw | E | MAE | IoU | F1 | Pd | Fa |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | DNA-Net | 120 | 0.49803 | 0.50562 | 0.67979 | 0.00028725 | 0.30000 | 0.46154 | 0.79046 | 8.7299e-05 |
  | MSHNet | 140 | 0.49818 | 0.54104 | 0.68601 | 0.00024695 | 0.29226 | 0.45232 | 0.80943 | 6.7277e-05 |
  | UIU-Net | 20 | 0.49876 | 0.27240 | 0.68115 | 0.00037482 | 0.21485 | 0.35371 | 0.68739 | 2.9648e-05 |

  UIU-Net 只监督 BCE，第 20 epoch 后滑向"全背景"不动点（val IoU 0），按 val 连续 5 次无改善停止，表中为其 val 最优检查点。
- 热红外 8 位转换结论：用 1–99 百分位裁剪再拉伸会放大约 10 倍背景噪声（目标 SNR 降到 1.5），纯区域损失退化为常数输出（val IoU 恒为 0）；改为逐图 min-max 后目标与背景差值是背景 std 的 2.2–29.7 倍，训练正常。后续不得再用百分位裁剪生成输入。
- 表中位置：第 21～23 行，Modal 记 `T`，Supervision 记 `Full GT`，重训/测试列写明「Seal train 全监督重训 → Seal test」。这些行与第 3～20 行的弱监督跨域结果不是同一协议，不能互相替代或混填。
- 实验记录：[ACM](docs/ACM_Seal实验记录.md)、[DNA-Net](docs/DNANet_Seal实验记录.md)、[MSHNet](docs/MSHNet_Seal实验记录.md)、[UIU-Net](docs/UIU-Net_Seal实验记录.md)。

## FCT-SAM-WSCOD

- 统一实验文档：[docs/FCT-SAM-WSCOD_Seal实验记录.md](docs/FCT-SAM-WSCOD_Seal实验记录.md)，包含直接测试与 Seal 重训的指标、配置、命令、产物路径和已知问题。
- 论文：`FCT-SAM-WSCOD-main/FAIA-413-FAIA250797.pdf`。主方法是涂鸦监督的 RGB 伪装目标检测，推理不使用 Seal 的 `T` 或 `box`。
- 使用独立的 `fct_sam_seal` Conda 环境。测试入口是 `FCT-SAM-WSCOD-main/infer_seal.py`；输入 384×384，恢复到 GT 原尺寸并保存灰度预测图。
- 当前测试权重：`FCT-SAM-WSCOD-main/pretrain/Pvtv2_TE_new_dataset_batch4_epoch60.pth`，使用 `decoder_baseline`。该文件的 batch size 4、60 epoch 命名与论文设置一致，参数键和形状完全匹配。现有发布材料没有明确标注论文最终权重的文件名，本实验记录为该检查点的 Seal 跨域测试。
- 每次实验在 `docs/` 记录权重、数据划分、推理和评估命令、预测图数量、四项指标。服务器环境及路径见 [docs/RSML-3_Seal环境与数据说明.md](docs/RSML-3_Seal环境与数据说明.md)。
- `run_seal.sh`、`run_box_sam_seal.sh` 及 Seal 标签生成属于历史适配实验入口，不用于跨域主表，不再执行。
- 固定公开检查点在 CAMO 250 张源域图上复核得到作者口径 S 0.84870636、adaptive F 0.82776561、mean E 0.90931096、MAE 0.05558098，支持当前推理链路正常。Seal 主表仍采用该检查点的既有跨域预测，八项指标来源记录见实验文档。
- Seal 点监督与框监督适配的伪标签诊断及产物保留在实验文档，不作为跨域结果。

## MIR-Net

- 作者公开完整权重为 `MIR-Net-main/published_checkpoint/31_wM_test1/model-best.pth`，主干为 PVTv2-B4，RGB 输入 384×384。Seal 跨域测试使用 `MIR-Net-main/run_published_seal.sh`，结果见 [docs/MIR-Net_Seal实验记录.md](docs/MIR-Net_Seal实验记录.md)。
- 主表采用公开完整权重的 Seal 跨域测试，复用 `results/MIR-Net/Seal/published_test_maps/` 计算八项指标。`run_finetune_seal.sh`、`run_box_sam_seal.sh`、`run_box_sam_seal_150.sh` 及 `experiments/{Seal,Seal_box_sam,Seal_box_sam_150}_original_eval/` 属于历史 Seal 适配，不再执行或恢复训练。

## SSFam

- 实验记录：[docs/SSFam_Seal实验记录.md](docs/SSFam_Seal实验记录.md)。Seal 的 RGB 与热红外输入使用作者发布的 RGBT 完整权重；固定原函数跨域测试结果为 S 0.46495、Fw 0.00562、E 0.27369、MAE 0.04515。
- Seal 热图是 16 位 TIFF，推理和训练按每张图的数值范围线性映射到 8 位；测试输入为 1024×1024，预测恢复至 GT 原尺寸并逐图归一化。入口为 `SSFam-main/infer_seal.py`。
- 主表采用公开 RGBT 完整权重的跨域结果及 `results/SSFam/Seal/RGBT_published/` 同批预测。`train_seal.py` 与 `experiments/Seal_box_original_eval/` 属于历史 Seal 框适配，停止训练及自动恢复，保留已有记录。

## NESS-Net

- 最终表 NESS-Net 行采用作者公开 `NESS_refinement.pth` + NAMLab 边缘的跨域测试，使用 `results/NESS-Net/Seal/refinement_namlab_test_maps/` 的同一批预测计算八项指标。不得用 Seal 框适配 epoch5 的指标或小目标结果填入该行。

- 统一实验记录：[docs/NESS-Net_Seal实验记录.md](docs/NESS-Net_Seal实验记录.md)。作者的 RGB-D 涂鸦监督模型使用 Swin-B 双分支与 NAMLab 边缘；Seal 的 16 位 `T` 按单图范围映射为 8 位三通道，作为第二模态输入。
- 跨域主方法为 `NESS-Net-main/NESS_refinement.pth` + NAMLab 边缘，属于源域涂鸦监督 RGB-D 模型以 Seal 热成像替代深度的跨模态迁移测试。S 0.4666185279426068、Fw 0.001733883532895621、E 0.263767112039326、MAE 0.042296525090932846；同批预测的 IoU 0.0007333271181027003、F1 0.001465579487023831、Pd 0.005146959230665041、Fa 0.032954249624511（原始比例）。统一来源见 `docs/NESS-Net_Seal评估产物/cross_domain_metrics.json`；general、Canny 对照与旧口径结果保留在实验记录中。
- 35 epoch Seal 框适配及第 5 epoch 的选择、测试记录保存在 `experiments/Seal_box/canny_selected_original_eval/` 和实验文档中，仅作历史适配结果，不用于跨域主表。
- `select_seal_box_after_train.sh` 已停用，不再等待训练或自动恢复 Seal 适配任务。

## 同步约定

大文件服务器路径见 [docs/大文件服务器归档清单.md](docs/大文件服务器归档清单.md)。Ours 可视化归档为服务器 `Ours/vis.zip`，本地仅保留小样例与实验记录。

本地维护代码和文档，服务器执行 GPU 推理及评估。大文件上传服务器后可删除本地副本。同步代码时保留服务器 `pretrain/` 和 `Seal_data/`，不重复传输大型数据与权重。服务器根目录 `evaluate.py` 的四个指标函数须与本地 `Ours/evaluate.py` 原文一致；入口路径适配同步本地与服务器，指标实现保持固定。
