import json
from pathlib import Path

b=Path('docs/NESS-Net_Seal评估产物')
p=json.loads((b/'published_metrics.json').read_text())['refinement_namlab']
s=json.loads((b/'refinement_namlab_small_target_metrics.json').read_text())
assert p['predictions']==s['prediction_dir'] and p['count']==s['counts']['n_images']==1983
metrics={key:p[key] for key in ('S','F','E','MAE')}
metrics['Fw']=metrics.pop('F')
metrics.update(s['metrics'])
record={'protocol':'published refinement + NAMLab; RGB-D model with Seal thermal replacing depth',
 'checkpoint':'/share_datasets/Seal_Model_Reproduction/NESS-Net-main/NESS_refinement.pth',
 'stage':'refinement','supervision':'source-domain scribble',
 'prediction_dir':s['prediction_dir'],'gt_dir':s['gt_dir'],'count':1983,
 'original_metrics_source':'published_metrics.json:refinement_namlab',
 'small_target_metrics_source':'refinement_namlab_small_target_metrics.json',
 'metrics':{key:metrics[key] for key in ('S','Fw','E','MAE','IoU','F1','Pd','Fa')}}
(b/'cross_domain_metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
path=Path('docs/NESS-Net_Seal实验记录.md')
old=path.read_text(encoding='utf-8')
public=old[old.index('## 公开权重迁移测试'):old.index('## Seal 框监督、Canny 边缘、general 阶段适配')]
history=old[old.index('## Seal 框监督、Canny 边缘、general 阶段适配'):]
history=history.replace('## Seal 框监督、Canny 边缘、general 阶段适配','## 历史 Seal 框监督适配（不用于跨域主表）',1)
history=history.replace('## 小目标评价指标','### 历史框适配小目标指标',1)
history=history.replace('- 服务器 NESS-Net 目录及项目内未发现可用的 NJU2K、NLPR 或 LFSD 源域测试集，本记录不提供源域性能核对。','- 现有作者公开完整权重可用；本记录尚未提供源域测试集性能核对。')
rows='\n'.join(f'| {key} | {record["metrics"][key]} |' for key in record['metrics'])
header=f'''# NESS-Net 在 Seal 上的实验记录

## 当前跨域主表协议

主表采用作者公开完整权重 **`NESS_refinement.pth` + NAMLab 边缘 + refinement 分支**，属于源域涂鸦监督 RGB-D 模型以 Seal 热成像替代深度的跨模态迁移测试。权重与边缘流程预先固定，不使用 Seal train/val 训练、选择权重、阈值或调参。

NESS-Net 的 Seal 适配训练、标签生成和自动跟进均停止执行。服务器未发现运行或等待启动的相关进程、crontab、at 或用户定时器，本地未发现 NESS-Net Codex 自动化或 Windows 计划任务。旧 `select_seal_box_after_train.sh` 已停用。历史 35 epoch 框适配及 epoch5 的全部结果保留在本文历史章节。

| 主表产物 | 路径 |
|---|---|
| 作者公开完整权重 | `{record['checkpoint']}` |
| 八项指标共同预测 | `{record['prediction_dir']}/`（1,983 张） |
| 测试 GT | `{record['gt_dir']}/` |
| 原四项完整精度 | [published_metrics.json](NESS-Net_Seal评估产物/published_metrics.json) 的 `refinement_namlab` |
| 小目标计数和协议 | [refinement_namlab_small_target_metrics.json](NESS-Net_Seal评估产物/refinement_namlab_small_target_metrics.json) |
| 统一八项与协议 | [cross_domain_metrics.json](NESS-Net_Seal评估产物/cross_domain_metrics.json) |

## 跨域主表八项结果

| 指标 | 原始比例 |
|---|---:|
{rows}

原四项从完整精度 JSON 读取；新增四项直接评估同一批公开 refinement + NAMLab 预测，没有使用 Seal 框适配的预测或指标。Fa 保存原始比例。主表 NESS-Net 仅一行，结果区仅第 1～20 行。

小目标评估共 1,983 张、514,338,642 像素，TP/FP/FN 为 12,554 / 17,013,524 / 93,158；GT 组件 7,383，预测组件 8,410，成功匹配 38，未匹配预测组件面积 16,949,644 像素。评估入口核对了全部 PNG 文件配对与原尺寸一致性。

## 固定评估与输入

S、Fw、E、MAE 使用 `Ours/evaluate.py` 的四个原函数，服务器根目录四函数原文与其一致，未修改内部实现。E 对连续灰度预测计算。新增指标使用根目录 `metrics.py` 的 `AS1000Metrics` 和 `evaluate_small_targets.py`：预测 >0.5，GT >128，8 连通，质心距离严格小于 3 像素的栅格顺序一对一匹配。IoU/F1 为全数据集像素累计，Pd 为检出组件/GT 组件，Fa 为未匹配预测组件面积/总像素数。

16 位热成像按每图 `lo=min(T)`、`hi=max(T)` 映射为 `round((T-lo)*255/(hi-lo))` 的 uint8，再复制三通道；恒定热图映射为零。RGB 使用 ImageNet 标准化，热图仅 ToTensor，输入均为 384×384。NAMLab 对原尺寸 RGB 与转换后的原尺寸热图生成边缘，读取后缩放至 384×384。推理保留作者 `sal_ref` 输出分支、双线性恢复尺寸、sigmoid、逐图 min-max 和 round×255 保存流程。

服务器项目根目录为 `/share_datasets/Seal_Model_Reproduction/`，Python 为 `/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python`。新增结果位于 `results/NESS-Net/Seal/original_eval/refinement_namlab_small_target_metrics.json`，日志为同目录 `refinement_namlab_small_target_evaluate.log`。原公开预测完整且来源明确，直接复用；没有重新训练或推理。

```bash
cd /share_datasets/Seal_Model_Reproduction
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate_small_targets.py \\
  --pred_root results/NESS-Net/Seal/refinement_namlab_test_maps \\
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \\
  --json_output results/NESS-Net/Seal/original_eval/refinement_namlab_small_target_metrics.json
```

## 源域数据归档

作者公开完整权重在服务器可用，采用该权重完成跨域测试。本地 `data/NESS-Net/train_data.zip` 包含 2,185 个文件各自对应的 `img`、`depth`、`gray`、`gt`、`gt_mask`、`mask`、`gt_refinement`、`rgb-lab`、`depth-lab` 九组目录；`test_data.zip` 的五组 `img/depth/gt/rgb-lab/depth-lab` 每组 4,579 个文件，按 STERE、ReDWeb-S、SIP、NJU2K_Test、DUTLF-Depth、NLPR_Test、DES、RGBD135、LFSD、SSD 分目录保存。服务器未发现这些归档或已解压源数据。源域训练仅在公开完整权重不可用时开展，使用原有非 Seal 数据及监督形式。

'''
path.write_text(header+public+history,encoding='utf-8')
