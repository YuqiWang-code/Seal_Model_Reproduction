import os
import numpy as np
from PIL import Image
import utils.metrics as Measure
# 定义文件夹路径
mask_folder = r'E:\Mr.Wu\dataset\CodDataset\train\sam_fg100_bgran_epoch10_0.66'  # 替换为你的路径
label_folder = r'E:\Mr.Wu\dataset\CodDataset\train\GT'  # 替换为你的路径

# 获取文件夹中的所有文件名
mask_files = os.listdir(mask_folder)
label_files = os.listdir(label_folder)

# 确保文件名一一对应
mask_files.sort()
label_files.sort()
WFM = Measure.WeightedFmeasure()
SM = Measure.Smeasure()
EM = Measure.Emeasure()
MAE = Measure.MAE()
# 遍历文件列表并计算每对掩码和标签的 IoU
for mask_file, label_file in zip(mask_files, label_files):
    # 加载掩码和标签图像
    mask_image = Image.open(os.path.join(mask_folder, mask_file))
    label_image = Image.open(os.path.join(label_folder, label_file))

    # 将图像转换为 NumPy 数组
    mask_array = np.array(mask_image)
    label_array = np.array(label_image)
    mask_len = len(mask_array.shape)
    if mask_len == 3:
        mask_array = mask_array[:, :, 0]
    label_len = len(label_array.shape)
    if label_len == 3:
        label_array = label_array[:, :, 0]
    WFM.step(pred=mask_array, gt=label_array)
    SM.step(pred=mask_array, gt=label_array)
    EM.step(pred=mask_array, gt=label_array)
    MAE.step(pred=mask_array, gt=label_array)
sm = SM.get_results()['sm'].round(3)
adpem = EM.get_results()['em']['adp'].round(3)
wfm = WFM.get_results()['wfm'].round(3)
mae = MAE.get_results()['mae'].round(3)
print('MAE: {},  wfm: {}, adpem: {}, sm: {}'.format(mae, wfm, adpem, sm))

