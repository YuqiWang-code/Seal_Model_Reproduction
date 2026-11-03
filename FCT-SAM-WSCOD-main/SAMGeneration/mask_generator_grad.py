import random

from scipy.stats import t, norm, chi2

import numpy as np
import os
import glob
from PIL import Image
import cv2
import torch

from segment_anything import SamPredictor, sam_model_registry
sam_checkpoint = "E:\Mr.Wu\codes\Weakly-Supervised-Camouflaged-Transformer\pretrained\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = 'cuda'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
np.random.seed(42)
def get_mask(image, fig_mask):

    # get instance mask
    predictor.set_image(image)

    # 获取前景点坐标
    fg_row_indices, fg_col_indices = np.where(fig_mask[:, :, 0] == 1)
    fg_coordinates = list(zip(fg_col_indices, fg_row_indices))
    np.random.shuffle(fg_coordinates)  # 随机打乱前景点坐标
    fg_coordinates = np.array(fg_coordinates)
    num_fg_points = min(fg_coordinates.shape[0], 100)  # 选择前景点的数量，最多选择10个
    fg_coordinates_label = np.ones(num_fg_points)

    # 获取背景点坐标
    bg_row_indices, bg_col_indices = np.where(fig_mask[:, :, 0] == 2)
    bg_coordinates = list(zip(bg_col_indices, bg_row_indices))
    np.random.shuffle(bg_coordinates)  # 随机打乱背景点坐标
    bg_coordinates = np.array(bg_coordinates)
    rand_length = random.randint(0, len(bg_coordinates)//10)
    num_bg_points = min(bg_coordinates.shape[0], rand_length)  # 选择背景点的数量，确保总共选择10个点
    bg_coordinates_label = np.zeros(num_bg_points)

    coordinates = np.concatenate((fg_coordinates[0:num_fg_points], bg_coordinates[0:num_bg_points]), axis=0)
    coordinates_labels = np.concatenate((fg_coordinates_label, bg_coordinates_label), axis=0)

    coordinate = coordinates[0:1, :]
    coordinates_label = coordinates_labels[0:1]
    mask, scores, logits = predictor.predict(
        point_coords=coordinate,
        point_labels=coordinates_label,
        multimask_output=True,
    )

    coordinate = coordinates[:, :]
    coordinates_label = coordinates_labels[:]
    mask_input = logits[np.argmin(scores), :, :]
    mask, scores, logits = predictor.predict(
        point_coords=coordinate,
        point_labels=coordinates_label,
        mask_input=mask_input[None, :, :],
        multimask_output=True,
    )
    fg_mask = mask[1].astype(int)
    fg_mask = np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2)

    # fuse background
    fig_mask[fig_mask == 1] = 0
    mask = fg_mask + fig_mask
    mask[mask == 3] = 2
    return mask

def get_random_mask(image_files, size):
    sam_image = cv2.imread(image_files)
    mask_path = image_files.replace('Image', 'oldScribble').replace('jpg', 'png')
    mask = cv2.imread(mask_path).astype(np.float32)[:, :, ::-1]

    H, W, C = sam_image.shape
    sam_image = cv2.resize(sam_image, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    # get different image and mask
    # try:
    # flip
    flipped_image = np.flip(sam_image, axis=1)
    flipped_mask = np.flip(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST), axis=1)
    flipped_mask = get_mask(flipped_image, flipped_mask)
    flipped_mask = np.flip(flipped_mask, axis=1)
    flipped_mask = cv2.resize(flipped_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # rotation
    angle = 90
    rotation_matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    rotated_image = cv2.warpAffine(sam_image, rotation_matrix, (size, size))
    rotated_mask = cv2.warpAffine(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST), rotation_matrix,
                                  (size, size))
    rotated_mask = get_mask(rotated_image, rotated_mask)
    return_angle = -90
    rotation_matrix = cv2.getRotationMatrix2D((size / 2, size / 2), return_angle, 1)
    rotated_mask = cv2.warpAffine(rotated_mask, rotation_matrix, (size, size))
    rotated_mask = cv2.resize(rotated_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # shrink
    normal_sam_image = cv2.resize(sam_image, (size, size), interpolation=cv2.INTER_NEAREST)
    normal_mask = cv2.resize(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST), (size, size),
                             interpolation=cv2.INTER_LINEAR)
    normal_mask = get_mask(normal_sam_image, normal_mask)
    normal_mask = cv2.resize(normal_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # fusion different mask

    final_mask = (flipped_mask + rotated_mask + normal_mask)
    final_mask[final_mask == 6] = 0


    return final_mask, mask_path

def convolve2d(matrix):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    km, kn = kernel.shape

    # 计算填充后的矩阵大小
    padded_matrix = np.pad(matrix, 1, mode='constant')
    pm, pn = padded_matrix.shape

    # 计算卷积后的矩阵大小
    result_m, result_n = pm - km + 1, pn - kn + 1

    # 创建一个空的结果矩阵
    result = np.zeros((result_m, result_n))

    # 执行卷积操作
    for i in range(result_m):
        for j in range(result_n):
            if padded_matrix[i + km - 1, j + kn - 1] == 0:
                result[i, j] = np.nan
                continue
            result[i, j] = np.sum(padded_matrix[i:i + km, j:j + kn] * kernel)

    mean = np.nanmean(result)
    std_dev = np.nanstd(result)
    # 置信水平
    nan_count = np.sum(np.isnan(result))
    num_count = result_m * result_n - nan_count

    confidence_level = 0.99
    alpha = 1 - confidence_level
    critical = t.ppf(1 - alpha / 2, num_count - 1)

    SE = std_dev / np.sqrt(num_count)
    lower_limit = mean - SE * critical
    upper_limit = mean + SE * critical

    result[(result < upper_limit) & (result > lower_limit)] = 0
    result[(result >= upper_limit) | (result <= lower_limit) | (result == np.nan)] = 255
    result = 255 - result
    result = np.tile(result[:, :, np.newaxis], (1, 1, 3))
    return result

if __name__=='__main__':
    # read image
    path = r'E:\Mr.Wu\dataset\CodDataset\train\Image'
    image_files = glob.glob(os.path.join(path, "*.jpg"))
    for image_files in image_files:
        size = 512
        i = 0
        epoch = 10
        try:
            while i < epoch:
                if i == 0:
                    mask, mask_path = get_random_mask(image_files, size)
                    final_mask = mask
                elif i == epoch-1:
                    mask, _ = get_random_mask(image_files, size)
                    final_mask = mask + final_mask
                else:
                    mask, _ = get_random_mask(image_files, size)
                    final_mask = mask + final_mask
                i += 1
        except:
            print(image_files, r'wrong')

        final_mask = final_mask.astype(np.uint8)
        final_mask = final_mask[:, :, 0]
        final_mask = convolve2d(final_mask)
        maskpath_save = mask_path.replace('oldScribble', 'sam_fg100_bgran_grad_epoch10')
        final_mask = final_mask.astype(np.uint8)
        mask_image = Image.fromarray(final_mask)
        mask_image.save(maskpath_save)
        print(image_files)

        # except:
        #     print(mask_path)