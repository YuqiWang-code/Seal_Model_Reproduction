
import math
import random


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

import numpy as np


# 定义函数进行九宫格采样
def nine_grid_sampling(points):
    # 获取坐标范围
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # 计算九宫格尺寸
    grid_width = (x_max - x_min) / 3
    grid_height = (y_max - y_min) / 3

    # 选取每个格子的中心点作为采样点
    sample_points = []
    for i in range(0, 3):
        for j in range(0, 3):
            grid_x_min = x_min + j * grid_width
            grid_x_max = x_min + (j + 1) * grid_width
            grid_y_min = y_min + i * grid_height
            grid_y_max = y_min + (i + 1) * grid_height
            count = 0
            while True:
                random_point = (np.random.randint(grid_x_min, grid_x_max + 1), np.random.randint(grid_y_min, grid_y_max + 1))
                count += 1
                if count > grid_height * grid_width:
                    break
                if random_point in points:
                    sample_points.append(random_point)
                    break

    return sample_points


def mask_method(fig_mask, label_method):
    if label_method == 0:
        kernel = np.ones((2, 2), np.uint8)
        fig_mask = cv2.dilate(fig_mask, kernel, iterations=1)
    elif label_method == 1:
        pass
    elif label_method == 2:
        kernel = np.ones((2, 2), np.uint8)
        temp_mask = cv2.erode(fig_mask, kernel, iterations=1)
        fg_row_indices, fg_col_indices = np.where(temp_mask[:, :, 0] == 1)
        fg_coordinates = list(zip(fg_col_indices, fg_row_indices))
        fg_coordinates = np.array(fg_coordinates)
        num_fg_points = int(len(fg_coordinates))

        bg_row_indices, bg_col_indices = np.where(temp_mask[:, :, 0] == 2)
        bg_coordinates = list(zip(bg_col_indices, bg_row_indices))
        bg_coordinates = np.array(bg_coordinates)
        num_bg_points = int(len(bg_coordinates))
        if num_fg_points == 0 or num_bg_points == 0:
            return fig_mask
    return fig_mask


def get_mask(image, fig_mask):
    np.random.seed(42)
    # get instance mask
    predictor.set_image(image)

    # 获取前景点坐标
    fg_row_indices, fg_col_indices = np.where(fig_mask[:, :, 0] == 1)
    fg_coordinates = list(zip(fg_col_indices, fg_row_indices))
    np.random.shuffle(fg_coordinates)  # 随机打乱前景点坐标
    fg_coordinates = np.array(fg_coordinates)
    num_fg_points = int(len(fg_coordinates) / 10)
    fg_coordinates_label = np.ones(num_fg_points)

    # 获取背景点坐标
    bg_row_indices, bg_col_indices = np.where(fig_mask[:, :, 0] == 2)
    bg_coordinates = list(zip(bg_col_indices, bg_row_indices))
    np.random.shuffle(bg_coordinates)  # 随机打乱背景点坐标
    bg_coordinates = np.array(bg_coordinates)
    num_bg_points = int(len(bg_coordinates) / 5)
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
    mask = mask[0].astype(int)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return mask


def get_strong_mask(image, mask):
    return_mask = np.zeros_like(mask)
    for i in range(3):
        s_mask = mask_method(mask, i)
        return_mask += 0.25 * (i + 1) * get_mask(image, s_mask)
    return return_mask


def get_random_mask(image_files):
    random_number = random.randint(0, 7)

    sam_image = cv2.imread(image_files)
    mask_path = image_files.replace('Image', 'oldScribble').replace('jpg', 'png')
    mask = cv2.imread(mask_path).astype(np.float32)[:, :, ::-1]
    H1, W1, C1 = sam_image.shape
    sam_image = cv2.resize(sam_image, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    H, W, C = sam_image.shape

    if random_number == 0:
        flipped_image = np.flip(sam_image, axis=1)
        flipped_mask = np.flip(mask, axis=1)
        flipped_mask = get_strong_mask(flipped_image, flipped_mask)
        return_mask = np.flip(flipped_mask, axis=1)

    if random_number >= 1 and random_number <= 3:
        # rotation
        if random_number == 1:
            angle = cv2.ROTATE_90_CLOCKWISE
            return_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif random_number == 2:
            angle = cv2.ROTATE_180
            return_angle = cv2.ROTATE_180
        elif random_number == 3:
            angle = cv2.ROTATE_90_COUNTERCLOCKWISE
            return_angle = cv2.ROTATE_90_CLOCKWISE
        rotated_image = cv2.rotate(sam_image, angle)
        rotated_mask = cv2.rotate(mask, angle)
        rotated_mask = get_strong_mask(rotated_image, rotated_mask)
        return_mask = cv2.rotate(rotated_mask, return_angle)

    if random_number == 4:
        return_mask = get_strong_mask(sam_image, mask)

    if random_number > 4:
        if random_number == 5:
            # 0.5
            scaling = 2
        if random_number == 6:
            # 1
            scaling = 1
        if random_number == 7:
            # 2
            scaling = 0.5
        # shrink
        normal_sam_image = cv2.resize(sam_image, (int(W/scaling), int(H/scaling)), interpolation=cv2.INTER_NEAREST)
        normal_mask = cv2.resize(mask, (int(W/scaling), int(H/scaling)), interpolation=cv2.INTER_NEAREST)
        normal_mask = get_strong_mask(normal_sam_image, normal_mask)
        return_mask = cv2.resize(normal_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return_mask = cv2.resize(return_mask, (W1, H1), interpolation=cv2.INTER_NEAREST)

    return return_mask, mask_path
if __name__=='__main__':
    np.random.seed(42)
    # read image
    path = r'E:\Mr.Wu\dataset\CodDataset\train\Image'
    image_files = glob.glob(os.path.join(path, "*.jpg"))
    K = 6
    Ua = 1
    Ta = 0.1
    Ur = 1
    Tr = 0.5
    for image_files in image_files:
        # try:
            for k in range(K):
                if k == 0:
                    mask, mask_path = get_random_mask(image_files)
                    final_mask = mask
                elif k == K - 1:
                    mask, mask_path = get_random_mask(image_files)
                    final_mask = mask + final_mask
                    final_mask = final_mask
                else:
                    mask, _ = get_random_mask(image_files)
                    final_mask = mask + final_mask
            final_mask[final_mask > 0.25 * K] = 255
            final_mask[final_mask <= 0.25 * K] = 0
            final_mask = final_mask.astype(np.uint8)
            maskpath_save = mask_path.replace('oldScribble', 'K=6')
            mask_image = Image.fromarray(final_mask)
            mask_image.save(maskpath_save)
            print(image_files)
        # except:
        #     print(image_files, r'wrong')