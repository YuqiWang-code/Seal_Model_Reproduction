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
    nine_boxs = []
    for i in range(0, 3):
        for j in range(0, 3):
            grid_x_min = x_min + j * grid_width
            grid_x_max = x_min + (j + 1) * grid_width
            grid_y_min = y_min + i * grid_height
            grid_y_max = y_min + (i + 1) * grid_height
            nine_box = (grid_x_min, grid_x_max, grid_y_min, grid_y_max)
            nine_boxs.append(nine_box)

    a1 = False
    a2 = False
    a3 = False
    a4 = False
    a5 = False
    a6 = False
    a7 = False
    a8 = False
    a9 = False
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    b5 = 0
    b6 = 0
    b7 = 0
    b8 = 0
    b9 = 0
    pick_number = 1
    np.random.shuffle(points)
    for point in points:
        if len(sample_points) > 9 * pick_number:
            break
        if nine_boxs[0][0] <= point[0] <= nine_boxs[0][1] and nine_boxs[0][2] <= point[1] <= nine_boxs[0][3]:
            if a1 != True:
                b1 += 1
                sample_points.append(point)
            if b1 == pick_number:
                a1 = True
        if nine_boxs[1][0] <= point[0] <= nine_boxs[1][1] and nine_boxs[1][2] <= point[1] <= nine_boxs[1][3]:
            if a2 != True:
                b2 += 1
                sample_points.append(point)
            if b2 == pick_number:
                a2 = True
        if nine_boxs[2][0] <= point[0] <= nine_boxs[2][1] and nine_boxs[2][2] <= point[1] <= nine_boxs[2][3]:
            if a3 != True:
                b3 += 1
                sample_points.append(point)
            if b3 == pick_number:
                a3 = True
        if nine_boxs[3][0] <= point[0] <= nine_boxs[3][1] and nine_boxs[3][2] <= point[1] <= nine_boxs[3][3]:
            if a4 != True:
                b4 += 1
                sample_points.append(point)
            if b4 == pick_number:
                a4 = True
        if nine_boxs[4][0] <= point[0] <= nine_boxs[4][1] and nine_boxs[4][2] <= point[1] <= nine_boxs[4][3]:
            if a5 != True:
                b5 += 1
                sample_points.append(point)
            if b5 == pick_number:
                a5 = True
        if nine_boxs[5][0] <= point[0] <= nine_boxs[5][1] and nine_boxs[5][2] <= point[1] <= nine_boxs[5][3]:
            if a6 != True:
                b6 += 1
                sample_points.append(point)
            if b6 == pick_number:
                a6 = True
        if nine_boxs[6][0] <= point[0] <= nine_boxs[6][1] and nine_boxs[6][2] <= point[1] <= nine_boxs[6][3]:
            if a7 != True:
                b7 += 1
                sample_points.append(point)
            if b7 == pick_number:
                a7 = True
        if nine_boxs[7][0] <= point[0] <= nine_boxs[7][1] and nine_boxs[7][2] <= point[1] <= nine_boxs[7][3]:
            if a8 != True:
                b8 += 1
                sample_points.append(point)
            if b8 == pick_number:
                a8 = True
        if nine_boxs[8][0] <= point[0] <= nine_boxs[8][1] and nine_boxs[8][2] <= point[1] <= nine_boxs[8][3]:
            if a9 != True:
                b9 += 1
                sample_points.append(point)
            if b9 == pick_number:
                a9 = True
    return sample_points


def get_bbx(fig_mask, select_points):
    bbxs = []
    for select_point in select_points:
        x, y = select_point
        # get bounding box
        left = x
        right = x
        up = y
        down = y
        lu_x, lu_y = x, y
        ld_x, ld_y = x, y
        ru_x, ru_y = x, y
        rd_x, rd_y = x, y

        while True:
            if fig_mask[left, y, 0] != 1:
                break
            left = left - 1
        while True:
            if fig_mask[right, y, 0] != 1:
                break
            right = right + 1
        while True:
            if fig_mask[x, up, 0] != 1:
                break
            up = up + 1
        while True:
            if fig_mask[x, down, 0] != 1:
                break
            down = down - 1
        while True:
            if fig_mask[lu_x, lu_y, 0] != 1:
                break
            lu_x = lu_x - 1
            lu_y = lu_y + 1
        while True:
            if fig_mask[ld_x, ld_y, 0] != 1:
                break
            ld_x = ld_x - 1
            ld_y = ld_y - 1
        while True:
            if fig_mask[ru_x, ru_y, 0] != 1:
                break
            ru_x = ru_x + 1
            ru_y = ru_y + 1
        while True:
            if fig_mask[rd_x, rd_y, 0] != 1:
                break
            rd_x = rd_x + 1
            rd_y = rd_y - 1

        bbx = np.array([left, down, right, up])
        bbxs.append(bbx)
    return bbxs


def get_mask(image, fig_mask):
    np.random.seed(42)
    # get instance mask
    predictor.set_image(image)

    # 获取前景点坐标
    fg_row_indices, fg_col_indices = np.where(fig_mask[:, :, 0] == 1)
    fg_coordinates = list(zip(fg_col_indices, fg_row_indices))
    fg_sample_points = nine_grid_sampling(fg_coordinates)

    num_fg_points = len(fg_sample_points)  # 选择前景点的数量，最多选择10个
    fg_coordinates_label = np.ones(num_fg_points)

    # 获取背景点坐标
    bg_row_indices, bg_col_indices = np.where(fig_mask[:, :, 0] == 2)
    bg_coordinates = list(zip(bg_col_indices, bg_row_indices))
    bg_sample_points = nine_grid_sampling(bg_coordinates)
    num_bg_points = len(bg_sample_points)  # 选择背景点的数量，确保总共选择10个点
    bg_coordinates_label = np.zeros(num_bg_points)
    coordinates = np.concatenate((fg_sample_points, bg_sample_points), axis=0)
    coordinates_labels = np.concatenate((fg_coordinates_label, bg_coordinates_label), axis=0)
    coordinate = coordinates[0:1, :]
    coordinates_label = coordinates_labels[0:1]

    # 获取边框信息
    bbx_masks = []
    input_boxs = get_bbx(fig_mask, fg_sample_points)
    for input_box in input_boxs:
        bbx_mask, _, _ = predictor.predict(
            point_coords=coordinate,
            point_labels=coordinates_label,
            box=input_box[None, :],
            multimask_output=False,
        )
        bbx_mask = bbx_mask[0].astype(int)
        bbx_mask = np.repeat(bbx_mask[:, :, np.newaxis], 3, axis=2)
        bbx_masks.append(bbx_mask)
    bbx_mask = np.sum(np.stack(bbx_masks), axis=0)

    return bbx_mask

def get_random_mask(image_files):
    random_number = random.randint(0, 7)

    sam_image = cv2.imread(image_files)
    mask_path = image_files.replace('Image', 'oldScribble').replace('jpg', 'png')
    mask = cv2.imread(mask_path).astype(np.float32)[:, :, ::-1]

    model_mask_path = image_files.replace('Image', 'step2_temp_output').replace('jpg', 'png')
    model_mask = cv2.imread(model_mask_path).astype(np.float32)[:, :, ::-1] / 255

    fg_mask = mask.copy()
    fg_mask[fg_mask == 2] = 0
    fg_mask = model_mask + fg_mask
    fg_mask[fg_mask == 2] = 1

    bg_mask = mask.copy()
    bg_mask[bg_mask == 1] = 0
    bg_mask = fg_mask + bg_mask
    bg_mask[bg_mask == 3] = 2
    mask = bg_mask.copy()

    H1, W1, C1 = sam_image.shape
    sam_image = cv2.resize(sam_image, (384, 384), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)
    H, W, C = sam_image.shape

    if random_number == 0:
        flipped_image = np.flip(sam_image, axis=1)
        flipped_mask = np.flip(mask, axis=1)
        flipped_mask = get_mask(flipped_image, flipped_mask)
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
        rotated_mask = get_mask(rotated_image, rotated_mask)
        return_mask = cv2.rotate(rotated_mask, return_angle)

    if random_number == 4:
        return_mask = get_mask(sam_image, mask)

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
        normal_mask = get_mask(normal_sam_image, normal_mask)
        return_mask = cv2.resize(normal_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return_mask = cv2.resize(return_mask, (W1, H1), interpolation=cv2.INTER_NEAREST)

    return return_mask, mask_path

def update_label():
    path = r'E:\Mr.Wu\dataset\CodDataset\train\Image'
    np.random.seed(42)
    # read image
    image_files = glob.glob(os.path.join(path, "*.jpg"))
    K = 6

    for image_files in image_files:
        try:
            for k in range(K):

                if k == 0:
                    mask, mask_path = get_random_mask(image_files)
                    final_mask = mask
                elif k == K - 1:
                    mask, mask_path = get_random_mask(image_files)
                    final_mask = mask + final_mask
                    non_zero_values = final_mask[final_mask != 0]
                    average_non_zero = np.mean(non_zero_values)
                    final_mask[final_mask > average_non_zero / 3] = 255
                else:
                    mask, _ = get_random_mask(image_files)
                    final_mask = mask + final_mask

            final_mask = final_mask.astype(np.uint8)
            maskpath_save = mask_path.replace('oldScribble', 'bbx_scribble')
            mask_image = Image.fromarray(final_mask)
            mask_image.save(maskpath_save)
            print(image_files)
        except:
                print(image_files, r'wrong')
if __name__=='__main__':
    update_label()