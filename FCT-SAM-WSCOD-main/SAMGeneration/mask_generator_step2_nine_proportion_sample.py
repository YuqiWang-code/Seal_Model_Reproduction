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

    sample1_points = []
    sample2_points = []
    sample3_points = []
    sample4_points = []
    sample5_points = []
    sample6_points = []
    sample7_points = []
    sample8_points = []
    sample9_points = []
    np.random.shuffle(points)
    for point in points:
        if nine_boxs[0][0] <= point[0] < nine_boxs[0][1] and nine_boxs[0][2] <= point[1] < nine_boxs[0][3]:
            sample1_points.append(point)
        if nine_boxs[1][0] <= point[0] < nine_boxs[1][1] and nine_boxs[1][2] <= point[1] < nine_boxs[1][3]:
            sample2_points.append(point)
        if nine_boxs[2][0] <= point[0] <= nine_boxs[2][1] and nine_boxs[2][2] <= point[1] < nine_boxs[2][3]:
            sample3_points.append(point)
        if nine_boxs[3][0] <= point[0] < nine_boxs[3][1] and nine_boxs[3][2] <= point[1] < nine_boxs[3][3]:
            sample4_points.append(point)
        if nine_boxs[4][0] <= point[0] < nine_boxs[4][1] and nine_boxs[4][2] <= point[1] < nine_boxs[4][3]:
            sample5_points.append(point)
        if nine_boxs[5][0] <= point[0] <= nine_boxs[5][1] and nine_boxs[5][2] <= point[1] < nine_boxs[5][3]:
            sample6_points.append(point)
        if nine_boxs[6][0] <= point[0] < nine_boxs[6][1] and nine_boxs[6][2] <= point[1] <= nine_boxs[6][3]:
            sample7_points.append(point)
        if nine_boxs[7][0] <= point[0] < nine_boxs[7][1] and nine_boxs[7][2] <= point[1] <= nine_boxs[7][3]:
            sample8_points.append(point)
        if nine_boxs[8][0] <= point[0] <= nine_boxs[8][1] and nine_boxs[8][2] <= point[1] <= nine_boxs[8][3]:
            sample9_points.append(point)
    total_number = len(points)
    pick_number = 20
    if total_number > pick_number:
        number1_proportion = int(np.ceil((len(sample1_points) / total_number * pick_number)))
        number2_proportion = int(np.ceil((len(sample2_points) / total_number * pick_number)))
        number3_proportion = int(np.ceil((len(sample3_points) / total_number * pick_number)))
        number4_proportion = int(np.ceil((len(sample4_points) / total_number * pick_number)))
        number5_proportion = int(np.ceil((len(sample5_points) / total_number * pick_number)))
        number6_proportion = int(np.ceil((len(sample6_points) / total_number * pick_number)))
        number7_proportion = int(np.ceil((len(sample7_points) / total_number * pick_number)))
        number8_proportion = int(np.ceil((len(sample8_points) / total_number * pick_number)))
        number9_proportion = int(np.ceil((len(sample9_points) / total_number * pick_number)))
    else:
        number1_proportion = len(sample1_points)
        number2_proportion = len(sample2_points)
        number3_proportion = len(sample3_points)
        number4_proportion = len(sample4_points)
        number5_proportion = len(sample5_points)
        number6_proportion = len(sample6_points)
        number7_proportion = len(sample7_points)
        number8_proportion = len(sample8_points)
        number9_proportion = len(sample9_points)
    np.random.shuffle(sample1_points)
    np.random.shuffle(sample2_points)
    np.random.shuffle(sample3_points)
    np.random.shuffle(sample4_points)
    np.random.shuffle(sample5_points)
    np.random.shuffle(sample6_points)
    np.random.shuffle(sample7_points)
    np.random.shuffle(sample8_points)
    np.random.shuffle(sample9_points)

    sample_points_list = [sample1_points, sample2_points, sample3_points,
                          sample4_points, sample5_points, sample6_points,
                          sample7_points, sample8_points, sample9_points]

    non_empty_sample_points = [sample_points_list[i][:number_proportion] for i, number_proportion in enumerate(
        [number1_proportion, number2_proportion, number3_proportion, number4_proportion, number5_proportion,
         number6_proportion, number7_proportion, number8_proportion, number9_proportion]) if number_proportion > 0]

    sample_points = np.concatenate(non_empty_sample_points, axis=0) if non_empty_sample_points else np.array([])

    return sample_points


def get_mask(image, fig_mask):
    np.random.seed(42)
    # get instance mask
    predictor.set_image(image)

    # 获取前景点坐标
    fg_row_indices, fg_col_indices = np.where(fig_mask[:, :, 0] == 1)
    fg_coordinates = list(zip(fg_col_indices, fg_row_indices))
    fg_sample_points = nine_grid_sampling(fg_coordinates)
    num_fg_points = len(fg_sample_points)
    fg_coordinates_label = np.ones(num_fg_points)

    # 获取背景点坐标
    bg_row_indices, bg_col_indices = np.where(fig_mask[:, :, 0] == 2)
    bg_coordinates = list(zip(bg_col_indices, bg_row_indices))
    bg_sample_points = nine_grid_sampling(bg_coordinates)
    num_bg_points = len(bg_sample_points)
    bg_coordinates_label = np.zeros(num_bg_points)

    coordinates = np.concatenate((fg_sample_points, bg_sample_points), axis=0)
    coordinates_labels = np.concatenate((fg_coordinates_label, bg_coordinates_label), axis=0)
    result = np.transpose(np.zeros_like(image), (2, 0, 1))
    for i in range(20):
        coordinate = coordinates[i:i+1, :]
        coordinates_label = coordinates_labels[i:i+1]
        mask, scores, logits = predictor.predict(
            point_coords=coordinate,
            point_labels=coordinates_label,
            multimask_output=True,
        )

        coordinate = coordinates[:, :]
        coordinates_label = coordinates_labels[:]
        mask_input = logits[np.argmax(scores), :, :]
        mask, scores, logits = predictor.predict(
            point_coords=coordinate,
            point_labels=coordinates_label,
            mask_input=mask_input[None, :, :],
            multimask_output=True,
        )
        result += mask

    result = result[0].astype(int)
    result = np.repeat(result[:, :, np.newaxis], 3, axis=2)
    return result


def get_random_mask(image_files):
    random_number = random.randint(0, 7)

    sam_image = cv2.imread(image_files)
    mask_path = image_files.replace('Image', 'oldScribble').replace('jpg', 'png')
    mask = cv2.imread(mask_path).astype(np.float32)[:, :, ::-1]
    H1, W1, C1 = sam_image.shape
    sam_image = cv2.resize(sam_image, (768, 768), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)
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
        normal_sam_image = cv2.resize(sam_image, (int(W / scaling), int(H / scaling)), interpolation=cv2.INTER_NEAREST)
        normal_mask = cv2.resize(mask, (int(W / scaling), int(H / scaling)), interpolation=cv2.INTER_NEAREST)
        normal_mask = get_mask(normal_sam_image, normal_mask)
        return_mask = cv2.resize(normal_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return_mask = cv2.resize(return_mask, (W1, H1), interpolation=cv2.INTER_NEAREST)

    return return_mask, mask_path


if __name__ == '__main__':
    np.random.seed(42)
    # read image
    path = r'E:\Mr.Wu\dataset\CodDataset\train\Image'
    image_files = glob.glob(os.path.join(path, "*.jpg"))
    K = 10
    for image_files in image_files:
        # try:
            for k in range(K):
                if k == 0:
                    mask, mask_path = get_random_mask(image_files)
                    final_mask = mask
                elif k == K - 1:
                    mask, mask_path = get_random_mask(image_files)
                    final_mask = mask + final_mask
                    non_zero_values = final_mask[final_mask != 0]
                    average_non_zero = np.mean(non_zero_values)
                    final_mask[final_mask > average_non_zero/4] = 255
                    final_mask[final_mask <= average_non_zero/4] = 0
                else:
                    mask, _ = get_random_mask(image_files)
                    final_mask = mask + final_mask

            final_mask = final_mask.astype(np.uint8)
            maskpath_save = mask_path.replace('oldScribble', 'K=12')
            mask_image = Image.fromarray(final_mask)
            mask_image.save(maskpath_save)
            print(image_files)
        # except:
        #     print(image_files, r'wrong')