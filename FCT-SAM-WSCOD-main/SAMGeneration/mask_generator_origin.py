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

    # shrink
    normal_sam_image = cv2.resize(sam_image, (size, size), interpolation=cv2.INTER_NEAREST)
    normal_mask = cv2.resize(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST), (size, size),
                             interpolation=cv2.INTER_LINEAR)
    normal_mask = get_mask(normal_sam_image, normal_mask)
    normal_mask = cv2.resize(normal_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # fusion different mask

    final_mask = normal_mask
    final_mask[final_mask == 1] = 255
    final_mask[final_mask == 2] = 0


    return final_mask, mask_path
if __name__=='__main__':
    # read image
    path = r'E:\Mr.Wu\dataset\CodDataset\train\Image'
    image_files = glob.glob(os.path.join(path, "*.jpg"))
    for image_files in image_files:
        size = 512
        i = 0
        epoch = 1
        try:
            while i < epoch:
                if i == 0:
                    mask, mask_path = get_random_mask(image_files, size)
                    final_mask = mask
                elif i == epoch-1:
                    mask, _ = get_random_mask(image_files, size)
                    final_mask = mask + final_mask
                    final_mask[final_mask < epoch / 2] = 0
                    final_mask[(final_mask >= epoch / 2) & (final_mask <= epoch)] = 255
                    final_mask[final_mask == epoch * 2] = 0
                else:
                    mask, _ = get_random_mask(image_files, size)
                    final_mask = mask + final_mask
                i += 1
        except:
            print(image_files, r'wrong')
        final_mask = final_mask.astype(np.uint8)
        maskpath_save = mask_path.replace('oldScribble', 'origin_sam')
        mask_image = Image.fromarray(final_mask)
        mask_image.save(maskpath_save)
        print(image_files)


        # except:
        #     print(mask_path)