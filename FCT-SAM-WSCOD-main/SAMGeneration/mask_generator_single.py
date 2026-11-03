import numpy as np
import random
import os
import glob
import plotly.graph_objs as go
from PIL import Image
import cv2
from segment_anything import SamPredictor, sam_model_registry
sam_checkpoint = "E:\Mr.Wu\codes\Weakly-Supervised-Camouflaged-Transformer\pretrained\sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = 'cuda'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def get_random_mask(image_files, maskpath_save, i):
    sam_image = cv2.imread(image_files)
    mask_path = image_files.replace('Images', 'Scribble').replace('jpg', 'png')
    mask = cv2.imread(mask_path).astype(np.float32)[:, :, ::-1]

    H, W, C = sam_image.shape
    sam_image = cv2.resize(sam_image, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # get different image and mask
    # try:
    # flip
    flipped_image = np.flip(sam_image, axis=1)
    flipped_mask = np.flip(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST), axis=1)
    flipped_mask = get_mask(flipped_image, flipped_mask)

    flipped_save = maskpath_save + '\\' + 'flip' + str(i) + '.png'
    flipped_save_mask = flipped_mask.copy()
    flipped_save_mask[flipped_save_mask == 1] = 255
    flipped_save_mask = flipped_save_mask.astype(np.uint8)
    flipped_mask_image = Image.fromarray(flipped_save_mask)
    flipped_mask_image.save(flipped_save)

    flipped_mask = np.flip(flipped_mask, axis=1)
    flipped_mask = cv2.resize(flipped_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # rotation
    angle = 90
    rotation_matrix = cv2.getRotationMatrix2D((512 / 2, 512 / 2), angle, 1)
    rotated_image = cv2.warpAffine(sam_image, rotation_matrix, (512, 512))
    rotated_mask = cv2.warpAffine(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST), rotation_matrix,
                                  (512, 512))
    rotated_mask = get_mask(rotated_image, rotated_mask)

    rotated_save = maskpath_save + '\\' + 'rotate' + str(i) + '.png'
    rotated_save_mask = rotated_mask.copy()
    rotated_save_mask[rotated_save_mask == 1] = 255
    rotated_save_mask = rotated_save_mask.astype(np.uint8)
    rotated_mask_image = Image.fromarray(rotated_save_mask)
    rotated_mask_image.save(rotated_save)

    return_angle = -90
    rotation_matrix = cv2.getRotationMatrix2D((512 / 2, 512 / 2), return_angle, 1)
    rotated_mask = cv2.warpAffine(rotated_mask, rotation_matrix, (512, 512))
    rotated_mask = cv2.resize(rotated_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # shrink
    normal_sam_image = cv2.resize(sam_image, (512, 512), interpolation=cv2.INTER_NEAREST)
    normal_mask = cv2.resize(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST), (512, 512),
                             interpolation=cv2.INTER_LINEAR)
    normal_mask = get_mask(normal_sam_image, normal_mask)

    normal_save = maskpath_save + '\\' + 'normal' + str(i) + '.png'
    normal_save_mask = normal_mask.copy()
    normal_save_mask[normal_save_mask == 1] = 255
    normal_save_mask = normal_save_mask.astype(np.uint8)
    normal_mask_image = Image.fromarray(normal_save_mask)
    normal_mask_image.save(normal_save)

    normal_mask = cv2.resize(normal_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # fusion different mask

    final_mask = (flipped_mask + rotated_mask + normal_mask)
    final_mask[(final_mask < 2)] = 0
    final_mask[(final_mask >= 2) & (final_mask <= 3)] = 1
    final_mask[final_mask == 6] = 2


    return final_mask, mask_path

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

if __name__=='__main__':
    np.random.seed(42)
    # read image
    path = r"E:\Mr.Wu\dataset\CodDataset\train\Image\camourflage_00009.jpg"
    i = 0
    epoch = 10
    maskpath_save = r"C:\Users\farewell_iamloser\Desktop\Liu版本"
    # try:
    while i < epoch:
        if i == 0:
            mask, mask_path = get_random_mask(path, maskpath_save, i)
            final_mask = mask

        elif i == epoch - 1:
            mask, _ = get_random_mask(path, maskpath_save, i)
            final_mask = mask + final_mask

            final_mask[final_mask == epoch * 2] = 0
            final_mask[final_mask < epoch / 4] = 0
            final_mask[(final_mask >= epoch / 4) & (final_mask <= epoch)] = 255

        else:
            mask, _ = get_random_mask(path, maskpath_save, i)
            final_mask = mask + final_mask
        i += 1
        temp_save = maskpath_save + '\\' + str(i) + '.png'
        mask = mask.copy()
        mask[mask == 1] = 255
        mask = mask.astype(np.uint8)
        mask_image = Image.fromarray(mask)
        mask_image.save(temp_save)
    # except:
    #     print(path, r'wrong')
    final_mask = final_mask.astype(np.uint8)
    mask_image = Image.fromarray(final_mask)
    final_path = maskpath_save + '\\' + 'final.jpg'
    mask_image.save(final_path)
    print(maskpath_save)

