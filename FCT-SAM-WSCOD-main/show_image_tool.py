import matplotlib.pyplot as plt
import numpy as np
import torch

def show_gray(gray_image_tensor, image_index, feature_index):
    # 假设你的灰度图像张量名为 gray_image_tensor，形状为 (10, 1, 384, 384)，并位于GPU上

    # 选择要查看的图像索引（0到9之间的整数）

    # 将灰度图像张量从GPU移回CPU并转换为NumPy数组
    gray_image_cpu = gray_image_tensor[image_index].cpu().detach().numpy()

    # 使用Matplotlib显示灰度图像
    plt.imshow(gray_image_cpu[feature_index], cmap='gray')  # cmap='gray'指定灰度图像
    plt.axis('off')  # 关闭坐标轴
    plt.show()

def show_rgb(image_tensor):

    # 假设你的张量名为 image_tensor，形状为 (10, 3, 384, 384)，并位于GPU上

    # 选择要查看的图像索引（0到9之间的整数）
    image_index = 0

    # 将张量从GPU移回CPU并转换为NumPy数组
    image_cpu = image_tensor[image_index].cpu().numpy()

    # 现在，image_cpu 包含了图像的数据，可以进行可视化
    # 可以选择某个通道进行显示，例如选择第一个通道（0）进行可视化
    plt.imshow(image_cpu.transpose(1, 2, 0))  # 转置以适应Matplotlib的通道顺序
    plt.axis('off')  # 关闭坐标轴
    plt.show()
