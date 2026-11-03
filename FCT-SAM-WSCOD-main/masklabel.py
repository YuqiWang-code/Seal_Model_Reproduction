from PIL import Image
import os

# 输入和输出文件夹路径
input_folder = r"E:\Mr.Wu\dataset\CodDataset\train\onlyScribble"

output_folder = r"E:\Mr.Wu\dataset\CodDataset\train\笔画"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 打开图像
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)
        real_img = Image.open(image_path.replace('onlyScribble', 'Image').replace('png', 'jpg'))
        # 遍历图像的每个像素并更改颜色
        width, height = img.size
        for y in range(height):
            for x in range(width):
                pixel = img.getpixel((x, y))
                # 根据像素值更改颜色
                if pixel == (2, 2, 2):  # 前景像素
                    real_img.putpixel((x, y), (255, 0, 0))  # 红色
                elif pixel == (1, 1, 1):  # 背景像素
                    real_img.putpixel((x, y), (0, 255, 0))  # 绿色

        # 保存更改颜色后的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        real_img.save(output_path)

print("处理完成")
