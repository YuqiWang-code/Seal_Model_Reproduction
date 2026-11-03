from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io

app = Flask(__name__)
CORS(app)


def segment_image(img):
    # 将 PIL 图像转换为 NumPy 数组
    img_array = np.array(img)

    # 使用 OpenCV 的 grabCut 方法进行图像分割
    mask = np.zeros(img_array.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, img_array.shape[1] - 50, img_array.shape[0] - 50)
    cv2.grabCut(img_array, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_img_array = img_array * mask2[:, :, np.newaxis]

    # 将结果图像转换回 PIL 图像
    segmented_img = Image.fromarray(segmented_img_array)
    return segmented_img


@app.route('/segment', methods=['POST'])
def segment_image_route():
    if 'file' not in request.files:
        return 'No file provided', 400

    file = request.files['file']
    img = Image.open(file.stream)

    # 调用实际的图像分割函数
    segmented_img = segment_image(img)

    # 将结果图像返回到前端
    img_io = io.BytesIO()
    segmented_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
