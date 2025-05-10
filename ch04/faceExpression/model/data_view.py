import cv2
import numpy as np
import os

# 指定图片的路径
image_path = '/root/autodl-tmp/faceExpression/data/process/images'

# 读取数据
data = np.loadtxt('/root/autodl-tmp/faceExpression/data/process/pixels.csv')

# 创建存储图片的目录，如果不存在的话
if not os.path.exists(image_path):
    os.makedirs(image_path)

# 处理数据
for i in range(data.shape[0]):
    face_array = data[i, :].reshape((48, 48))  # reshape
    image_filename = os.path.join(image_path, f'{i}.jpg')
    cv2.imwrite(image_filename, face_array)  # 保存图片