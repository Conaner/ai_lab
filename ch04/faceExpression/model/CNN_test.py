import cv2
import torch
import torch.nn as nn
import numpy as np
from statistics import mode
import os
from tqdm import tqdm
import random


# 人脸数据归一化,将像素值从0-255映射到0-1之间
def preprocess_input(images):
    images = images / 255.0
    return images


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y


# 指定.pth文件路径
detection_model_path = r'D:\pythonpa\ch01-1\faceExpression\model\haarcascade_frontalface_default.xml'
classification_model_path = r"D:\pythonpa\ch01-1\faceExpression\model\model_cnn_cpu.pth"

# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(detection_model_path)

# 加载表情识别模型
device = torch.device('cpu')
# 直接加载整个模型实例
emotion_classifier = torch.load(classification_model_path, map_location=device)
emotion_classifier.to(device)
emotion_classifier.eval()

frame_window = 10

# 表情标签
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

emotion_window = []

# 调起摄像头，0是笔记本自带摄像头
#video_capture = cv2.VideoCapture(0)
# 视频文件识别
video_capture = cv2.VideoCapture(r"D:\pythonpa\ch01-1\faceExpression\model\video\example_dsh.mp4")

# 获取视频的帧率和尺寸
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 输出帧图像的目录
output_dir = r'D:\pythonpa\ch01-1\faceExpression\model\video\output_frames'

os.makedirs(output_dir, exist_ok=True)

font = cv2.FONT_HERSHEY_SIMPLEX

# 获取视频总帧数
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# 随机选择几帧进行处理
num_frames_to_process = 10
random_frames = sorted(random.sample(range(total_frames), num_frames_to_process))

frame_count = 0

# 使用 tqdm 包装循环以显示进度条
for frame_number in tqdm(range(total_frames), desc="Processing frames"):
    # 读取一帧
    ret, frame = video_capture.read()
    if not ret:
        break  # 如果视频结束，退出循环

    if frame_number in random_frames:
        frame = frame[:, ::-1, :]  # 水平翻转，符合自拍习惯
        frame = frame.copy()
        # 获得灰度图，并且在内存中创建一个图像对象
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 获取当前帧中的全部人脸
        faces = face_detection.detectMultiScale(gray, 1.3, 5)
        # 对于所有发现的人脸
        for (x, y, w, h) in faces:
            # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
            cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)

            # 获取人脸图像
            face = gray[y:y + h, x:x + w]

            try:
                # shape变为(48,48)
                face = cv2.resize(face, (48, 48))
            except:
                continue

            # 扩充维度，shape变为(1,48,48,1)
            # 将（1，48，48，1）转换成为(1,1,48,48)
            face = np.expand_dims(face, 0)
            face = np.expand_dims(face, 0)

            # 人脸数据归一化，将像素值从0-255映射到0-1之间
            face = preprocess_input(face)
            face_tensor = torch.from_numpy(face).float().to(device)

            # 调用我们训练好的表情识别模型，预测分类
            with torch.no_grad():
                emotion_arg = torch.argmax(emotion_classifier(face_tensor)).item()
            emotion = emotion_labels[emotion_arg]

            emotion_window.append(emotion)

            if len(emotion_window) >= frame_window:
                emotion_window.pop(0)

            try:
                # 获得出现次数最多的分类
                emotion_mode = mode(emotion_window)
            except:
                continue

            # 在矩形框上部，输出分类文字
            cv2.putText(frame, emotion_mode, (x, y - 30), font, .7, (0, 0, 255), 1, cv2.LINE_AA)

        # 保存当前帧为图像文件
        frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_filename, frame)

    # 增加帧计数
    frame_count += 1

    # 按q退出
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

video_capture.release()
#cap.release()
#cv2.destroyAllWindows()

print(f"共处理了 {len(random_frames)} 帧，已保存到 {output_dir}")
