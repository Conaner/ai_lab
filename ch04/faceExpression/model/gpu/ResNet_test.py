# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from statistics import mode

# 人脸数据归一化,将像素值从0-255映射到0-1之间
def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images/255.0
    return images

class ResNet(nn.Module):
    def __init__(self, *args):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

resnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7 , stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
resnet.add_module("global_avg_pool", GlobalAvgPool2d())
resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7)))

# 检查CUDA是否可用
device = torch.device("cuda")
resnet.to(device)

detection_model_path = r'C:\Users\17216\Desktop\ai_lab\ch04\faceExpression\model\haarcascade_frontalface_default.xml'
classification_model_path = r'C:\Users\17216\Desktop\ai_lab\ch04\faceExpression\model\gpu\model_resnet_gpu.pth'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = torch.load(classification_model_path, map_location=device)  # 将模型加载到GPU

frame_window = 10
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
emotion_window = []

# 调起摄像头，0是笔记本自带摄像头
# video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture(r"C:\Users\17216\Desktop\ai_lab\ch04\faceExpression\model\video\example_dsh.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')

while True:
    _, frame = video_capture.read()
    frame = frame[:,::-1,:]  # 水平翻转，符合自拍习惯
    frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)
        face = gray[y:y+h,x:x+w]
        try:
            face = cv2.resize(face,(48,48))
        except:
            continue
        face = np.expand_dims(face,0)
        face = np.expand_dims(face,0)
        face = preprocess_input(face)
        new_face = torch.from_numpy(face).to(device)  # 将数据发送到GPU
        new_new_face = new_face.float().requires_grad_(False)
        emotion_arg = np.argmax(emotion_classifier(new_new_face).detach().cpu().numpy())
        emotion = emotion_labels[emotion_arg]
        emotion_window.append(emotion)
        if len(emotion_window) >= frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        cv2.putText(frame, emotion_mode, (x, y-30), font, .7, (0,0,255), 1, cv2.LINE_AA)

    try:
        cv2.imshow('window_frame', frame)
    except:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()