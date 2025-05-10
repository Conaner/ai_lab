# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO

# 加载预训练的YOLOv8模型
model = YOLO(r'D:\pythonpa\ch01-1\FaceMask\maskDet\model\runs\detect\train\weights\best.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # 使用YOLOv8模型进行检测
    results = model(frame)

    # 解析检测结果并绘制边界框
    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            if cls == 2:  # 类别为2时，显示绿色
                color = (0, 255, 0)
            else:  # 其他类别显示红色
                color = (0, 0, 255)
            # 绘制边界框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 显示带有检测结果的视频帧
    cv2.imshow('YOLOv8 Real-time Detection', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()

