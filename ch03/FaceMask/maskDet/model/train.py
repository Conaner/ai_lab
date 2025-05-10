import os
from ultralytics import YOLO

def train_model(data_yaml_path, model_cfg_path, weights_path, epochs=5, batch_size=16, img_size=640):
    # 加载模型
    model = YOLO(model_cfg_path)

    # 加载预训练权重
    if weights_path:
        model.load(weights_path)

    # 设置训练参数
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size
    )

    # 保存训练好的模型
    model.save("trained_model.pt")

    print("Training completed!")

if __name__ == "__main__":
    # 指定路径
    data_yaml_path = r'C:\Users\17216\Desktop\ai_lab\ch03\FaceMask\maskDet\data\datasets.yaml'
    #D:\pythonpa\ch01-1\FaceMask\maskDet\model
    model_cfg_path = 'yolov8n.yaml'
    weights_path = 'yolov8n.pt'
    epochs = 5
    batch_size = 16
    img_size = 640

    # 调用训练函数
    train_model(data_yaml_path, model_cfg_path, weights_path, epochs, batch_size, img_size)  # 原来的
    # train_model(data_yaml_path, weights_path, epochs, batch_size, img_size)