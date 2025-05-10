from ultralytics import YOLO
#import os

def test_model(model_path, data_yaml_path, test_images_dir):
    # 加载训练好的模型
    model = YOLO(model_path)

    # 进行评估
    metrics = model.val(data=data_yaml_path, split='test')
    print(metrics)

    # 对测试集中的图像进行预测
    predictions = model.predict(source=test_images_dir, save=True)

    print("Testing completed!")

# 指定路径
# model_path = '\pythonpa\ch01-1\FaceMask\maskDet\model\yolov8n.pt'
# data_yaml_path = '\pythonpa\ch01-1\FaceMask\maskDet\model\datasets.yaml'
# test_images_dir = r'\pythonpa\ch01-1\FaceMask\maskDet\data\test'
model_path = r'E:\YOLO11\ultralytics\runs\detect\train29\weights\best.pt'
data_yaml_path = r'C:\Users\17216\Desktop\ai_lab\ch06\defect-detection\data\NEU-DET\data.yaml'
test_images_dir = r'C:\Users\17216\Desktop\ai_lab\ch06\defect-detection\data\NEU-DET\test\images'
# 注意这里指向整个测试数据集目录

if __name__ == "__main__":
    # 调用测试函数
    test_model(model_path, data_yaml_path, test_images_dir)