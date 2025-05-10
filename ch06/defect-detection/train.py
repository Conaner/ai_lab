from ultralytics import YOLO
 
# Load a model

model = YOLO(r'C:\Users\17216\Desktop\ai_lab\ch06\defect-detection\yolov8n.pt')  # load a pretrained model (recommended for training)
 
# Train the model
if __name__ == '__main__':
    model.train(data=r'C:\Users\17216\Desktop\ai_lab\ch06\defect-detection\data\NEU-DET\data.yaml', epochs=20, imgsz=640, device='0')  # device指定0号GPU执行