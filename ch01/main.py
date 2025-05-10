from ultralytics import YOLO

# Load a model


if __name__ == '__main__':
# Train the model
# results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
    # model = YOLO('yolov8x.yaml')  # build a new model from YAML
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8x.yaml').load('yolov8x.pt')  # build from YAML and transfer weights
    results = model.train(data=r'C:\Users\17216\Desktop\ai_lab\ch01\drone.yaml', epochs=10, imgsz=640)