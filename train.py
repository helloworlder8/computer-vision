from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data_cfg='coco128.yaml', epochs=100, imgsz=640, device=[0, 1,2,3,4,5,6,7])