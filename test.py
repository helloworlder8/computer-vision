from ultralytics.projects.yolo.project_yolo import YOLO#(项目)
#以打开的文夹为参照物
# Load a yolo_calss
# yolo_calss = YOLO("yolov8n.yaml")  # build a new yolo_calss from scratch
YOLO_Project = YOLO("ultralytics/cfg_yaml/model_yaml/v8/yolov8.yaml")  # load a pretrained yolo_calss 默认模型 和任务

# Use the yolo_calss
# overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
# YOLO_Calss.train(data_cfg="coco128.yaml", epochs=3)  # train the yolo_calss

YOLO_Project.train(data_str='datasets/coco128/coco128.yaml', epochs=1, device=[2,3,4,5,6,7])
# YOLO_Project.train(data_str='datasets/coco128/coco128.yaml', epochs=1,device=[6])
# metrics = yolo_calss.val()  # evaluate yolo_calss performance on the validation set
# results = yolo_calss("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = yolo_calss.export(format="onnx")  # export the yolo_calss to ONNX format
# results.save()


# from ultralytics import YOLO

# # Build a YOLOv9c model from scratch
# model = YOLO('yolov9c.yaml')

# # Build a YOLOv9c model from pretrained weight
# model = YOLO('yolov9c.pt')

# # Display model information (optional)
# model.info()

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# # Run inference with the YOLOv9c model on the 'bus.jpg' image
# results = model('path/to/bus.jpg')
# 一开始情况数据
