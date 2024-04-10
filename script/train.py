import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import requests
def send_notice(content):
    token = "853672d072e640479144fba8b29b314b"
    title = "训练成功"
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template=html"
    response = requests.request("GET", url)
    print(response.text)


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg_yaml/models_yaml/v8/yolov8.yaml',task_name='detect')
    metrics = model.train(data_str="../../datasets/BIRDSAI-FORE-BACKUP/BIRDSAI-FORE.yaml",
                cache=False,
                imgsz=640,
                # close_mosaic=10,
                # workers=4,
                # optimizer='SGD', # using SGD
                val_interval=1,
                # resume='true', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                task_name='detect',
                project='',
                device='2,3',
                epochs=100,
                batch=250,
                name='V8-',
                )
    send_notice(f"Precision(B): {metrics['metrics/precision(B)']}, "
      f"Recall(B): {metrics['metrics/recall(B)']}, "
      f"mAP50(B): {metrics['metrics/mAP50(B)']}, "
      f"mAP50-95(B): {metrics['metrics/mAP50-95(B)']}, "
      f"Fitness: {metrics['fitness']}")
    

    # model = YOLO('ultralytics/cfg_yaml/model_yaml/v5/yolov5.yaml',task_name='detect')
    # metrics = model.train(data_str="../datasets/BIRDSAI-FORE-BACKUP/BIRDSAI-FORE.yaml",
    #             cache=False,
    #             imgsz=640,
    #             # close_mosaic=10,
    #             # workers=4,
    #             # optimizer='SGD', # using SGD
    #             val_interval=1,
    #             # resume='', # last.pt path
    #             # amp=False # close amp
    #             # fraction=0.2,
    #             task_name='detect',
    #             project='',
    #             device='2',
    #             epochs=200,
    #             batch=150,
    #             name='V5',
    #             )

    # send_notice(f"Precision(B): {metrics['/precision(B)']}, "
    #   f"Recall(B): {metrics['/recall(B)']}, "
    #   f"mAP50(B): {metrics['/mAP50(B)']}, "
    #   f"mAP50-95(B): {metrics['/mAP50-95(B)']}, "
    #   f"Fitness: {metrics['fitness']}")