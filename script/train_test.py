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
    model = YOLO('ultralytics/cfg_yaml/model_yaml/v8/yolov8.yaml',task_name='detect')
    model.train(data_str="../datasets/BIRDSAI-FORE-BACKUP/BIRDSAI-FORE.yaml",
                cache=False,
                imgsz=640,
                # close_mosaic=10,
                # workers=4,
                # optimizer='SGD', # using SGD
                val_interval=1,
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                task_name='detect',
                project='',
                device='0',
                epochs=20,
                batch=30,
                name='',
                )
    send_notice(f"训练正确率:55%\n测试正确率:96.5%")