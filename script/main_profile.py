import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('ultralytics/cfg_yaml/test_model_yaml/ShuffleNet_24_04_04.0_light.yaml',task_name='detect')
    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()

    # Model summary (fused): 189 layers, 315890 parameters, 315874 gradients
    # Model summary (fused): 149 layers, 1408398 parameters, 1408382 gradients