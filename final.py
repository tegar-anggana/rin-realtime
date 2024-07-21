# https://docs.ultralytics.com/modes/predict
from ultralytics import YOLO
from enum import Enum

class ModelType(Enum):
    YOLOv8n = 'yolov8n.pt'

class Camera(Enum):
    LAPTOP = '0'

def liveObjectDetection(modelType:ModelType):
    model = YOLO(modelType.value)
    model.predict(
        # source=Camera.LAPTOP.value, 
        source='video.mp4', 
        show=True,
        # stream=True,
    )

if __name__ == '__main__':
    liveObjectDetection(ModelType.YOLOv8n)