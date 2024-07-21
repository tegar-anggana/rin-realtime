from ultralytics import YOLO
from enum import Enum
import threading
import time
import keyboard

class ModelType(Enum):
    YOLOv8n = 'yolov8n.pt'

class Camera(Enum):
    LAPTOP = '0'

class ObjectDetection:
    def __init__(self, modelType: ModelType):
        self.modelType = modelType
        self.model = YOLO(self.modelType.value)
        self.running = False
        self.thread = None

    def start_detection(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._detect)
            self.thread.start()
            print("Object detection started.")

    def stop_detection(self):
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()
            print("Object detection stopped.")

    def _detect(self):
        while self.running:
            self.model.predict(source=Camera.LAPTOP.value, show=True)
            # Adding a small sleep to prevent excessive CPU usage
            time.sleep(0.1)

def main():
    detector = ObjectDetection(ModelType.YOLOv8n)

    def on_start_stop_event(e):
        if detector.running:
            detector.stop_detection()
        else:
            detector.start_detection()

    # Listen for 's' key to start/stop detection
    keyboard.on_press_key('s', on_start_stop_event)

    print("Press 's' to start/stop object detection. Press 'q' to quit.")

    # Main loop to keep the script running and listen for the 'q' key to quit
    while True:
        if keyboard.is_pressed('q'):
            if detector.running:
                detector.stop_detection()
            print("Quitting program.")
            break
        time.sleep(0.1)

if __name__ == '__main__':
    main()
