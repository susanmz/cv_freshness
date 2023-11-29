from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output
# ! yolo checks

# ! yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=8 
model = YOLO()
results = model.train(task='detect', data='C:\\Users\\susan\\Documents\\_UM\\23Fall\\EECS504\\cv_freshness\\data.yaml', epochs=8, imgsz=640)