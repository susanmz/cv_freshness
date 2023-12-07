from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output
# # ! yolo checks

# # ! yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=8 
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
results = model.train(task='detect', data='./dataset2_data.yaml', epochs=30, imgsz=640)
