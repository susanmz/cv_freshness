# from ultralytics import YOLO
# import os
# from IPython.display import display, Image
# from IPython import display
# display.clear_output
# # ! yolo checks

# # ! yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=8 
# # model = YOLO()
# # results = model.train(task='detect', data='C:\\Users\\susan\\Documents\\_UM\\23Fall\\EECS504\\cv_freshness\\data.yaml', epochs=8, imgsz=640)
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco128.yaml', epochs=100, imgsz=640)