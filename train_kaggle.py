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
results = model.train(task='detect', data='./kaggle_data.yaml', epochs=1, imgsz=640)


# pip install ultralytics

# from ultralytics import YOLO
# import os
# from IPython.display import display, Image
# from IPython import display
# display.clear_output
# ! yolo checks

# ! yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'

# ! yolo task=detect mode=train model=yolov8n.pt data=/content/drive/MyDrive/data.yaml epochs=8 imgsz=640
