from ultralytics import YOLO
from roboflow import Roboflow
#rf = Roboflow(api_key="nLllOIGdCVe8CBQ0w1qB")
#project = rf.workspace("college-74jj5").project("freshness-fruits-and-vegetables")
#dataset = project.version(7).download("yolov8")

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data=f'./freshness--fruits-and-vegetables-7/data.yaml', epochs=30, imgsz=800, plots=True)
# yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
