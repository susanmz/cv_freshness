from PIL import Image
from ultralytics import YOLO
model = YOLO('runs/detect/train9/weights/best.pt') # Kaggle data
results = model.predict(source='.\\GOODPHOTOS\\aob.jpg', imgsz=800, conf=0.25, save=True)

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image
