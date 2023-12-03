from PIL import Image
from ultralytics import YOLO
model = YOLO('runs/detect/train6/weights/best.pt') # Kaggle data
metrics = model.val()
# results = model.predict(source='final_test_orange_banana/images', conf=0.25, save=True)

# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image