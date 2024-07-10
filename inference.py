from ultralytics import YOLO
import torch

# model = YOLO('/home/ruvi/Documents/code/python/yolo/yolov9-c-converted.pt')
model = YOLO('finetuned_model.pt')

model.info()

# results = model(['image.png'])  # return a list of Results objects
model.predict(source="image.png", save=True, classes=[0,28,26])