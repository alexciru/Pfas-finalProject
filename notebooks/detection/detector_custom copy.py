from ultralytics import YOLO
from pathlib import Path
import sys
import os
import cv2
import onnx


"""
Note (Alex): I used Elles code to train with the custom dataset.
Managed to make it work but is not able to save the weights.
"""


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = root_path + "\\data\\external\\customYolov8\\"
MODEL_PATH = root_path + "\\models\\yolov8n.pt"


img = root_path + "/data/validation/seq_01/image_02/data/0000000005.png"


# load weights
model = onnx.load(root_path + "/models/10epoch_low_pedestrians.onnx")
res = model(img)
boxes = res[0].boxes
print(boxes)


# draw boxes in the image
for box in boxes:
    box = box.xyxy
    label = box[5]
    box = box.tolist()[0]
    cv2.rectangle(img, (int(box[0]), int(box[1])) ,(int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.putText(img, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow("YOLOv8 Inference", img)





