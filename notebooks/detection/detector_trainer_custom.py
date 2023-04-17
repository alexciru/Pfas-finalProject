from ultralytics import YOLO
from pathlib import Path
import sys
import os



"""
Note (elle): I used this to train the pretrained model for just 1 epoch.
Not sure if that did anything in our favor, but it was just as a proof of concept.
"""


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = root_path + "\\data\\external\\customYolov8\\"
MODEL_PATH = root_path + "\\models\\yolov8n.pt"


print("DATASET PATH:  ")
print(DATASET_PATH)
# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model
model = YOLO(root_path + "/models/yolov8n.pt")

# Train the model using the 'coco128.yaml' dataset
# NOTE: name is subdirectory where model will be saved within 'project' dir
# see https://github.com/ultralytics/ultralytics/issues/512 for more details
results = model.train(
    data=DATASET_PATH + "data.yaml", epochs=1, project=DATASET_PATH, name="runs/detect/train"
)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model(root_path + "data\video\seq_01\image_02\data\0000000000.png")

# Export the model to ONNX format
print("Exporting model to ONNX format...")
success = model.export(format="onnx")
print("Export succeeded?:", success)

# load saved model -- aka do quick sanity test
model = YOLO(DATASET_PATH / "weights/best.onnx")
res = model(root_path + "/data/validation/seq_01/image_02/data/0000000000.png")
boxes = res[0].boxes
print(boxes)

# save model
model.save(root_path + "/models/yolov8n_afterTrain.pt")

