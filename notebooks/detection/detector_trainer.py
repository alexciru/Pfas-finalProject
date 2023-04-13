from ultralytics import YOLO
from pathlib import Path
import sys
import os


"""
Note (elle): I used this to train the pretrained model for just 1 epoch.
Not sure if that did anything in our favor, but it was just as a proof of concept.
"""
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH.
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative.

# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO(ROOT / "yolov8n.pt")

# Train the model using the 'coco128.yaml' dataset
# NOTE: name is subdirectory where model will be saved within 'project' dir
# see https://github.com/ultralytics/ultralytics/issues/512 for more details
results = model.train(
    data="coco128.yaml", epochs=1, project=ROOT, name="runs/detect/train"
)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
print("Exporting model to ONNX format...")
success = model.export(format="onnx")
print("Export succeeded?:", success)

# load saved model -- aka do quick sanity test
model = YOLO(ROOT / "weights/best.onnx")
res = model(ROOT / "src/data/validation/seq_01/image_02/data/0000000000.png")
boxes = res[0].boxes
print(boxes)
