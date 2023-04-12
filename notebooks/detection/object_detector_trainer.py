from ultralytics import YOLO

## Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model using the 'coco128.yaml' dataset
results = model.train(data="coco128.yaml", epochs=1)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")

# load saved model -- aka do quick sanity test
model = YOLO(
    "/Users/ellemcfarlane/Documents/dtu/Perception_AF/Pfas/runs/detect/train2/weights/best.onnx"
)
res = model(
    "/Users/ellemcfarlane/Documents/dtu/Perception_AF/Pfas/final_project/seq_01/image_02/data/0000000000.png"
)
boxes = res[0].boxes
print(boxes)
