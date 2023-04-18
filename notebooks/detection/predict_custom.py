import cv2
from ultralytics import YOLO
import glob
from pathlib import Path
import sys
import os


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = root_path + "\\data\\external\\customYolov8\\"
MODEL_PATH = root_path + "\\models\\yolov8n.pt"

# load saved model -- aka do quick sanity test
model = YOLO(DATASET_PATH + "weights/best.onnx")
res = model(root_path + "/data/validation/seq_01/image_02/data/0000000000.png")
boxes = res[0].boxes
print(boxes)

glob_path = os.path.join(root_path, "src/data/validation/seq_01/image_02/data/*.png")

frame_paths = sorted(glob.glob(glob_path))
if not frame_paths:
    raise ValueError(f"No frames found at {glob_path}")

for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
    success = frame is not None
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
# cap.release()
cv2.destroyAllWindows()
