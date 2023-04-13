import cv2
from ultralytics import YOLO
import glob
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH.
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative.

# Load the YOLOv8 model
model = YOLO(ROOT / "runs/detect/train2/weights/best.onnx")

# Open the video file
# video_path = "seq_01/image_02/data/0000000000.png"
# cap = cv2.VideoCapture(video_path)
# glob_path = "seq_02/image_02/data/*.png"
glob_path = "seq_01/image_02/data/0000000000.png"
frame_paths = sorted(glob.glob(glob_path))
# Loop through the video frames
# while cap.isOpened():
for frame_path in frame_paths:
    # Read a frame from the video
    # success, frame = cap.read()
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
