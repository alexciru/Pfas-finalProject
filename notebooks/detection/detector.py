import cv2
from ultralytics import YOLO
import glob
from pathlib import Path
import os

FILE = Path(__file__).resolve()
# ROOT is the Pfas-finalProject git repo
local_parent = FILE.parents[0]
ROOT = local_parent.parents[0].parents[0]  # root directory.

def execute(data_glob=None, model=None):
    # Load the YOLOv8 model
    model_path = ROOT / model
    model = YOLO(model_path)
    print(f"Loaded model from {model_path}")

    data_glob = os.path.join(ROOT, data_glob)

    frame_paths = sorted(glob.glob(data_glob))
    if not frame_paths:
        raise ValueError(f"No frames found at {data_glob}")

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


if __name__ == "__main__":
    data_glob = "data/video/seq_01/image_02/data/*.png"
    model = "models/yolov8n.pt"
    execute(data_glob=data_glob, model=model)
