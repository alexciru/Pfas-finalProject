import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image
import glob
from pathlib import Path
import os
import numpy as np

FILE = Path(__file__).resolve()
# ROOT is the Pfas-finalProject git repo
local_parent = FILE.parents[0]
ROOT = local_parent.parents[0].parents[0]  # root directory.

def execute(data_glob=None, model=None):
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
            masks = results[0].masks.data.numpy()
            # rearrange mask dims from (N, H, W) to (H, W, N) for scale_image
            masks = np.moveaxis(masks, 0, -1)
            # rescale masks to original image dims
            # per https://github.com/ultralytics/ultralytics/issues/561
            masks = scale_image(masks, results[0].masks.orig_shape)
            # rearrange masks back to (N, H, W) for visualization
            masks = np.moveaxis(masks, -1, 0)
            pixel_img = np.zeros_like(frame)
            for mask in masks:
                mask = mask.astype(bool)
                pixel_img[mask] = frame[mask]
            cv2.imshow("segmented", pixel_img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_glob = "data/video/seq_01/image_02/data/*.png"
    model = "yolov8s-seg.pt"
    execute(data_glob=data_glob, model=model)
