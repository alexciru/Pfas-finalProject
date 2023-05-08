import cv2
from ultralytics import YOLO
import glob
from pathlib import Path
import os
import torch
#from utils.utils import ROOT_DIR, DATA_DIR, get_frames

def execute(data_glob=None, model_path=None):
    # Load the YOLOv8 model
    #model = YOLO(model_path)
    model = torch.load('/home/hackercosmos/DTU/PFAS/Pfas-finalProject/models/yolo/new_yolo.pt')
    print(f"Loaded model from {model_path}")

    data_glob = os.path.join("/home/hackercosmos/DTU/PFAS/Pfas-finalProject", data_glob)

    frame_paths = sorted(glob.glob(data_glob))
    if not frame_paths:
        raise ValueError(f"No frames found at {data_glob}")
    
    i = 0
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        success = frame is not None
        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame)

            # Visualize the results on the frame
            # Save the prediction as an image
            for r in results:
                object_class = r.json_prediction["class"]
                if object_class == 'Pedestrian':
                    center_x = r.json_prediction["x"]
                    center_y = r.json_prediction["y"]
                    print(center_x)
                    print(center_y)
                    hh = r.json_prediction["height"]
                    ww = r.json_prediction["width"]
                    top_x = center_x - (ww/2)
                    bottom_x = center_x + (ww/2)
                    top_y = center_y - (hh/2)
                    bottom_y = center_y + (hh/2)
                    cv2.rectangle(frame, (int(top_x), int(top_y)) ,(int(bottom_x), int(bottom_y)), (0, 255, 0), 2)
                    cv2.putText(frame, str(object_class), (int(top_x), int(top_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    #cv2.imshow("YOLOv8 Inference", frame)
                if object_class == 'Cyclist':
                    center_x = r.json_prediction["x"]
                    center_y = r.json_prediction["y"]
                    print(center_x)
                    print(center_y)
                    hh = r.json_prediction["height"]
                    ww = r.json_prediction["width"]
                    top_x = center_x - (ww/2)
                    bottom_x = center_x + (ww/2)
                    top_y = center_y - (hh/2)
                    bottom_y = center_y + (hh/2)
                    cv2.rectangle(frame, (int(top_x), int(top_y)) ,(int(bottom_x), int(bottom_y)), (0, 255, 0), 2)
                    cv2.putText(frame, str(object_class), (int(top_x), int(top_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    #cv2.imshow("YOLOv8 Inference", frame)
                if object_class == 'Car':
                    center_x = r.json_prediction["x"]
                    center_y = r.json_prediction["y"]
                    print(center_x)
                    print(center_y)
                    hh = r.json_prediction["height"]
                    ww = r.json_prediction["width"]
                    top_x = center_x - (ww/2)
                    bottom_x = center_x + (ww/2)
                    top_y = center_y - (hh/2)
                    bottom_y = center_y + (hh/2)
                    cv2.rectangle(frame, (int(top_x), int(top_y)) ,(int(bottom_x), int(bottom_y)), (0, 255, 0), 2)
                    cv2.putText(frame, str(object_class), (int(top_x), int(top_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    #cv2.imshow("YOLOv8 Inference", frame)
            cv2.imshow("YOLOv8 Inference", frame)
            cv2.waitKey(0)

            #annotated_frame = results[0].plot()

            # Display the annotated frame
            #cv2.imshow("YOLOv8 Inference", annotated_frame)

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
    data_glob = "/home/hackercosmos/DTU/PFAS/Pfas-finalProject/data/video/seq_01/image_02/data/*.png"
    model_path = "/home/hackercosmos/DTU/PFAS/Pfas-finalProject/models/yolo/new_yolo.pt"

    execute(data_glob=data_glob, model_path=model_path)
