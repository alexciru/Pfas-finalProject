import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO
import math

FILE = Path(__file__).resolve()
# ROOT is the Pfas-finalProject git repo
local_parent = FILE.parents[0]
ROOT = local_parent.parents[0].parents[0]  # root directory.

# from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

import cv2
import glob


# writes deepsort tracking id, bounding box, and class to frame
def disp_track(frame, data):
    frame = frame.copy()
    # TODO (elle): change color of bbox based on track id
    label = f"{data['id']}:{data['class'][:3]}"
    # baseline is line where letters sit
    font_scale = 0.5
    font_thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    bbox = [data["x1"], data["y1"], data["x2"], data["y2"]]
    top_left = tuple(
        map(
            int,
            [int(bbox[0]), int(bbox[1]) - (label_height + baseline)],
        )
    )
    top_right = tuple(map(int, [int(bbox[0]) + label_width, int(bbox[1])]))
    org = tuple(map(int, [int(bbox[0]), int(bbox[1]) - baseline]))

    # bounding box
    cv2.rectangle(
        frame,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (255, 0, 0),
        1,
    )
    # label
    cv2.rectangle(frame, top_left, top_right, (255, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )
    return frame


# gets value of closest key to bbox
# we have to do this since the bbox from deepsort does not exactly match the bbox from yolo
# even when rounding to nearest int
# and the tracked objects are not necessarily in the detection order, so we also can't use order idx to match
def get_class_data(coords2classdata, bbox):
    _bbox, class_data = min(
        coords2classdata.items(), key=lambda x: math.dist(bbox, x[0])
    )
    return class_data


def execute(data_glob=None, model=None):
    # deepsort
    # TODO (elle): how to tune params?
    max_cosine_distance = 0.4
    nn_budget = None
    model_filename = local_parent / "networks/mars-small128.pb"
    model = ROOT / model

    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric, n_init=0)
    detector = YOLO(model)

    if data_glob is None:
        print("No frame path pattern provided, exiting")
        return

    frame_paths = sorted(glob.glob(data_glob))
    if len(frame_paths) == 0:
        print(f"No frames found at {data_glob}, exiting")
        return

    # TODO (elle): hardcoding bc can't find programatic way to get
    # supposedly detector.names should work but it returns None
    # and I can see names in detector_pred object but can't access it via detector_pred.names
    # can try to debug with detector.__dict__ and dir(detector)
    # also, I am overriding person->Pedestrian, car->Car, and bicycle->Cyclist to match labels.txt
    val2class = {
        0: "Pedestrian",
        1: "Cyclist",
        2: "Car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }

    for frame_idx, path in enumerate(frame_paths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"image not found at {path}")
            exit(1)

        # TODO: filter by confidence?
        # detect only person, car, and bicycle
        detector_pred = detector(frame, classes=[0, 1, 2])
        bboxes = []
        coords2classdata = {}
        confs = []
        if detector_pred:
            boxes = detector_pred[0].boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls_val = int(box.cls[0])
                cls = val2class[cls_val]
                # box.xyxy is tlbr (top-left, bottom-right)
                # convert from tensor to list
                box = box.xyxy.tolist()[0]
                coords2classdata[tuple(box)] = (cls, conf)
                confs.append(conf)
                top_left_x, top_left_y = box[0], box[1]
                width = box[2] - box[0]
                height = box[3] - box[1]
                bbox = [top_left_x, top_left_y, width, height]
                bboxes.append(bbox)

        # get appearance features of the object.
        features = encoder(frame, bboxes)
        # get all the required info in a list.
        detections = [
            Detection(bbox, conf, feature)
            for bbox, conf, feature in zip(bboxes, confs, features)
        ]
        # predict tracks
        tracker.predict()
        tracker.update(detections)
        print()
        # get track info (bounding boxes, etc)
        for track in tracker.tracks:
            # track can be tentative (recently created, needs more evidence aka associations in n_init+1 frames),
            # confirmed (associated for n_init+1 or more frames), or deleted (no longer tracked)
            # a new object is classified as tentative in the first n_init frames
            # https://github.com/nwojke/deep_sort/issues/48
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # change track bbox to top left, bottom right coordinates.
            bbox = list(track.to_tlbr())
            class_data = get_class_data(coords2classdata, bbox)
            cls, conf = class_data
            # TODO (elle): calculate actual x,y,z values instead of hardcoding -1's
            data = {
                "frame_idx": frame_idx,
                "id": track.track_id,
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3]),
                "3Dx": -1,
                "3Dy": -1,
                "3Dz": -1,
                "class": cls,
                "conf": conf,
            }
            print(f"cls: {cls}, box: {bbox}")
            frame = disp_track(frame, data)

        cv2.imshow("YOLOv8 Inference", frame)
        # break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    data_glob = "data/video/seq_01/image_02/data/*.png"
    model = "models/yolov8n.pt"
    execute(data_glob=data_glob, model=model)
