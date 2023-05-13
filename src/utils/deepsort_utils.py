#!/usr/bin/env python
# Created by Jonathan Mikler on 05/May/23
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import ultralytics.yolo.utils.ops as yolo_utils
import cv2

LABELS_DICT = {
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
UNKNOWN_DEFAULT = "?"


@dataclass
class DeepSortObject:
    id: int
    cls: str
    confidence: float  # -1 means occluded
    xyxy: List[float]
    mask: np.ndarray

    @property
    def label(self):
        return LABELS_DICT.get(self.cls, UNKNOWN_DEFAULT)

    @property
    def occluded(self) -> bool:
        return self.confidence == -1

    @property
    def tlwh(self):
        """
        Top left corner, width, height representation of bounding box
        """
        top_left_x, top_left_y = self.xyxy[0], self.xyxy[1]
        width = self.xyxy[2] - self.xyxy[0]
        height = self.xyxy[3] - self.xyxy[1]
        return [top_left_x, top_left_y, width, height]

    def __repr__(self) -> str:
        return f"DeepSortObject(id={self.id}, cls={self.cls}, label={self.label}, confidence={self.confidence})"

    def get_avg_mask(self):
        """Return the avg position of the mask in the image"""
        # calculate moments of binary image
        y, x = np.nonzero(self.mask)
        centroid_x = int(np.mean(x))
        centroid_y = int(np.mean(y))
        return (centroid_x, centroid_y)


def resize_masks(masks, orig_shape):
    # rearrange mask dims from (N, H, W) to (H, W, N) for scale_image
    masks = np.moveaxis(masks, 0, -1)
    # rescale masks to original image dims
    # per https://github.com/ultralytics/ultralytics/issues/561
    masks = yolo_utils.scale_image(masks, orig_shape)
    # rearrange masks back to (N, H, W) for visualization
    masks = np.moveaxis(masks, -1, 0)
    return masks


def get_closest_prediction(x, y, ds_pred):
    """Return the closest prediction to the given x,y coordinate"""


def combine_cyclist_clases(cyclist_preds: str, ds_objects: List[DeepSortObject]):
    """Combine pedestrian and bikes into cyclist class"""

    for c in cyclist_preds:
        x, y = c.get_avg_mask()
        closest = get_closest_prediction(x, y, ds_objects)


# writes deepsort tracking id, bounding box, and class to frame
def disp_track(frame, data, color=None, label_offset=0, expected=None):
    frame = frame.copy()
    # TODO (elle): change color of bbox based on track id
    label = f"{data['track_id']}:{data['type'][:3]}"
    if expected is not None:
        label = f"{data['track_id']}/{expected['track_id']}:{data['type'][:3]}/{expected['type']}"
    # baseline is line where letters sit
    font_scale = 0.3
    font_thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    bbox = [
        data["bbox_left"],
        data["bbox_top"],
        data["bbox_right"],
        data["bbox_bottom"],
    ]
    top_left = tuple(
        map(
            int,
            [int(bbox[0]), int(bbox[1]) - (label_height + baseline + label_offset)],
        )
    )
    top_right = tuple(map(int, [int(bbox[0]) + label_width, int(bbox[1])]))
    org = tuple(map(int, [int(bbox[0]), int(bbox[1]) - baseline]))
    default_color = (255, 0, 0)
    # bounding box
    bbox_color = default_color if color is None else color
    cv2.rectangle(
        frame,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        bbox_color,
        1,
    )
    # label
    label_color = default_color if color is None else color
    cv2.rectangle(frame, top_left, top_right, label_color, -1)
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


if __name__ == "__main__":
    pass
