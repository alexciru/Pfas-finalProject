#!/usr/bin/env python
# Created by Jonathan Mikler on 18/April/23
import os, pathlib
import cv2 as cv
import pandas as pd
import numpy as np

from typing import Dict


# root file has .git folder in it
def get_root_dir():
    """
    returns the root directory of the project, by checking for the .git folder
    :return: root directory (pathlib.Path)
    """
    _root = pathlib.Path(__file__).parent
    # check if _root has .git folder
    while not os.path.isdir(_root / ".git"):
        _root = _root.parent
        if _root == pathlib.Path("/"):
            raise FileNotFoundError("Could not find root directory")

    print(f"Root directory is {_root}")
    return _root


ROOT_DIR = get_root_dir()

DATA_DIR = ROOT_DIR / "data/video_rect"
SEQ_01 = DATA_DIR / "seq_01"
SEQ_02 = DATA_DIR / "seq_02"
SEQ_03 = DATA_DIR / "seq_03"

class ObjectTracker():
    _trajectory: Dict[int, np.ndarray] # obj.id -> np.ndarray of shape (3) # could be a np.ndarray of shape (t,3)
    def __init__(self):
        self.tracked_objects: Dict[int, Dict] = dict()  # obj.id -> trajectory
        self.tracked_kinematics: Dict[int, np.ndarray] = dict() # obj.id -> deltaX # QUESTION: Maybe we want to keep track of the last K deltas?

    def register_position(self, obj_key_:int, time_:int, position_:np.ndarray):
        self.tracked_objects[obj_key_][time_] = position_

        if time_ > 0:
            self.register_kinematics(obj_key_, time_, position_ - self.tracked_objects[obj_key_][time_ - 1])

    def register_kinematics(self, obj_key_:int, time_:int, dx_:np.ndarray):
        self.tracked_kinematics[obj_key_][time_] = dx_
    
    def predict_position(self, obj_key_:int, before_time_:int) -> np.ndarray:
        """
        Predicts the position of the object at before_time_ + 1
        args:
            obj_key_: int   -- object id
            time_: int      -- time step
        """
        assert obj_key_ in self.tracked_objects.keys(), f"Object {obj_key_} not found in tracker"
        assert before_time_ in self.tracked_objects[obj_key_].keys(), f"Position of object {obj_key_} not found at time {before_time_}"
        assert before_time_ in self.tracked_kinematics[obj_key_].keys(), f"Kinematics of object {obj_key_} not found at time {before_time_}"

        return self.tracked_objects.get(obj_key_).get(before_time_) + self.tracked_kinematics.get(obj_key_).get(before_time_)



def get_frames(frame_num_, seq_dir_):
    """
    returns the right and left frames of the given sequence for a given frame number
    :param frame_num_: frame number (int)
    :param seq_dir_: sequence directory (pathlib.Path)
    :return: right and left frame (np.ndarray, np.ndarray)
    """
    _frame_name_r = str(seq_dir_ / "image_02/data" / f"{frame_num_:06d}.png")
    _frame_name_l = str(seq_dir_ / "image_03/data" / f"{frame_num_:06d}.png")

    if not os.path.isfile(_frame_name_r):
        raise FileNotFoundError(f"File {_frame_name_r} not found")
    if not os.path.isfile(_frame_name_l):
        raise FileNotFoundError(f"File {_frame_name_l} not found")

    return cv.cvtColor(cv.imread(_frame_name_r), cv.COLOR_BGR2RGB), cv.cvtColor(
        cv.imread(_frame_name_l), cv.COLOR_BGR2RGB
    )


def get_labels_df(seq_dir_):
    """
    returns the labels (ground truth data) of the given sequence as a pandas dataframe
    :param seq_dir_: sequence directory (pathlib.Path)
    :return: labels dataframe (pd.DataFrame)
    """

    _labels_file = str(seq_dir_ / "labels.txt")
    headers = [
        "frame",
        "track_id",
        "type",
        "truncated",
        "occluded",
        "alpha",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "height",
        "width",
        "length",
        "x",
        "y",
        "z",
        "yaw",
    ]
    return pd.read_csv(_labels_file, sep=" ", header=None, names=headers)


def draw_bboxes(frame_, bbox_coords_):
    print("REMEMBER: bboxes are with respect to the left frame")
    _frame = frame_.copy()
    for bbox in bbox_coords_:
        cv.rectangle(
            _frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (255, 0, 0),
            2,
        )
    return _frame


if __name__ == "__main__":
    pass
