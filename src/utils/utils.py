#!/usr/bin/env python
# Created by Jonathan Mikler on 18/April/23
import os, pathlib
import cv2 as cv
import pandas as pd



# root file has .git folder in it
def _get_root_dir():
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

    return _root


ROOT_DIR = _get_root_dir()

DATA_DIR = ROOT_DIR / "data/video_rect"
SEQ_01 = DATA_DIR / "seq_01"
SEQ_02 = DATA_DIR / "seq_02"
SEQ_03 = DATA_DIR / "seq_03"


def get_frames(frame_num_, seq_dir_):
    """
    returns the right and left frames of the given sequence for a given frame number
    :param frame_num_: frame number (int)
    :param seq_dir_: sequence directory (pathlib.Path)
    :return: right and left frame (np.ndarray, np.ndarray)
    """
    _frame_name_r = str(seq_dir_ / "image_02/data" / f"{frame_num_:010d}.png")
    _frame_name_l = str(seq_dir_ / "image_03/data" / f"{frame_num_:010d}.png")

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
