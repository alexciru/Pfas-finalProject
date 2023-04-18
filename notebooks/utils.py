#!/usr/bin/env python
# Created by Jonathan Mikler on 18/April/23
import os, pathlib, sys
import cv2 as cv
import pandas as pd

ROOT_DIR = pathlib.Path(os.getcwd()).parent
DATA_DIR = ROOT_DIR / "data/ground_truth"
SEQ_01 = DATA_DIR / "seq_01"
SEQ_02 = DATA_DIR / "seq_01"
SEQ_03 = DATA_DIR / "seq_01"


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
    
    return cv.cvtColor(cv.imread(_frame_name_r), cv.COLOR_BGR2RGB), cv.cvtColor(cv.imread(_frame_name_l), cv.COLOR_BGR2RGB)


def get_labels_df(seq_dir_):
    """
    returns the labels (ground truth data) of the given sequence as a pandas dataframe
    :param seq_dir_: sequence directory (pathlib.Path)
    :return: labels dataframe (pd.DataFrame)
    """

    _labels_file = str(seq_dir_ / "labels.txt")
    headers = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'x', 'y', 'z', 'yaw']
    return pd.read_csv(_labels_file, sep=' ', header=None, names=headers)
if __name__ == '__main__':
    pass