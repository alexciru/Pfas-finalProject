import open3d as o3d
import numpy as np
from typing import Dict
from pathlib import Path

# own
from utils.ObjectTracker import ObjectTracker
from utils.deepsort_utils import LABELS_DICT, DeepSortObject
from depth.registration import calculate_bounding_box, get_avg_point_pointCloud

"""
Funtion to save the results of the object detection to a file
Format:
    frame_id, track_id, type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rotation_y
"""


def reset_file(filename_: Path):
    """
    Resets the file with the given filename
    """
    with open(filename_, "w") as f:
        f.write("")


def log_info(filename_: Path, text_: str):
    """
    Writes the given text to the given file
    """
    with open(filename_, "a") as f:
        f.write(text_ + "\n")


def new_save_timeframe_results(
    frame_t_: int,
    object_tracker_: ObjectTracker,
    ds_tracked_objects_: Dict[int, DeepSortObject],
    pointcloud_dict_: dict,
    filename: str,
):
    """
    Saves the results of the given timeframe to the given file
    Row format:
        frame_id, _obj_id, label, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rotation_y
    """

    objects_in_t = object_tracker_.objects_in_time[frame_t_]

    for _obj_id, _obj_pos in objects_in_t.items():
        frame_id = frame_t_
        obj_id = _obj_id
        label = LABELS_DICT.get(ds_tracked_objects_[_obj_id].cls)
        truncated = 0  # TODO
        occluded = ds_tracked_objects_[_obj_id].cls
        alpha = 0  # TODO
        bbox_left = ds_tracked_objects_[_obj_id].xyxy[0]
        bbox_top = ds_tracked_objects_[_obj_id].xyxy[1]
        bbox_right = ds_tracked_objects_[_obj_id].xyxy[2]
        bbox_bottom = ds_tracked_objects_[_obj_id].xyxy[3]

        height = 0  # TODO
        width = 0  # TODO
        length = 0  # TODO

        x = _obj_pos[0]
        y = _obj_pos[1]
        z = _obj_pos[2]

        # If we find the object obtain box dimensions
        possible_points = pointcloud_dict_.get(_obj_id)
        if possible_points != None:
            bbox = calculate_bounding_box(possible_points)
            max_bound = bbox.get_max_bound()
            min_bound = bbox.get_min_bound()
            width = -(max_bound[0] - min_bound[0])
            height = -(max_bound[1] - min_bound[1])
            length = -(max_bound[2] - min_bound[2])

            # offset the location to be the center of the bottom box
            # get point from bbox
            box_corners = bbox.get_box_points()
            lower_corners = np.asarray(
                [box_corners[0], box_corners[1], box_corners[3], box_corners[6]]
            )
            # compute the center point of the lower plane vertices
            lower_center = np.mean(lower_corners, axis=0)

            # create bbox from cluster
            # convert to left hand coordinates for openCV
            x = lower_center[0]
            y = lower_center[1]
            z = lower_center[2]

        location = _obj_pos
        try:
            prev_location = object_tracker_.get_object_trajectory(_obj_id)[frame_t_ - 1]
            _rotation_y = np.arctan2(
                location[2] - prev_location[2], location[0] - prev_location[0]
            )
        except:
            print(
                f"rotation for object {_obj_id} in frame {frame_t_} could not be calculated"
            )
            _rotation_y = 0
        rotation_y = _rotation_y
        score = ds_tracked_objects_[_obj_id].confidence

        row = [
            frame_id,
            obj_id,
            label,
            truncated,
            occluded,
            alpha,
            bbox_left,
            bbox_top,
            bbox_right,
            bbox_bottom,
            height,
            width,
            length,
            x,
            y,
            z,
            rotation_y,
            score,
        ]

        # Write the formatted data to file
        with open(filename, "a") as f:
            formatted_data = " ".join(str(value) for value in row) + "\n"
            f.write(formatted_data)

    return


def save_timeframe_results(
    frame_t_: int,
    ds_tracked_objects_: dict,
    pointcloud_dict_: dict,
    object_tracker_: ObjectTracker,
    filename: str,
):
    """
    Saves the results of the given timeframe to the given file
    """

    for _ds_obj_id, _ds_obj in ds_tracked_objects_.items():
        row = []

        # 2D bounding box of object in the image
        # Obtain from YoloInDeepSort

        bbox_left = _ds_obj.xyxy[0]
        bbox_top = _ds_obj.xyxy[1]
        bbox_right = _ds_obj.xyxy[2]
        bbox_bottom = _ds_obj.xyxy[3]

        # get pointcloud ``
        obj_pcd = pointcloud_dict_.get(_ds_obj_id)

        # bbox = calculate_bounding_box(obj_pcd)

        # max_bound = bbox.get_max_bound()
        # min_bound = bbox.get_min_bound()

        # Calculate the dimensions
        # width = -1* (max_bound[0] - min_bound[0])
        # height = -1* (max_bound[1] - min_bound[1])
        # length = -1* (max_bound[2] - min_bound[2])

        # dimensions in camera coordinates
        # center location in camera coordinates
        # obtain from cluter, make a box and get the center

        # get rotation vector from previous location
        location = object_tracker_.get_object_trajectory(_ds_obj_id)[frame_t_]
        prev_location = object_tracker_.get_object_trajectory(_ds_obj_id)[frame_t_ - 1]
        try:
            rotation_y = np.arctan2(
                location[2] - prev_location[2], location[0] - prev_location[0]
            )
        except:
            print(
                f"rotation for object {_ds_obj_id} in frame {frame_t_} could not be calculated"
            )
            rotation_y = 0

        # row format
        # frame_id, track_id, label, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rotation_y
        row.append(frame_t_)  # frame_id
        row.append(_ds_obj_id)  # track_id
        row.append(LABELS_DICT.get(_ds_obj.cls))  # label
        row.append(0)  # TODO: truncated pending
        row.append(_ds_obj.occluded)  # occluded
        row.append(0)  # TODO: alpha pending
        row.append(bbox_left)  # 2D bbox left
        row.append(bbox_top)  # 2D bbox top
        row.append(bbox_right)  # 2D bbox right
        row.append(bbox_bottom)  # 2D bbox bottom
        row.append(0)  # 3D bbox height # TODO Pending
        row.append(0)  # 3D bbox width # TODO Pending
        row.append(0)  # 3D bbox length # TODO Pending
        row.append(location[0])  # x
        row.append(location[1])  # y
        row.append(location[2])  # z
        row.append(rotation_y)  # rotation_y
        row.append(_ds_obj.confidence)  # score

        formatted_data = " ".join(str(value) for value in row) + "\n"

        # Write the formatted data to file
        with open(filename, "a") as f:
            f.write(formatted_data)
