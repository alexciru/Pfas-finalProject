import open3d as o3d
import numpy as np
from pathlib import Path

# own
from utils.ObjectTracker import ObjectTracker
from utils.deepsort_utils import LABELS_DICT
from depth.registration import calculate_bounding_box, get_avg_point_pointCloud
"""
Funtion to save the results of the object detection to a file
Format:
    frame_id, track_id, type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rotation_y
"""
def reset_results_file(filename_:Path):
    """
    Resets the file with the given filename
    """
    with open(filename_, 'w') as f:
        f.write("")

def save_timeframe_results(frame_t_:int, ds_tracked_objects_:dict, pointcloud_dict_:dict, object_tracker_:ObjectTracker,  filename:str):
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
        prev_location = object_tracker_.get_object_trajectory(_ds_obj_id)[frame_t_-1]
        try:
            rotation_y = np.arctan2(location[2] - prev_location[2], location[0] - prev_location[0])
        except:
            print(f"rotation for object {_ds_obj_id} in frame {frame_t_} could not be calculated")
            rotation_y = 0
        
        # row format
        # frame_id, track_id, label, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rotation_y
        row.append(frame_t_) # frame_id
        row.append(_ds_obj_id) # track_id
        row.append(LABELS_DICT.get(_ds_obj.cls)) # label
        row.append(0) # TODO: truncated pending
        row.append(_ds_obj.occluded) # occluded
        row.append(0) #TODO: alpha pending
        row.append(bbox_left) # 2D bbox left
        row.append(bbox_top) # 2D bbox top
        row.append(bbox_right) # 2D bbox right
        row.append(bbox_bottom) # 2D bbox bottom
        row.append(0) # 3D bbox height # TODO Pending
        row.append(0) # 3D bbox width # TODO Pending
        row.append(0) # 3D bbox length # TODO Pending
        row.append(location[0]) # x
        row.append(location[1]) # y
        row.append(location[2]) # z
        row.append(rotation_y) # rotation_y
        row.append(_ds_obj.confidence ) # score

        formatted_data = ' '.join(str(value) for value in row) + '\n'

        # Write the formatted data to file
        with open(filename, 'a') as f:
            f.write(formatted_data)