from final_project.src.depth.registration import calculate_bounding_box, get_avg_point_pointCloud
import open3d as o3d
import numpy as np
"""
Funtion to save the results of the object detection to a file
Format:
    frame_id, track_id, type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rotation_y
"""



def write_results_to_file(frame_id, ds_track_objects, cluster_dicc, tracker,  filename = "results.txt"):
    
    # Get the data from the pointcloud1 
    # pointcloud2 only use for calculating the vector beetween frames
    # and the translation vector

    # get all the data from the deepSortObject in frame f
    
    for _obj_id, _obj in ds_track_objects.items():
        row = []

        track_id = ds_track_objects.id 
        otype = ds_track_objects.cls
        truncated = 0 # TODO:  change this
        occluded =  _obj.ocluded() == -1
        alpha = 0 # TODO:  change this

        # 2D bounding box of object in the image
        # Obtain from YoloInDeepSort
        _obj.xyxy.tolist()[0]
        bbox_left = 0 
        bbox_top = 0 
        bbox_right = 0 
        bbox_bottom = 0 

        # get fetch pointcloud
        cluster = cluster_dicc[_obj_id]
        bbox = calculate_bounding_box(cluster)

        max_bound = bbox.get_max_bound()
        min_bound = bbox.get_min_bound()

        # Calculate the dimensions
        width = max_bound[0] - min_bound[0]
        height = max_bound[1] - min_bound[1]
        length = max_bound[2] - min_bound[2]
        # dimensions in camera coordinates

        # center location in camera coordinates
        # obtain from cluter, make a box and get the center
        avg_point = get_avg_point_pointCloud(cluster)
        box_corners = bbox.get_box_points()
        lower_corners = box_corners[0:4]
        # compute the center point of the lower plane vertices
        lower_center = np.mean(lower_corners, axis=0)

        # create bbox from cluster
        # convert to left hand coordinates for openCV
        x = lower_center[0]
        y = lower_center[1]
        z = lower_center[2]

        rotation_y = 0 # TODO: change this

        # get rotation vector from previous location
        location = tracker.get_object_trajectory(frame_id, track_id)
        prev_location = tracker.get_object_trajectory(frame_id - 1, track_id)
        score = _obj.confidence 
        # Obtain from clusterList2 with same index and obtain rotation of vector
        
        
        # Obtain from DeepSort
        
        row.append(frame_id)
        row.append(track_id)
        row.append(otype)
        row.append(truncated)
        row.append(occluded)
        row.append(alpha)
        row.append(bbox_left)
        row.append(bbox_top)
        row.append(bbox_right)
        row.append(bbox_bottom)
        row.append(height)
        row.append(width)
        row.append(length)
        row.append(x)
        row.append(y)
        row.append(z)
        row.append(rotation_y)
        row.append(score)

        formatted_data = ' '.join(str(value) for value in row) + '\n'

        # Write the formatted data to file
        with open(filename, 'a') as f:
            f.write(formatted_data)