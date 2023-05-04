from registration import calculate_bounding_box, calculate_rotation_beetween_cluster, get_avg_point_pointCloud
from DephtStimation import get_Q_matrix
import open3d as o3d
import numpy as np
import cv2
"""
Funtion to save the results of the object detection to a file
Format:
    frame_id, track_id, type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rotation_y
"""



def write_results_to_file(frame_id, DeepSortId, clusterlist1_list, cluster2_list, filename = "results.txt"):
    
    # Get the data from the pointcloud1 
    # pointcloud2 only use for calculating the vector beetween frames
    # and the translation vector
    

    frame = frame_id
    for i in range(2):
        row = []

        cluster = clusterlist1_list[i]

        track_id = 1 # TODO:  change this
        otype = "Car"# TODO:  change this
        truncated = 0 # TODO:  change this
        occluded = 0 # TODO:  change this
        alpha = 0 # TODO:  change this

        # 2D bounding box of object in the image
        # Obtain from YoloInDeepSort
        bbox_left = 0 
        bbox_top = 0 
        bbox_right = 0 
        bbox_bottom = 0 

        # 3D object dimensions: height, width, length (in meters)
        # Obtain from cluter
        min_x, min_y, min_z = np.min(cluster, axis=0)
        max_x, max_y, max_z = np.max(cluster, axis=0)

        width = max_x - min_x
        height = max_y - min_y
        length = max_z - min_z

        # dimensions in camera coordinates

        # center location in camera coordinates
        # obtain from cluter, make a box and get the center
        avg_point = get_avg_point_pointCloud(cluster)
        print("location: ", avg_point)

        # transform to world coordinates with R1 and T1
        
        
        # transform camera coordinates to world coordinates
        ex_mat = np.array([[9.999838e-01,-5.012736e-03, -2.710741e-03, 5.989688e-02],
                        [5.002007e-03,9.999797e-01,-3.950381e-03,-1.367835e-03],
                        [2.730489e-03,3.936758e-03,9.999885e-01, 4.637624e-03],
                        [0,0,0,1]])
        Q = np_mat = np.array([[   1.        ,    0.        ,    0.        , -604.08142853],
                                [   0.        ,    1.        ,    0.        , -180.50659943],
                                [   0.        ,    0.        ,    0.        ,  707.04931641],
                                [   0.        ,    0.        ,   -1.85185185,   0.0]])
        # transform camera coordinates to world coordinates
        avg_point = np.append(avg_point, 1)
        points_world = np.matmul(ex_mat, avg_point)

        # convert world coordinates to camera coordinates
        #points_world = np.append(avg_point, 1)
        #points_camera = np.matmul(np.linalg.inv(ex_mat), points_world)


        x = avg_point[0]
        y = avg_point[1]
        z = avg_point[2]


        # Obtain from clusterList2 with same index and obtain rotation of vector
        rotation_y = calculate_rotation_beetween_cluster(cluster, cluster2_list[i])
        
        # Obtain from DeepSort
        score = 69.420 

        row.append(frame)
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