from registration import calculate_bounding_box, calculate_rotation_beetween_cluster, get_avg_point_pointCloud
import open3d as o3d
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
        print("location: ", avg_point)
        print("Box location: ", bbox.get_center())

        # convert to left hand coordinates for openCV
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